# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Per-step sub-patch jitter for the tiled rollout.

The 4-px patch kink accumulates over a tiled rollout because the patchify grid
is anchored to the same pixels of every padded tile at every step: each step
injects a period-4 pattern at the same phase, so the k = n/4 component adds
coherently (~N). ``JitteredModel`` wraps a ``ScreamcastModel`` and, at every
step, runs the network on an ``(H-J) x (W-J)`` sub-window of each padded tile
taken at a sub-patch offset ``(ox, oy)`` in ``{0..J-1}^2`` (with ``J`` = the
patch size, the window stays a multiple of the patch). Shifting the window by
``phi`` px rotates the k = n/4 phasor by ``pi*phi/2``. A 2-D component at
``(kx, ky)`` has phase ``~ kx*ox + ky*oy``, so the offsets must cancel four
things at once: the two axis lines, the ``(+k, +k)`` corner (``~ ox + oy``)
and the ``(+k, -k)`` anti-diagonal corner (``~ ox - oy``). The diagonal cycle
``ox = oy = t mod J`` (``mode="cycle"``) cancels the axes and the ``(+,+)``
corner exactly but keeps ``ox - oy = 0``, so the anti-diagonal corner is never
rotated and accumulates as 45-degree stripes (base-adaLN, 12 h: anti-corner
674x background vs 1034x un-jittered, while the axis metric read a "clean"
1.09 -- the corners sit at radius k*sqrt(2), outside an axis-only k = n/4
measure). The default ``mode="cycle16"`` visits all J*J offsets (``ox = t mod
J``, ``oy = (t // J) mod J``) and cancels all four exactly every J*J steps.
``mode="random"`` has no blind spot but only reaches the ~1/sqrt(N) random-walk
floor; a 2-phase {0,1} set cannot cancel the fundamental at all.

The shift is global per step (the same offset for every tile in the batch):
each 128 core is predicted by exactly one tile, and with a shared offset the
two tiles meeting in a halo band carry the same phase, so the overlap-add stays
consistent across seams. The un-sampled J-px border of each tile is filled by
replicating the sub-window edge (``fill="replicate"``) rather than zeroed: the
halo adjoint normalises by a fixed window weight, so a zero VALUE with non-zero
weight would bias the blend at the tile edge. Measured on a local 4x160 tiled
12 h rollout: softbias 2.46 -> 1.15, base-adaLN 12.90 -> 1.09 (k = 32
prominence at the kink tile), with no change to the large-scale field.

Tensor convention: ``x`` is ``[batch, channels, faces, H, W]`` with ``H`` the
``coords["x"]`` axis and ``W`` the ``coords["y"]`` axis (the SCREAM catalog
returns ``[..., x, y]``), so ``ox`` slices dim -2 and ``oy`` slices dim -1.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

MODES = ("cycle", "random", "cycle16")
FILLS = ("replicate", "zero")


class JitteredModel:
    """Wrap a prognostic model so each step runs on a sub-patch-shifted window."""

    def __init__(
        self,
        model,
        jitter: int = 4,
        mode: str = "cycle16",
        seed: int = 0,
        fill: str = "replicate",
    ) -> None:
        if jitter < 0:
            raise ValueError(f"jitter must be >= 0, got {jitter}")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if fill not in FILLS:
            raise ValueError(f"fill must be one of {FILLS}, got {fill!r}")
        # Set the inner model FIRST: __getattr__ delegates to it, and would
        # recurse on a missing attribute before it exists.
        self.model = model
        self.jitter = int(jitter)
        self.mode = mode
        self.fill = fill
        self.rng = np.random.default_rng(seed)
        self.step = 0
        self.last_shift: tuple[int, int] | None = None
        self._net_size: tuple[int, int] | None = None

    def __getattr__(self, name):
        # Only reached when normal lookup fails: delegate everything else
        # (dt, output_coords, input_coords, set_tile_size, ...) to the model.
        return getattr(self.model, name)

    def _ensure_network_size(self, sh: int, sw: int) -> None:
        """Register the sub-window size with the network, once.

        The DiT trunk asserts its input spatial shape against the size set by
        ``set_tile_size`` (``strata_sharded.forward_tokens``), and the rollout
        leaves that at the padded tile (160), so the ``(H-J) x (W-J)``
        sub-window must be registered before the first jittered forward.
        Cached because ``set_tile_size`` rebuilds the RoPE buffers. Models
        without a ``pipeline.network`` (e.g. test mocks) are left alone.
        """
        if self._net_size == (sh, sw):
            return
        net = getattr(getattr(self.model, "pipeline", None), "network", None)
        if net is not None and hasattr(net, "set_tile_size"):
            net.set_tile_size(sh, sw)
        self._net_size = (sh, sw)

    def next_shift(self) -> tuple[int, int]:
        """Draw the (ox, oy) offset for the current step and advance the step."""
        t = self.step
        self.step += 1
        j = self.jitter
        if self.mode == "cycle":
            s = t % j
            return s, s
        if self.mode == "cycle16":
            return t % j, (t // j) % j
        return int(self.rng.integers(0, j)), int(self.rng.integers(0, j))

    def __call__(self, x: torch.Tensor, coords: OrderedDict):
        if self.jitter == 0:
            return self.model(x, coords)
        H, W = x.shape[-2], x.shape[-1]
        j = self.jitter
        sh, sw = H - j, W - j
        if sh <= 0 or sw <= 0:
            raise ValueError(f"jitter {j} too large for tile {H}x{W}")
        ox, oy = self.next_shift()
        self.last_shift = (ox, oy)
        self._ensure_network_size(sh, sw)

        x_sub = x[..., ox : ox + sh, oy : oy + sw]
        c_sub = OrderedDict(coords)
        c_sub["x"] = np.asarray(coords["x"])[ox : ox + sh]
        c_sub["y"] = np.asarray(coords["y"])[oy : oy + sw]
        out_sub, _ = self.model(x_sub, c_sub)  # [b, c_out, f, sh, sw]

        b, c_out, f = out_sub.shape[:3]
        if self.fill == "zero":
            out = torch.zeros(
                b, c_out, f, H, W, device=out_sub.device, dtype=out_sub.dtype
            )
            out[..., ox : ox + sh, oy : oy + sw] = out_sub
        else:
            # F.pad pad order is (W_left, W_right, H_top, H_bottom); replicate
            # padding needs a 4-D input, so fold (b, c_out, f) into one dim.
            o4 = out_sub.reshape(b * c_out * f, 1, sh, sw)
            o4 = F.pad(o4, (oy, W - oy - sw, ox, H - ox - sh), mode="replicate")
            out = o4.reshape(b, c_out, f, H, W)
        # Output coords for the FULL tile (lead_time advanced by the model).
        return out, self.model.output_coords(coords)
