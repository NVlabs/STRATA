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
"""Dealiased 3D patch embedding.

Moved verbatim from ``dit_3d.py``. State-dict compatible with the plain
``PatchEmbed3D`` (same ``proj`` conv shape; the dealias filter is a
non-persistent buffer), so it can be swapped in for the physicsnemo Strata
``patch_embed`` after construction without changing checkpoint keys.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DealiasedPatchEmbed3D(nn.Module):
    """PatchEmbed3D variant that adds shift-invariance via a fixed dealiasing filter.

    Stage 1: stride-1 *valid* conv (padding=0).
      stage1[j] = conv(x[j : j+patch_size]) — purely from real input, no zeros.
      In particular stage1[0] == PatchEmbed3D token 0 exactly.

    Stage 2: per-axis low-pass filter + strided decimation.
      The low-pass is applied only along axes that are actually being decimated
      (axes with ``patch_size > 1``). Axes with ``patch_size == 1`` use a
      length-1 identity filter, so no information is mixed across them — this
      matters for vertical patching of size 1 where adjacent atmospheric levels
      should not bleed into each other.

      Per-axis pad totals ``(k_axis - 1)``, split symmetrically: ``floor((k-1)/2)``
      on the near side and the remainder on the far side. Replicate-padded so no
      zeros are introduced and the filter sits on-grid (no half-filter phase shift).

    Token-count proof per axis (S divisible by p, axis filter length k_axis):
      stage1 size  = S - p + 1
      after pad    = S - p + k_axis      (total pad = k_axis - 1)
      stage2 size  = floor((S - p + k_axis - k_axis) / p) + 1 = S // p  ✓

    proj weights are Conv3d-shape-compatible with PatchEmbed3D (same kernel,
    same in/out channels), but the effective output differs on any axis with
    ``patch_size > 1`` because stage 2 adds the low-pass on those axes.
    """

    def __init__(
        self,
        depth,
        height,
        width,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        resample_filter=(1, 1),
    ):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        pd, ph, pw = patch_size
        if depth % pd != 0:
            raise ValueError(
                f"Depth ({depth}) must be divisible by vertical patch size ({pd})"
            )
        if height % ph != 0:
            raise ValueError(
                f"Height ({height}) must be divisible by horizontal patch size ({ph})"
            )
        if width % pw != 0:
            raise ValueError(
                f"Width ({width}) must be divisible by horizontal patch size ({pw})"
            )
        self.depth = depth
        self.height = height
        self.width = width
        self.patch_size = patch_size
        self.num_patches = (depth // pd) * (height // ph) * (width // pw)

        # Stage 1: valid conv (no padding) so every feature is computed from real data
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=1,
            padding=0,
            bias=True,
        )

        # Stage 2: per-axis low-pass filter. Use `resample_filter` only on axes
        # that are actually decimated (patch_size > 1); on stride-1 axes there
        # is no aliasing to prevent, so a length-1 identity filter is used to
        # avoid mixing data across that axis.
        f_active = torch.as_tensor(resample_filter, dtype=torch.float32)
        identity = torch.tensor([1.0])
        fd = identity if pd == 1 else f_active
        fh = identity if ph == 1 else f_active
        fw = identity if pw == 1 else f_active
        f3d = fd[:, None, None] * fh[None, :, None] * fw[None, None, :]
        f3d = (f3d / (fd.sum() * fh.sum() * fw.sum())).unsqueeze(0).unsqueeze(1)
        # persistent=False: exclude from state_dict so checkpoints saved with
        # PatchEmbed3D (which has no dealias_filter) load cleanly under strict=True.
        # The filter is a fixed constant recomputed from resample_filter at init.
        self.register_buffer("dealias_filter", f3d, persistent=False)
        self._embed_dim = embed_dim

    def forward(self, x):
        # Stage 1: valid conv — stage1[0] == PatchEmbed3D token 0, no zero contamination
        x = self.proj(x)  # (B, embed_dim, D-pd+1, H-ph+1, W-pw+1)

        f = self.dealias_filter.to(x.dtype)
        _, _, kd, kh, kw = f.shape

        # Per-axis symmetric pad totaling (k_axis - 1). For axes with k_axis=1
        # this is zero pad. For odd (k_axis - 1) the extra slot goes on the far
        # side (PyTorch 'same'-style convention).
        pad_d_lo, pad_d_hi = (kd - 1) // 2, (kd - 1) - (kd - 1) // 2
        pad_h_lo, pad_h_hi = (kh - 1) // 2, (kh - 1) - (kh - 1) // 2
        pad_w_lo, pad_w_hi = (kw - 1) // 2, (kw - 1) - (kw - 1) // 2
        if kd > 1 or kh > 1 or kw > 1:
            # F.pad order: (W_left, W_right, H_left, H_right, D_left, D_right)
            x = F.pad(
                x,
                (pad_w_lo, pad_w_hi, pad_h_lo, pad_h_hi, pad_d_lo, pad_d_hi),
                mode="replicate",
            )

        x = F.conv3d(
            x,
            f.tile([self._embed_dim, 1, 1, 1, 1]),
            groups=self._embed_dim,
            stride=self.patch_size,
            padding=0,
        )
        return x
