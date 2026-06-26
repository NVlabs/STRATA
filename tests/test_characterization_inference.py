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
"""Characterization (golden) tests that guard production-model inference across
the OSRB cleanup.

Rationale
---------
The cleanup deletes a large amount of dead code (UNet, unused DiT options,
nudging, dual-norm, pooled-loss, fixed-Z, ...). None of it is reachable from the
production configs, so production inference *must* be byte-for-byte unchanged.
These tests prove that.

For each production config we build a **small stand-in** of the real
architecture: the exact ``DiTConfig`` / ``PixelDiTConfig`` flags from
``train_configs.CONFIGS[name]`` (so every production code path is exercised),
with only the sizes shrunk (``embed_dim``/``n_layers``/``tile``/``depth``) so the
test is fast. The model is built through the real ``train.DiT3D`` /
``train.DiT3DPixel`` builders, so it tracks the production construction path.

Two subtleties make the guard reliable:

* **Init-order independence.** Removing a default-off option (e.g.
  ``do_absolute_pos_embedding``) changes how many sub-modules ``__init__``
  creates, which would shift the global RNG sequence and perturb a seeded random
  init — a *false* failure. We therefore fill every parameter deterministically
  *by sorted name* after construction, so identical (name, shape) param sets
  produce identical weights regardless of construction order. Only a genuine
  change to the forward math can move the output.
* **fp32 / eval / fixed input.** Deterministic forward, no bf16, no dropout.

Usage
-----
Requires a GPU (the DiT stack uses transformer_engine + natten). The golden
tensors are **local/transient and git-ignored** — they are binaries, which this
repo must not check in (see ``AGENTS.md``). Regenerate them at the start of a
cleanup session, **before** the first deletion::

    source ~/Documents/python_envs/screamcast_env/bin/activate
    SCREAMCAST_REGEN_GOLDEN=1 pytest tests/test_characterization_inference.py -q

Then, after each cleanup step, re-run *without* the env var; any change to
production inference fails the test with the max abs diff. (For a permanent,
committable CI guard, swap the binary reference for a text fingerprint via
``pytest-regtest``, which the repo already depends on.)
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pytest
import torch

REF_DIR = Path(__file__).parent / "regression_data"
REGEN = os.environ.get("SCREAMCAST_REGEN_GOLDEN") == "1"
ATOL, RTOL = 1e-4, 1e-4

# The production configs we must not perturb. dit3d + pixeldit both production.
# NB: the dit3d entry uses the registered `..._r3_cos` key — the bare `..._r3`
# name is the pre-training checkpoint it resumes from (identical _SWEEP1_BASE
# architecture), not a CONFIGS key.
PRODUCTION_CONFIGS = [
    "sweep1_nodilation_gated_tile64_kernel3_lr1em4_dim1024_hpatch4_depth32_r3_cos",
    "pixeldit_sem1024d24l_pix128d4l_3src_lr5e5cos_qvfix",
    "pixeldit_sem1024d24l_pix128d4l_bilineardwgeluproject_unfreeze_3src_const1em5_t128",
]

# Shrunk sizes — small/fast but large enough to satisfy patch/kernel/dilation
# constraints. tile=64, depth=12 mirror the shapes already exercised by
# test_dit_3d.py, so the production patch_size_horiz / attn_kernel values build.
SMALL_EMBED, SMALL_LAYERS, SMALL_HEADS = 64, 2, 4
SMALL_PIXEL_EMBED, SMALL_PIXEL_LAYERS, SMALL_PIXEL_HEADS = 32, 2, 2
TILE, DEPTH, NSIDE, CH = 64, 12, 512, 6

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="DiT stack requires CUDA (transformer_engine + natten)",
)


def _fill_params_deterministically(model: torch.nn.Module) -> torch.nn.Module:
    """Overwrite every parameter with a deterministic value keyed by sorted name.

    Independent of construction order, so adding/removing default-off options
    does not change the result. Buffers (RoPE tables, etc.) are left untouched —
    they are derived from geometry and must stay as the model computed them.
    """
    with torch.no_grad():
        for idx, (_name, p) in enumerate(
            sorted(model.named_parameters(), key=lambda kv: kv[0])
        ):
            ramp = torch.arange(p.numel(), dtype=torch.float32, device=p.device)
            vals = torch.sin(ramp * 0.013 + idx * 0.7) * 0.02
            p.copy_(vals.reshape(p.shape).to(p.dtype))
    return model


def _build_small(name: str) -> torch.nn.Module:
    """Build a shrunk stand-in of a production config's architecture (HEALPix)."""
    import train  # noqa: PLC0415 — heavy import, GPU env only
    from train_configs import CONFIGS  # noqa: PLC0415

    cfg = CONFIGS[name]
    dit = dataclasses.replace(
        cfg.dit, embed_dim=SMALL_EMBED, n_layers=SMALL_LAYERS, num_heads=SMALL_HEADS
    )
    common = dict(
        in_channels=CH,
        out_channels=CH,
        nside=NSIDE,
        tile_size=TILE,
        dit_cfg=dit,
        do_bf16_mixed=False,
        depth_levels=DEPTH,
        wind_channel_indices=(0, 1) if dit.do_rotate_wind else None,
        grid_type="healpix",  # avoids the cubesphere latlon .nc dependency
        cubesphere_latlon_path=None,
    )
    if cfg.experiment.model_type == "pixeldit":
        pix = dataclasses.replace(
            cfg.pixel_dit,
            embed_dim=SMALL_PIXEL_EMBED,
            n_layers=SMALL_PIXEL_LAYERS,
            num_heads=SMALL_PIXEL_HEADS,
        )
        net = train.DiT3DPixel(**common, pixel_cfg=pix)
    else:
        net = train.DiT3D(**common)
    return net.cuda(), dit.index_is_latlon


def _fixed_x() -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(0)
    return torch.randn(1, CH, DEPTH, TILE, TILE, generator=gen, device="cuda")


def _make_index(index_is_latlon: bool):
    """Index for ``DiT.forward``: a lat/lon dict (radians) when the model is built
    with ``index_is_latlon=True``, else a flat HEALPix pixel-index tensor."""
    if index_is_latlon:
        lat = (
            torch.linspace(0.2, 0.7, TILE, device="cuda")
            .view(1, TILE, 1)
            .expand(1, TILE, TILE)
        )
        lon = (
            torch.linspace(0.1, 0.6, TILE, device="cuda")
            .view(1, 1, TILE)
            .expand(1, TILE, TILE)
        )
        return {"lat": lat.contiguous(), "lon": lon.contiguous()}
    return torch.arange(TILE * TILE, device="cuda").reshape(1, TILE, TILE)


@requires_cuda
@pytest.mark.parametrize("name", PRODUCTION_CONFIGS)
def test_production_inference_unchanged(name: str) -> None:
    """Production DiT3D / PixelDiT forward output must be identical post-cleanup."""
    pytest.importorskip("train_configs", reason="needs the screamcast training env")
    torch.manual_seed(0)
    model, index_is_latlon = _build_small(name)
    _fill_params_deterministically(model)
    model.eval()
    x = _fixed_x()
    index = _make_index(index_is_latlon)
    with torch.no_grad():
        y = model(x, index).float().cpu()
    assert y.shape == x.shape

    ref_path = REF_DIR / f"{name}.pt"
    if REGEN:
        REF_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(y, ref_path)
        return
    if not ref_path.exists():
        pytest.skip(
            f"no local golden {ref_path.name}; (re)generate with "
            f"SCREAMCAST_REGEN_GOLDEN=1 pytest {Path(__file__).name}"
        )
    ref = torch.load(ref_path)
    assert y.shape == ref.shape, f"{name}: shape {tuple(y.shape)} != {tuple(ref.shape)}"
    max_diff = (y - ref).abs().max().item()
    assert torch.allclose(y, ref, atol=ATOL, rtol=RTOL), (
        f"{name}: production inference output CHANGED (max abs diff {max_diff:.3e}). "
        f"The cleanup altered a reachable code path."
    )
