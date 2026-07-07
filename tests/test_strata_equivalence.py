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
"""A/B equivalence: legacy DiT/DiT_Pixel vs physicsnemo-backed Strata wrappers.

Transition-time test (deleted with the legacy files): builds the OLD model and
the NEW wrapper with the same config, remaps the old model's weights into the
new one via checkpoint_compat, and compares forwards on identical inputs.

- Non-stereographic configs must match to fp32 tolerance (max abs <= 1e-5):
  the only intentional numeric change in the migration is the stereographic
  RoPE coordinate math, so everything else is required to be equivalent.
- Stereographic configs are compared with a loose bound and the drift is
  printed: physicsnemo pools/centers coordinates with the spherical centroid
  instead of the legacy arithmetic/circular mean (accepted difference).

GPU-only: neighborhood attention requires CUDA natten kernels.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="NA3D attention requires CUDA"
)

D, H, W = 8, 32, 32
IN_CHANS = 5
COMMON = dict(
    depth=D,
    height=H,
    width=W,
    in_chans=IN_CHANS,
    base_out_chans=IN_CHANS,
    n_layers=3,
    embed_dim=64,
    num_heads=4,
    attn_kernel=3,
    patch_size_vert=1,
    patch_size_horiz=4,
    qk_norm=True,
    gated_attention=True,
    do_alt_depthwise_attn=True,
    do_rotate_wind=True,
    wind_channel_indices=(0, 1),
    grid_type="healpix",
    nside=16,  # lookup buffers unused with index_is_latlon=True; keep tiny
    index_is_latlon=True,
)

PIXEL_COMMON = dict(
    embed_dim_pixel=32,
    n_layers_pixel=2,
    num_heads_pixel=2,
    attn_kernel_pixel=3,
    qk_norm_pixel=True,
)


def _index(batch: int, device: str = "cuda"):
    lat_vals = torch.linspace(-0.4, 0.4, H, device=device)
    lon_vals = torch.linspace(0.1, 0.7, W, device=device)
    lat_grid, lon_grid = torch.meshgrid(lat_vals, lon_vals, indexing="ij")
    return {
        "lat": lat_grid.unsqueeze(0).expand(batch, -1, -1),
        "lon": lon_grid.unsqueeze(0).expand(batch, -1, -1),
    }


def _randomize(model, seed: int = 3):
    """Fill all params with deterministic noise.

    A freshly-initialized model outputs exactly zero (DiT zero-inits its
    output head and AdaLN projections), which would make the A/B comparison
    vacuous. The random values flow to the new model through the remapped
    state-dict load, so both models compute with identical weights.
    """
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, p in sorted(model.named_parameters()):
            p.copy_(torch.randn(p.shape, generator=gen) * 0.02)


def _forward_pair(old_model, new_model, batch: int = 2):
    from screamcast.checkpoint_compat import remap_legacy_state_dict

    _randomize(old_model)
    old_model = old_model.cuda().eval()
    new_model = new_model.cuda().eval()
    remapped, report = remap_legacy_state_dict(
        old_model.state_dict(), target_model=new_model
    )
    assert report.was_legacy
    assert not report.missing_in_target and not report.unexpected_in_target
    new_model.load_state_dict(remapped, strict=True)

    torch.manual_seed(7)
    x = torch.randn(batch, IN_CHANS, D, H, W, device="cuda")
    index = _index(batch)
    with torch.no_grad():
        y_old = old_model(x, index)
        y_new = new_model(x, index)
    assert y_old.shape == y_new.shape
    return y_old, y_new


def _report(tag, y_old, y_new):
    diff = (y_old - y_new).abs().max().item()
    rel = diff / y_old.std().item()
    print(f"[{tag}] max abs diff {diff:.3e} (rel to std {rel:.3e})")
    return diff


@pytest.mark.parametrize("rope", ["none", "axial"])
def test_backbone_equivalent_nonstereo(rope):
    from screamcast.dit_3d import DiT
    from screamcast.strata_wrappers import ScreamcastStrataBackbone

    cfg = dict(COMMON, do_rope_2d=(rope == "axial"))
    y_old, y_new = _forward_pair(DiT(**cfg), ScreamcastStrataBackbone(**cfg))
    assert _report(f"backbone/{rope}", y_old, y_new) <= 1e-5


def test_backbone_stereographic_drift_bounded():
    from screamcast.dit_3d import DiT
    from screamcast.strata_wrappers import ScreamcastStrataBackbone

    cfg = dict(COMMON, do_rope_2d_stereographic=True)
    y_old, y_new = _forward_pair(DiT(**cfg), ScreamcastStrataBackbone(**cfg))
    # Expected drift: spherical-centroid vs mean tile center / patch pooling.
    assert _report("backbone/stereo", y_old, y_new) / y_old.std().item() <= 5e-2


@pytest.mark.parametrize("adaln", ["pixel_proj", "bilinear_dw"])
def test_pixel_equivalent_nonstereo(adaln):
    from screamcast.dit_3d_pixel import DiT_Pixel
    from screamcast.strata_wrappers import ScreamcastStrata

    cfg = dict(
        COMMON,
        **PIXEL_COMMON,
        use_bilinear_dw_gelu_project_adaln_pixel=(adaln == "bilinear_dw"),
    )
    y_old, y_new = _forward_pair(DiT_Pixel(**cfg), ScreamcastStrata(**cfg))
    assert _report(f"pixel/{adaln}", y_old, y_new) <= 1e-5


def test_pixel_production_shape_stereographic():
    """Production-shaped config: stereo backbone RoPE + bilinear_dw adaln."""
    from screamcast.dit_3d_pixel import DiT_Pixel
    from screamcast.strata_wrappers import ScreamcastStrata

    cfg = dict(
        COMMON,
        **PIXEL_COMMON,
        do_rope_2d_stereographic=True,
        use_bilinear_dw_gelu_project_adaln_pixel=True,
    )
    y_old, y_new = _forward_pair(DiT_Pixel(**cfg), ScreamcastStrata(**cfg))
    assert _report("pixel/stereo+bilinear", y_old, y_new) / y_old.std().item() <= 5e-2


def test_pixel_dealias_and_freeze_equivalent():
    from screamcast.dit_3d_pixel import DiT_Pixel
    from screamcast.strata_wrappers import ScreamcastStrata

    cfg = dict(
        COMMON,
        **PIXEL_COMMON,
        use_dealiased_patch_embed=True,
        freeze_semantic=True,
    )
    old_model = DiT_Pixel(**cfg)
    new_model = ScreamcastStrata(**cfg)
    # Freeze flags must produce the same trainable-parameter partition.
    old_frozen = {
        n for n, p in old_model.named_parameters() if not p.requires_grad
    }
    new_frozen = {
        n for n, p in new_model.named_parameters() if not p.requires_grad
    }
    assert len(old_frozen) == len(new_frozen) and len(new_frozen) > 0
    y_old, y_new = _forward_pair(old_model, new_model)
    assert _report("pixel/dealias", y_old, y_new) <= 1e-5


def test_raw_legacy_load_through_module_machinery():
    """Loading an unremapped legacy dict (compile-era prefix included) works.

    Exercises the load_state_dict PRE-HOOK path: the caller does no remapping
    at all, exactly like lightning fabric / pipeline.load_checkpoint would.
    """
    from screamcast.dit_3d_pixel import DiT_Pixel
    from screamcast.strata_wrappers import ScreamcastStrata

    cfg = dict(COMMON, **PIXEL_COMMON)
    old_model = DiT_Pixel(**cfg)
    _randomize(old_model)
    raw = {"_orig_mod." + k: v for k, v in old_model.state_dict().items()}

    new_model = ScreamcastStrata(**cfg)
    new_model.load_state_dict(raw, strict=True)

    old_model = old_model.cuda().eval()
    new_model = new_model.cuda().eval()
    torch.manual_seed(7)
    x = torch.randn(1, IN_CHANS, D, H, W, device="cuda")
    index = _index(1)
    with torch.no_grad():
        y_old = old_model(x, index)
        y_new = new_model(x, index)
    assert _report("pixel/raw-legacy-load", y_old, y_new) <= 1e-5


def test_set_tile_size_axial_pixel_rope():
    """Wrapper's pixel-rope re-tiling reach-in matches a fresh construction."""
    from screamcast.strata_wrappers import ScreamcastStrata

    cfg = dict(COMMON, **PIXEL_COMMON, do_rope_2d_pixel=True)
    model = ScreamcastStrata(**cfg).cuda().eval()
    model.set_tile_size(2 * H, 2 * W)
    fresh = ScreamcastStrata(
        **{**cfg, "height": 2 * H, "width": 2 * W}
    ).cuda()
    torch.testing.assert_close(
        model.strata._rope_cos_pixel, fresh.strata._rope_cos_pixel
    )
    x = torch.randn(1, IN_CHANS, D, 2 * H, 2 * W, device="cuda")
    lat_vals = torch.linspace(-0.4, 0.4, 2 * H, device="cuda")
    lon_vals = torch.linspace(0.1, 0.7, 2 * W, device="cuda")
    lat_grid, lon_grid = torch.meshgrid(lat_vals, lon_vals, indexing="ij")
    index = {"lat": lat_grid.unsqueeze(0), "lon": lon_grid.unsqueeze(0)}
    with torch.no_grad():
        y = model(x, index)
    assert y.shape == (1, IN_CHANS, D, 2 * H, 2 * W)
