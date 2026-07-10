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
from types import SimpleNamespace

import pytest
import torch

from screamcast.dealias import DealiasedPatchEmbed3D
from screamcast.strata_wrappers import StrataModel, StrataBackboneModel


def _make_pixel_dit(tile_size: int = 64) -> StrataModel:
    # depth=12, patch_size_vert=2 → 6 depth tokens; attn_kernel=3 requires ≥3.
    # do_rope_2d=True precomputes RoPE for (tile_size/patch_horiz)^2 * depth tokens,
    # so feeding a larger spatial domain raises RuntimeError (the bug we test).
    D, PH, PV = 12, 4, 2
    return StrataModel(
        depth=D,
        height=tile_size,
        width=tile_size,
        patch_size_horiz=PH,
        patch_size_vert=PV,
        in_chans=6,
        base_out_chans=6,
        n_layers=2,
        embed_dim=64,
        num_heads=2,
        attn_kernel=3,
        gated_attention=False,
        embed_dim_pixel=32,
        n_layers_pixel=2,
        do_rope_2d=True,
        do_rope_2d_stereographic=False,
        do_rotate_wind=True,
        wind_channel_indices=(0, 1),
        grid_type="healpix",
        nside=512,
    ).cuda()


def _healpix_index(h: int, w: int) -> torch.Tensor:
    return torch.arange(h * w, device="cuda").reshape(1, h, w)


def test_pixel_dit_tile_size_passes():
    """StrataModel forward works for the tile size it was built with."""
    model = _make_pixel_dit(tile_size=64)
    x = torch.randn(1, 6, 12, 64, 64).cuda()
    y = model(x, _healpix_index(64, 64))
    assert y.shape == x.shape


def test_pixel_dit_set_tile_size():
    """set_tile_size recomputes RoPE buffers so 256x256 input succeeds."""
    model = _make_pixel_dit(tile_size=64)
    model.set_tile_size(256, 256)
    assert model._height == 256 and model._width == 256
    x = torch.randn(1, 6, 12, 256, 256).cuda()
    y = model(x, _healpix_index(256, 256))
    assert y.shape == x.shape


def test_pixel_dit_set_tile_size_refreshes_pixel_rope():
    """set_tile_size refreshes the pixel-pathway RoPE buffer when do_rope_2d_pixel=True."""
    D, PH, PV = 12, 4, 2
    model = StrataModel(
        depth=D,
        height=64,
        width=64,
        patch_size_horiz=PH,
        patch_size_vert=PV,
        in_chans=6,
        base_out_chans=6,
        n_layers=2,
        embed_dim=64,
        num_heads=2,
        attn_kernel=3,
        embed_dim_pixel=32,
        n_layers_pixel=2,
        num_heads_pixel=1,
        do_rope_2d_pixel=True,
        do_rotate_wind=True,
        wind_channel_indices=(0, 1),
        grid_type="healpix",
        nside=512,
    ).cuda()
    assert model.strata._rope_cos_pixel.shape[-2] == D * 64 * 64
    model.set_tile_size(128, 128)
    assert model.strata._rope_cos_pixel.shape[-2] == D * 128 * 128
    x = torch.randn(1, 6, D, 128, 128).cuda()
    y = model(x, _healpix_index(128, 128))
    assert y.shape == x.shape


def test_dit_3d():
    model = StrataBackboneModel(
        depth=32,
        height=64,
        width=64,
        patch_size=4,
        in_chans=6,
        base_out_chans=6,
        n_layers=2,
        do_rope_2d=True,
        do_concat_latitude=True,
    ).to("cuda")
    x = torch.randn(1, 6, 32, 64, 64).cuda()
    index = torch.arange(64 * 64).reshape(1, 64, 64).cuda()
    y = model(x, index)
    assert y.shape == x.shape


def test_dit_3d_stereographic():
    model = StrataBackboneModel(
        depth=32,
        height=64,
        width=64,
        patch_size=4,
        in_chans=6,
        base_out_chans=6,
        n_layers=2,
        do_rope_2d=False,
        do_concat_latitude=True,
        do_rope_2d_stereographic=True,
        do_rotate_wind=True,
        wind_channel_indices=(3, 4),  # index of U and V channels
    ).to("cuda")
    x = torch.randn(1, 6, 32, 64, 64).cuda()
    index = torch.arange(64 * 64).reshape(1, 64, 64).cuda()
    y = model(x, index)
    assert y.shape == x.shape


def _checkpoint_set(ratio: float, n_blocks: int, training: bool = True) -> set[int]:
    """Call StrataTransformer3D._should_checkpoint_block against a minimal stub.

    The method only reads three attributes, so we can avoid the cost of
    building a real model (CUDA + weight init). Guards the checkpointing-ratio
    semantics the wrapper's disable_activation_checkpointing relies on across
    physicsnemo pin bumps.
    """
    from physicsnemo.experimental.models.strata import StrataTransformer3D

    stub = SimpleNamespace(
        training=training,
        _activation_checkpointing_ratio=ratio,
        blocks=[None] * n_blocks,
    )
    return {
        i
        for i in range(n_blocks)
        if StrataTransformer3D._should_checkpoint_block(stub, i)
    }


def test_should_checkpoint_block_eval_mode_skips_all():
    # Even at ratio=1.0, eval mode must not checkpoint.
    assert _checkpoint_set(1.0, 32, training=False) == set()


def test_should_checkpoint_block_ratio_extremes():
    assert _checkpoint_set(0.0, 32) == set()
    assert _checkpoint_set(1.0, 32) == set(range(32))


@pytest.mark.parametrize(
    "ratio,n_blocks,expected_n",
    [
        (0.25, 32, 8),
        (0.5, 32, 16),
        (0.75, 32, 24),
        # Regression cases: previously collapsed to a nearby reciprocal-integer ratio.
        (0.9, 32, 29),  # was 32 (acted as 1.0)
        (0.6, 32, 19),  # was 16 (acted as 0.5)
        (0.4, 32, 13),  # was 16 (acted as 0.5)
    ],
)
def test_should_checkpoint_block_picks_first_n(ratio, n_blocks, expected_n):
    assert _checkpoint_set(ratio, n_blocks) == set(range(expected_n))


def test_should_checkpoint_block_small_ratio_rounds_to_zero():
    # round(0.1 * 4) == 0 → nothing checkpointed.
    assert _checkpoint_set(0.1, 4) == set()


def test_dealiased_patch_embed_asymmetric_patch_shape():
    # patch_size=(1, 4, 4): output should be (D, H/4, W/4), not D-reduced.
    D, H, W = 8, 32, 32
    embed = DealiasedPatchEmbed3D(
        D,
        H,
        W,
        patch_size=(1, 4, 4),
        in_chans=3,
        embed_dim=16,
        resample_filter=(1, 4, 6, 4, 1),
    )
    embed.eval()
    x = torch.randn(1, 3, D, H, W)
    with torch.no_grad():
        y = embed(x)
    assert y.shape == (1, 16, D, H // 4, W // 4)


def test_dealiased_patch_embed_pd1_no_vertical_mixing():
    # For patch_size_vert=1, perturbing one input depth slice must leave all
    # other output depth slices unchanged — adjacent atmospheric levels must
    # not bleed into each other.
    D, H, W = 8, 32, 32
    embed = DealiasedPatchEmbed3D(
        D,
        H,
        W,
        patch_size=(1, 4, 4),
        in_chans=3,
        embed_dim=16,
        resample_filter=(1, 4, 6, 4, 1),
    )
    embed.eval()

    x = torch.randn(1, 3, D, H, W)
    x_perturbed = x.clone()
    x_perturbed[:, :, 4, :, :] += 100.0

    with torch.no_grad():
        y = embed(x)
        y_perturbed = embed(x_perturbed)
    diff = (y_perturbed - y).abs()

    for d in range(D):
        max_diff = diff[:, :, d, :, :].max().item()
        if d == 4:
            assert max_diff > 0.0, "expected change at d=4 where input changed"
        else:
            assert max_diff == 0.0, (
                f"depth {d} changed (max diff {max_diff}) when only input d=4 "
                f"was perturbed — vertical bleed at patch_size_vert=1"
            )


def test_dealiased_patch_embed_all_ones_matches_vanilla():
    # patch_size=(1,1,1) means no axis is decimated → filter is all-identity →
    # output should equal vanilla PatchEmbed3D output when proj weights match.
    from physicsnemo.experimental.models.strata.layers import PatchEmbed3D

    D, H, W = 4, 8, 8
    vanilla = PatchEmbed3D(D, H, W, patch_size=1, in_chans=3, embed_dim=8)
    dealias = DealiasedPatchEmbed3D(
        D,
        H,
        W,
        patch_size=1,
        in_chans=3,
        embed_dim=8,
        resample_filter=(1, 4, 6, 4, 1),
    )
    dealias.proj.load_state_dict(vanilla.proj.state_dict())
    vanilla.eval()
    dealias.eval()

    x = torch.randn(1, 3, D, H, W)
    with torch.no_grad():
        assert torch.allclose(vanilla(x), dealias(x), atol=1e-6)
