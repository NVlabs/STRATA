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
"""Wrapper-level tests for the two-stage pixel model (StrataModel).

Block-level behavior (AdaLN-zero init, DepthwiseConv chunking, ...) is tested
upstream in physicsnemo's strata suite; these tests cover the wrapper
composition, config translation, and geometry plumbing.
"""

import pytest
import torch

from screamcast.strata_wrappers import StrataModel


def test_dit_pixel():
    D, H, W = 24, 64, 64
    model = StrataModel(
        depth=D,
        height=H,
        width=W,
        patch_size_horiz=4,
        patch_size_vert=2,
        in_chans=6,
        base_out_chans=6,
        n_layers=2,
        embed_dim=128,
        num_heads=2,
        attn_kernel=3,
        embed_dim_pixel=64,
        n_layers_pixel=2,
        num_heads_pixel=1,
        do_concat_latitude=True,
    ).to("cuda")
    x = torch.randn(1, 6, D, H, W).cuda()
    index = torch.arange(H * W).reshape(1, H, W).cuda()
    y = model(x, index)
    assert y.shape == x.shape


def test_dit_pixel_stereographic():
    D, H, W = 24, 64, 64
    model = StrataModel(
        depth=D,
        height=H,
        width=W,
        patch_size_horiz=4,
        patch_size_vert=2,
        in_chans=6,
        base_out_chans=6,
        n_layers=2,
        embed_dim=128,
        num_heads=2,
        attn_kernel=3,
        do_rope_2d_stereographic=True,
        do_rotate_wind=True,
        wind_channel_indices=(0, 1),
        embed_dim_pixel=64,
        n_layers_pixel=2,
        num_heads_pixel=1,
        do_bf16_mixed=True,
        do_bf16_mixed_pixel=True,
    ).to("cuda")
    x = torch.randn(1, 6, D, H, W).cuda()
    index = torch.arange(H * W).reshape(1, H, W).cuda()
    y = model(x, index)
    assert y.shape == x.shape


def test_dit_pixel_bilinear_dw_gelu_project():
    D, H, W = 24, 64, 64
    model = StrataModel(
        depth=D,
        height=H,
        width=W,
        patch_size_horiz=4,
        patch_size_vert=1,  # bilinear upsamples H×W only; requires patch_size_vert=1
        in_chans=6,
        base_out_chans=6,
        n_layers=2,
        embed_dim=128,
        num_heads=2,
        attn_kernel=3,
        embed_dim_pixel=64,
        n_layers_pixel=2,
        num_heads_pixel=1,
        do_concat_latitude=True,
        use_bilinear_dw_gelu_project_adaln_pixel=True,
    ).to("cuda")
    x = torch.randn(1, 6, D, H, W).cuda()
    index = torch.arange(H * W).reshape(1, H, W).cuda()
    y = model(x, index)
    assert y.shape == x.shape


def test_dit_pixel_first_block_only_adaln():
    from physicsnemo.experimental.models.strata.layers import (
        StrataPixel3DBlock,
        StrataTransformer3DBlock,
    )

    D, H, W = 24, 64, 64
    n_layers_pixel = 3
    model = StrataModel(
        depth=D,
        height=H,
        width=W,
        patch_size_horiz=4,
        patch_size_vert=2,
        in_chans=6,
        base_out_chans=6,
        n_layers=2,
        embed_dim=128,
        num_heads=2,
        attn_kernel=3,
        embed_dim_pixel=64,
        n_layers_pixel=n_layers_pixel,
        num_heads_pixel=1,
        do_concat_latitude=True,
        first_block_only_adaln_pixel=True,
    ).to("cuda")

    pixel_blocks = model.strata.pixel_blocks
    assert isinstance(pixel_blocks[0], StrataPixel3DBlock)
    for b in pixel_blocks[1:]:
        assert isinstance(b, StrataTransformer3DBlock)
        assert not isinstance(b, StrataPixel3DBlock)
        assert not hasattr(b, "adaln_pixel_proj")
        assert not hasattr(b, "adaln_bilinear_dw_proj")

    x = torch.randn(1, 6, D, H, W).cuda()
    index = torch.arange(H * W).reshape(1, H, W).cuda()
    y = model(x, index)
    assert y.shape == x.shape


def test_dit_pixel_stereographic_rope_pixel():
    D, H, W = 24, 64, 64
    model = StrataModel(
        depth=D,
        height=H,
        width=W,
        patch_size_horiz=4,
        patch_size_vert=2,
        in_chans=6,
        base_out_chans=6,
        n_layers=2,
        embed_dim=128,
        num_heads=2,
        attn_kernel=3,
        do_rope_2d_stereographic=True,
        do_rotate_wind=True,
        wind_channel_indices=(0, 1),
        embed_dim_pixel=64,
        n_layers_pixel=3,
        num_heads_pixel=1,
        do_bf16_mixed=True,
        do_bf16_mixed_pixel=True,
        first_block_only_adaln_pixel=True,
        do_rope_2d_stereographic_pixel=True,
    ).to("cuda")
    x = torch.randn(1, 6, D, H, W).cuda()
    index = torch.arange(H * W).reshape(1, H, W).cuda()
    y = model(x, index)
    assert y.shape == x.shape


def test_dit_pixel_rope_2d_pixel():
    D, H, W = 24, 64, 64
    model = StrataModel(
        depth=D,
        height=H,
        width=W,
        patch_size_horiz=4,
        patch_size_vert=2,
        in_chans=6,
        base_out_chans=6,
        n_layers=2,
        embed_dim=128,
        num_heads=2,
        attn_kernel=3,
        embed_dim_pixel=64,
        n_layers_pixel=2,
        num_heads_pixel=1,
        do_rope_2d_pixel=True,
    ).to("cuda")
    # Pixel grid is (D, H, W) since pixel patch is 1×1×1.
    assert model.strata._rope_cos_pixel.shape[-2] == D * H * W
    x = torch.randn(1, 6, D, H, W).cuda()
    index = torch.arange(H * W).reshape(1, H, W).cuda()
    y = model(x, index)
    assert y.shape == x.shape


def test_dit_pixel_rope_2d_pixel_and_stereographic_conflict():
    with pytest.raises(ValueError, match="row/column RoPE and stereographic RoPE"):
        StrataModel(
            depth=24,
            height=64,
            width=64,
            patch_size_horiz=4,
            patch_size_vert=2,
            in_chans=6,
            base_out_chans=6,
            n_layers=2,
            embed_dim=128,
            num_heads=2,
            attn_kernel=3,
            embed_dim_pixel=64,
            n_layers_pixel=2,
            num_heads_pixel=1,
            do_rope_2d_pixel=True,
            do_rope_2d_stereographic_pixel=True,
        )


def test_dit_pixel_cross_attn_not_available_in_public():
    with pytest.raises(NotImplementedError, match="cross_attn"):
        StrataModel(
            depth=24,
            height=64,
            width=64,
            patch_size_horiz=4,
            patch_size_vert=2,
            in_chans=6,
            base_out_chans=6,
            n_layers=2,
            embed_dim=128,
            num_heads=2,
            attn_kernel=3,
            embed_dim_pixel=64,
            n_layers_pixel=2,
            num_heads_pixel=1,
            pixel_cond_mode="cross_attn",
        )


def test_factory_translation_production_shape():
    """The train factory maps a production-shaped config correctly.

    Guards the two config-translation properties that would silently corrupt
    a production checkpoint load: the backbone gets stereographic RoPE while
    the pixel stage gets NO RoPE (production pixel RoPE is a separate config
    family), and the bilinear_dw adaln mode is selected.
    """
    import train as train_module
    from screamcast.config import DiTConfig, PixelDiTConfig

    dit_cfg = DiTConfig(
        embed_dim=64,
        n_layers=2,
        num_heads=2,
        attn_kernel=3,
        patch_size_horiz=4,
        patch_size_vert=1,
        do_rope_2d_stereographic=True,
        do_rotate_wind=True,
        qk_norm=True,
        gated_attention=True,
        do_alt_depthwise_attn=True,
        index_is_latlon=True,
    )
    pixel_cfg = PixelDiTConfig(
        embed_dim=32,
        n_layers=2,
        attn_kernel=3,
        qk_norm=True,
        use_bilinear_dw_gelu_project_adaln=True,
    )
    model = train_module.build_strata(
        6,
        6,
        1024,
        32,
        dit_cfg=dit_cfg,
        pixel_cfg=pixel_cfg,
        do_bf16_mixed=False,
        depth_levels=8,
        wind_channel_indices=(0, 1),
        grid_type="healpix",
    )
    assert model.strata.backbone.rope_mode == "stereographic"
    assert model.strata.rope_mode_pixel == "none"
    assert model.strata.adaln_mode == "bilinear_dw"
    assert model.strata.backbone.gated_attention if hasattr(
        model.strata.backbone, "gated_attention"
    ) else True
    assert model._index_is_latlon


def test_dit_pixel_freeze_pixel_blocks():
    D, H, W = 24, 64, 64
    model = StrataModel(
        depth=D,
        height=H,
        width=W,
        patch_size_horiz=4,
        patch_size_vert=2,
        in_chans=6,
        base_out_chans=6,
        n_layers=2,
        embed_dim=128,
        num_heads=2,
        attn_kernel=3,
        embed_dim_pixel=64,
        n_layers_pixel=2,
        num_heads_pixel=1,
        freeze_pixel_blocks=True,
    )
    for name, p in model.strata.pixel_blocks.named_parameters():
        assert not p.requires_grad, f"pixel block param should be frozen: {name}"
    for name, p in model.strata.pixel_final_layer.named_parameters():
        assert not p.requires_grad, f"pixel head param should be frozen: {name}"
    assert any(p.requires_grad for p in model.strata.backbone.parameters())


def _tiny_kwargs(**over):
    kw = dict(
        depth=8, height=32, width=32, patch_size_horiz=4, patch_size_vert=1,
        in_chans=5, base_out_chans=5, n_layers=1, embed_dim=64, num_heads=4,
        attn_kernel=3, do_rope_2d_stereographic=True,
        do_rope_2d_stereographic_pixel=True, index_is_latlon=True,
        grid_type="healpix", nside=16, embed_dim_pixel=32, n_layers_pixel=1,
        num_heads_pixel=2, attn_kernel_pixel=3,
    )
    kw.update(over)
    return kw


def test_rope_length_scale_default_matches_historical_constant():
    """The 0/None sentinel reproduces the exact pre-migration constant."""
    import math

    model = StrataModel(**_tiny_kwargs())
    ph = 4
    expected_backbone = math.sqrt(math.pi * ph**2 / (3.0 * 1024**2))
    expected_pixel = math.sqrt(math.pi / (3.0 * 1024**2))
    assert model.strata.backbone.rope_length_scale == expected_backbone
    assert model.strata.rope_length_scale_pixel == expected_pixel


def test_rope_length_scale_override_propagates():
    """An explicit per-pixel base reaches every stage in consistent units."""
    base = 5.66e-3  # e.g. an ne128pg2 pixel spacing
    model = StrataModel(**_tiny_kwargs(rope_length_scale=base))
    assert model.strata.backbone.rope_length_scale == base * 4
    assert model.strata.rope_length_scale_pixel == base
    with pytest.raises(ValueError, match="must be positive"):
        StrataModel(**_tiny_kwargs(rope_length_scale=-1.0))


def test_set_rope_length_scale_live_update():
    """The experimental setter retargets both stages; axial mode rejects it."""
    model = StrataModel(**_tiny_kwargs())
    model.set_rope_length_scale(1e-2)
    assert model.strata.backbone.rope_length_scale == 4e-2
    assert model.strata.rope_length_scale_pixel == 1e-2

    axial = StrataModel(
        **_tiny_kwargs(
            do_rope_2d_stereographic=False,
            do_rope_2d_stereographic_pixel=False,
            do_rope_2d=True,
        )
    )
    with pytest.raises(ValueError, match="stereographic"):
        axial.set_rope_length_scale(1e-2)
