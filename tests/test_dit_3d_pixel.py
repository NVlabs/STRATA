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
import pytest
import torch

from screamcast.dit_3d import DiTBlock
from screamcast.dit_3d_pixel import DiT_Pixel, PixelDiTBlock


def test_dit_pixel():
    D, H, W = 24, 64, 64
    model = DiT_Pixel(
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
    model = DiT_Pixel(
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
    model = DiT_Pixel(
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


def test_pixel_dit_block_adaln_zero_at_init():
    block = PixelDiTBlock(
        dim=64,
        cond_dim=128,
        pixels_per_patch=16,
        num_heads=1,
        use_bilinear_dw_gelu_project_adaln=True,
    )
    assert block.adaln_bilinear_dw_proj.weight.abs().max() == 0.0
    assert block.adaln_bilinear_dw_proj.bias.abs().max() == 0.0


def test_pixel_dit_block_dw_conv_toggle():
    """use_chunked_depthwise_conv selects DepthwiseConv vs plain nn.Conv2d
    and the two paths agree numerically up to fp32 reduction-order noise."""
    from screamcast.depthwise_conv import DepthwiseConv

    chunked = PixelDiTBlock(
        dim=64,
        cond_dim=128,
        pixels_per_patch=16,
        num_heads=1,
        use_bilinear_dw_gelu_project_adaln=True,
        use_chunked_depthwise_conv=True,
    )
    plain = PixelDiTBlock(
        dim=64,
        cond_dim=128,
        pixels_per_patch=16,
        num_heads=1,
        use_bilinear_dw_gelu_project_adaln=True,
        use_chunked_depthwise_conv=False,
    )
    assert isinstance(chunked.adaln_bilinear_dw_conv, DepthwiseConv)
    assert type(plain.adaln_bilinear_dw_conv) is torch.nn.Conv2d
    # Both paths must keep the identity-init (center-tap = 1, rest = 0).
    for blk in (chunked, plain):
        w = blk.adaln_bilinear_dw_conv.weight
        assert w.shape == (128, 1, 5, 5)
        assert w[:, 0, 2, 2].abs().min() == 1.0
        w_zeroed = w.clone()
        w_zeroed[:, 0, 2, 2] = 0.0
        assert w_zeroed.abs().max() == 0.0
        assert blk.adaln_bilinear_dw_conv.bias.abs().max() == 0.0

    # Numerical equivalence: copy chunked's weights into plain and verify
    # forward outputs agree under realistic input + replicate padding.
    # cuDNN dispatches different kernels for the two paths, so outputs are
    # not bit-identical; tolerances accommodate fp32 reduction-order drift.
    torch.manual_seed(0)
    with torch.no_grad():
        for p_chunked, p_plain in zip(
            chunked.adaln_bilinear_dw_conv.parameters(),
            plain.adaln_bilinear_dw_conv.parameters(),
        ):
            p_chunked.copy_(torch.randn_like(p_chunked))
            p_plain.copy_(p_chunked)
        x = torch.randn(2, 128, 16, 16)
        y_chunked = chunked.adaln_bilinear_dw_conv(x)
        y_plain = plain.adaln_bilinear_dw_conv(x)
    torch.testing.assert_close(y_chunked, y_plain, atol=1e-5, rtol=1e-5)


def test_dit_pixel_first_block_only_adaln():
    D, H, W = 24, 64, 64
    n_layers_pixel = 3
    model = DiT_Pixel(
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

    assert isinstance(model._pixel_blocks[0], PixelDiTBlock)
    for b in model._pixel_blocks[1:]:
        assert isinstance(b, DiTBlock)
        assert not isinstance(b, PixelDiTBlock)
        assert not hasattr(b, "adaln_pixel_proj")
        assert not hasattr(b, "adaln_bilinear_dw_proj")

    x = torch.randn(1, 6, D, H, W).cuda()
    index = torch.arange(H * W).reshape(1, H, W).cuda()
    y = model(x, index)
    assert y.shape == x.shape


def test_dit_pixel_stereographic_rope_pixel():
    D, H, W = 24, 64, 64
    model = DiT_Pixel(
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
    model = DiT_Pixel(
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
    assert model._rope_cos_pixel.shape[-2] == D * H * W
    x = torch.randn(1, 6, D, H, W).cuda()
    index = torch.arange(H * W).reshape(1, H, W).cuda()
    y = model(x, index)
    assert y.shape == x.shape


def test_dit_pixel_rope_2d_pixel_and_stereographic_conflict():
    with pytest.raises(ValueError, match="row/column RoPE and stereographic RoPE"):
        DiT_Pixel(
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


def test_dit_pixel_freeze_pixel_blocks():
    D, H, W = 24, 64, 64
    model = DiT_Pixel(
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
    for name, p in model._pixel_blocks.named_parameters():
        assert not p.requires_grad, f"pixel block param should be frozen: {name}"
    assert any(p.requires_grad for p in model.semantic.parameters())
