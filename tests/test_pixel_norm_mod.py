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
"""Pixel norm1 contracts and reversible installation; Cirrus-42 (AI agent)."""

import pytest
import torch
from physicsnemo.experimental.models.strata.layers import StrataPixel3DBlock
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode

from screamcast import pixel_norm_mod as norm
from screamcast import pixel_qkv_pack


@pytest.fixture
def strata():
    model = nn.Module()
    with torch.device("meta"):
        model.pixel_blocks = nn.ModuleList(
            [
                StrataPixel3DBlock(
                    128,
                    1024,
                    16,
                    2,
                    qk_norm=True,
                    qk_norm_affine=False,
                    attn_kernel=9,
                    gated_attention=False,
                    adaln_mode="bilinear_dw",
                    use_chunked_depthwise_conv=False,
                )
                for _ in range(4)
            ]
        )
    model.eval()
    pixel_qkv_pack.install(model)
    try:
        yield model
    finally:
        norm.restore()
        pixel_qkv_pack.restore()


def test_fake_output_is_bf16_contiguous():
    with FakeTensorMode():
        x = torch.empty((8, 128, 393216), device="cuda", dtype=torch.float32).transpose(
            1, 2
        )
        raw = torch.empty((8, 393216, 768), device="cuda", dtype=torch.bfloat16)
        out = norm.norm1_modulate(x, raw[:, :, :128], raw[:, :, 128:256])
        assert out.shape == x.shape and out.dtype == torch.bfloat16
        assert out.is_contiguous()


def test_install_restore_preserves_state_and_other_operations(strata):
    params = [id(p) for p in strata.parameters()]
    state = tuple(strata.state_dict())
    forwards = [b.forward for b in strata.pixel_blocks]
    children = [
        (
            b.attn,
            b.norm2,
            b.mlp,
            b.adaln_bilinear_dw_proj,
            b._compute_bilinear_dw_adaln_params,
        )
        for b in strata.pixel_blocks
    ]
    norm.install(strata)
    assert len(norm._ORIGINALS) == 4
    assert all(b.forward.__func__ is norm._forward for b in strata.pixel_blocks)
    assert params == [id(p) for p in strata.parameters()]
    assert state == tuple(strata.state_dict())
    assert children == [
        (
            b.attn,
            b.norm2,
            b.mlp,
            b.adaln_bilinear_dw_proj,
            b._compute_bilinear_dw_adaln_params,
        )
        for b in strata.pixel_blocks
    ]
    norm.restore()
    assert forwards == [b.forward for b in strata.pixel_blocks]
    assert all("forward" not in b.__dict__ for b in strata.pixel_blocks)


@pytest.mark.parametrize(
    "invalid", ("epsilon", "affine", "mode", "count", "training", "attention")
)
def test_invalid_contract_rejected_before_any_patch(strata, invalid):
    if invalid == "epsilon":
        strata.pixel_blocks[-1].norm1.eps = 1e-5
    elif invalid == "affine":
        strata.pixel_blocks[-1].norm1 = nn.LayerNorm(128, eps=1e-6, device="meta")
    elif invalid == "mode":
        strata.pixel_blocks[-1].adaln_mode = "pixel_proj"
    elif invalid == "count":
        strata.pixel_blocks = strata.pixel_blocks[:3]
    elif invalid == "training":
        strata.train()
    else:
        pixel_qkv_pack.restore()
    forwards = [b.forward for b in strata.pixel_blocks]
    with pytest.raises(ValueError):
        norm.install(strata)
    assert not norm._ORIGINALS
    assert forwards == [b.forward for b in strata.pixel_blocks]
