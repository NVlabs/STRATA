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
"""Check semantic packing contracts and disjoint restoration; Cirrus-42 (AI agent)."""

import pytest
import torch
from physicsnemo.experimental.models.strata.layers import Natten3DSelfAttention
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode

from screamcast import (
    column_rms_attention,
    semantic_attention_layout,
    semantic_qkv_pack,
)


@pytest.fixture
def strata():
    model = nn.Module()
    model.backbone = nn.Module()
    with torch.device("meta"):
        model.backbone.attentions = nn.ModuleList(
            [
                Natten3DSelfAttention(
                    1024,
                    num_heads=16,
                    qk_norm=True,
                    qk_norm_affine=False,
                    do_depthwise_attention=bool(index % 2),
                    gated_attention=True,
                    attn_kernel=9,
                )
                for index in range(24)
            ]
        )
    model.eval()
    try:
        yield model
    finally:
        semantic_qkv_pack.restore()
        column_rms_attention.restore()


def test_fake_pack_returns_three_distinct_native_bf16_buffers():
    with FakeTensorMode():
        raw = torch.empty((8, 24576, 3072), device="cuda", dtype=torch.bfloat16)
        table = torch.empty((8, 1, 24576, 64), device="cuda", dtype=torch.float32)
        outputs = semantic_qkv_pack.packed_qkv(raw, table, table)
        assert len(outputs) == 3
        assert len({id(x) for x in outputs}) == 3
        for output in outputs:
            assert output.shape == (8, 24576, 16, 64)
            assert output.dtype == torch.bfloat16
            assert output.is_contiguous()


def test_semantic_install_preserves_depth_methods_and_all_parameters(strata):
    modules = list(strata.backbone.attentions)
    originals = [module.forward for module in modules]
    ids = [id(p) for p in strata.parameters()]
    names = tuple(strata.state_dict())
    semantic_qkv_pack.install(strata)
    assert len(semantic_qkv_pack._ORIGINALS) == 12
    for index, module in enumerate(modules):
        if index % 2:
            assert module.forward == originals[index]
        else:
            assert module.forward.__func__ is semantic_qkv_pack._forward
    assert [id(p) for p in strata.parameters()] == ids
    assert tuple(strata.state_dict()) == names
    semantic_qkv_pack.restore()
    assert [module.forward for module in modules] == originals
    assert all("forward" not in module.__dict__ for module in modules)


def test_semantic_restore_keeps_column_fusion_until_column_restore(strata):
    modules = list(strata.backbone.attentions)
    originals = [module.forward for module in modules]
    ids = [id(p) for p in strata.parameters()]
    column_rms_attention.install(strata)
    semantic_qkv_pack.install(strata)
    for index, module in enumerate(modules):
        expected = (
            column_rms_attention._forward if index % 2 else semantic_qkv_pack._forward
        )
        assert module.forward.__func__ is expected
    semantic_qkv_pack.restore()
    for index, module in enumerate(modules):
        if index % 2:
            assert module.forward.__func__ is column_rms_attention._forward
        else:
            assert module.forward == originals[index]
    column_rms_attention.restore()
    assert [module.forward for module in modules] == originals
    assert [id(p) for p in strata.parameters()] == ids


@pytest.mark.parametrize("invalid", ("affine", "epsilon", "count", "tile"))
def test_incompatible_configuration_rejected_before_any_patch(
    strata, monkeypatch, invalid
):
    module = strata.backbone.attentions[-2]
    if invalid == "affine":
        module.q_norm = nn.RMSNorm(64, eps=1e-6, device="meta")
    elif invalid == "epsilon":
        module.k_norm.eps = 1e-5
    elif invalid == "count":
        module.do_depthwise_attention = True
    else:
        monkeypatch.setattr(semantic_attention_layout, "_Q_TILE", (4, 4, 8))
    with pytest.raises(ValueError):
        semantic_qkv_pack.install(strata)
    assert not semantic_qkv_pack._ORIGINALS
    assert all("forward" not in x.__dict__ for x in strata.backbone.attentions)


def test_forward_requires_rope_and_native_bf16_autocast(strata):
    semantic_qkv_pack.install(strata)
    module = strata.backbone.attentions[0]
    with pytest.raises(ValueError, match="RoPE tables"):
        module(torch.empty(0), latent_dhw=(24, 32, 32))
    with pytest.raises(ValueError, match="BF16 autocast"):
        module(
            torch.empty(0),
            latent_dhw=(24, 32, 32),
            rope_tables=(torch.empty(0), torch.empty(0)),
        )
