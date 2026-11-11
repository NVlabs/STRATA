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
"""Validate fused RMS attention contracts and state; Cirrus-42 (AI agent)."""

import pytest
import torch
from physicsnemo.experimental.models.strata.layers import Natten3DSelfAttention
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode

from screamcast import column_rms_attention


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
        column_rms_attention.restore()


def test_fake_returns_contiguous_bf16_projection_input():
    with FakeTensorMode():
        raw = torch.empty((8, 24576, 3072), device="cuda", dtype=torch.bfloat16)
        logits = torch.empty((8, 24576, 1024), device="cuda", dtype=torch.bfloat16)
        output = column_rms_attention.normalized_attention_gate(raw, logits)
        assert output.shape == logits.shape
        assert output.dtype == torch.bfloat16
        assert output.is_contiguous()
        assert output is not logits


def test_only_depth_modules_change_and_all_parameters_survive(strata):
    modules = list(strata.backbone.attentions)
    originals = [module.forward for module in modules]
    names = tuple(strata.state_dict())
    ids = [id(p) for p in strata.parameters()]
    column_rms_attention.install(strata)
    assert len(column_rms_attention._ORIGINALS) == 12
    for index, module in enumerate(modules):
        if index % 2:
            assert module.forward.__func__ is column_rms_attention._forward
        else:
            assert module.forward == originals[index]
    assert tuple(strata.state_dict()) == names
    assert [id(p) for p in strata.parameters()] == ids
    column_rms_attention.restore()
    assert [module.forward for module in modules] == originals
    assert all("forward" not in module.__dict__ for module in modules)


@pytest.mark.parametrize("invalid", ("affine", "epsilon", "count"))
def test_install_rejects_incompatible_norms_before_patching(strata, invalid):
    last = strata.backbone.attentions[-1]
    if invalid == "affine":
        last.q_norm = nn.RMSNorm(64, eps=1e-6, device="meta")
    elif invalid == "epsilon":
        last.k_norm.eps = 1e-5
    else:
        last.do_depthwise_attention = False
    with pytest.raises(ValueError):
        column_rms_attention.install(strata)
    assert not column_rms_attention._ORIGINALS
    assert all(
        "forward" not in module.__dict__ for module in strata.backbone.attentions
    )


def test_forward_requires_native_autocast_and_no_depth_rope(strata):
    column_rms_attention.install(strata)
    module = strata.backbone.attentions[-1]
    with pytest.raises(ValueError, match="BF16 autocast"):
        module(torch.empty(0), latent_dhw=(24, 32, 32))
    with pytest.raises(ValueError, match="without RoPE"):
        module(
            torch.empty(0),
            latent_dhw=(24, 32, 32),
            rope_tables=(torch.empty(0), torch.empty(0)),
        )
