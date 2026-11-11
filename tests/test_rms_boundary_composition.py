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
"""Verify independent restoration of composed fusions; Cirrus-42 (AI agent)."""

import torch
from physicsnemo.experimental.models.strata.layers import Natten3DSelfAttention
from torch import nn

from screamcast import column_rms_attention, residual_boundary, token_layout


def test_restore_boundary_retains_attention_patch_until_attention_restore():
    model = nn.Module()
    model.backbone = nn.Module()
    model.backbone.blocks = nn.ModuleList()
    with torch.device("meta"):
        for index in range(24):
            block = nn.Module()
            block.attn = Natten3DSelfAttention(
                1024,
                num_heads=16,
                qk_norm=True,
                qk_norm_affine=False,
                do_depthwise_attention=bool(index % 2),
                gated_attention=True,
                attn_kernel=9,
            )
            model.backbone.blocks.append(block)
    model.eval()
    last = model.backbone.blocks[-1]
    native = last.forward
    original_attention = last.attn.forward
    ids = [id(p) for p in model.parameters()]
    names = tuple(model.state_dict())
    try:
        token_layout.install(model)
        column_rms_attention.install(model)
        residual_boundary.install(model)
        assert len(token_layout._HANDLES) == 1
        assert last.forward.__func__ is residual_boundary._last_forward
        assert last.attn.forward.__func__ is column_rms_attention._forward
        assert [id(p) for p in model.parameters()] == ids
        assert tuple(model.state_dict()) == names
        residual_boundary.restore()
        assert last.forward == native
        assert len(token_layout._HANDLES) == 2
        assert last.attn.forward.__func__ is column_rms_attention._forward
        column_rms_attention.restore()
        assert last.attn.forward == original_attention
        assert [id(p) for p in model.parameters()] == ids
        assert tuple(model.state_dict()) == names
    finally:
        residual_boundary.restore()
        column_rms_attention.restore()
        token_layout.restore()
