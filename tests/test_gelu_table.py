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
"""Check fake tracing and exact restoration of GELU modules; Cirrus-42 (AI agent)."""

import pytest
import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode

from screamcast import gelu_table


@pytest.mark.parametrize("shape", ((8, 24576, 4096), (8, 393216, 512)))
def test_fake_lookup_preserves_shape_and_does_not_build_table(shape):
    before = dict(gelu_table._TABLES)
    with FakeTensorMode():
        input = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
        table = torch.empty((65536,), device="cuda", dtype=torch.bfloat16)
        output = gelu_table.lookup(input, table)
        assert output.shape == input.shape
        assert output.dtype == input.dtype
        assert output.is_contiguous()
    assert gelu_table._TABLES == before


@pytest.mark.parametrize("pixel", (False, True))
def test_install_restore_retains_parameters_and_state_dict(monkeypatch, pixel):
    strata = nn.Module()
    strata.backbone = nn.Module()
    strata.backbone.blocks = nn.ModuleList()
    strata.pixel_blocks = nn.ModuleList()
    for index in range(28):
        block = nn.Module()
        block.mlp = nn.Module()
        block.mlp.layers = nn.Sequential(nn.Linear(1, 1), nn.GELU(approximate="tanh"))
        (strata.backbone.blocks if index < 24 else strata.pixel_blocks).append(block)
    strata.eval()
    parameter_ids = [id(p) for p in strata.parameters()]
    state_keys = tuple(strata.state_dict())
    table = torch.empty(65536, dtype=torch.bfloat16)
    monkeypatch.setattr(gelu_table, "prepare_table", lambda device: table)
    try:
        gelu_table.install(strata, pixel=pixel)
        selected = list(strata.backbone.blocks) + (
            list(strata.pixel_blocks) if pixel else []
        )
        assert [id(p) for p in strata.parameters()] == parameter_ids
        assert tuple(strata.state_dict()) == state_keys
        for block in selected:
            assert block.mlp.layers[1]._cirrus42_gelu_table is table
        gelu_table.restore()
        for block in selected:
            activation = block.mlp.layers[1]
            assert "forward" not in activation.__dict__
            assert not hasattr(activation, "_cirrus42_gelu_table")
        assert tuple(strata.state_dict()) == state_keys
    finally:
        gelu_table.restore()
