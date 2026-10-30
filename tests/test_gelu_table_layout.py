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
"""Reversible exact table-layout contracts; Cirrus-42 (AI agent)."""

from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode

from screamcast import decoder_benchmark, gelu_table
from screamcast import gelu_table_layout as layout


@pytest.fixture
def root(monkeypatch):
    model = nn.Module()
    model.register_parameter("sentinel", nn.Parameter(torch.ones(1)))
    model.activations = nn.ModuleList([nn.GELU(approximate="tanh") for _ in range(28)])
    model.eval()
    with FakeTensorMode(allow_non_fake_inputs=True):
        full = torch.empty((65536,), dtype=torch.bfloat16, device="cuda")
        for activation in model.activations:
            activation.register_buffer("_cirrus42_gelu_table", full, persistent=False)
            activation.forward = MethodType(gelu_table._forward, activation)
        monkeypatch.setattr(
            gelu_table,
            "_ORIGINALS",
            [(activation, False, None) for activation in model.activations],
        )
        try:
            yield model
        finally:
            layout.restore()


def test_install_preserves_modules_parameters_state_and_restores_exact_parent(root):
    methods = [module.forward for module in root.activations]
    tables = [module._cirrus42_gelu_table for module in root.activations]
    modules = tuple(root.modules())
    parameters = [id(p) for p in root.parameters()]
    state = tuple(root.state_dict())
    for _ in range(2):
        layout.install(root)
        assert tuple(root.modules()) == modules
        assert [id(p) for p in root.parameters()] == parameters
        assert tuple(root.state_dict()) == state
        assert all(
            module._cirrus42_gelu_table is root.activations[0]._cirrus42_gelu_table
            for module in root.activations
        )
        assert root.activations[0]._cirrus42_gelu_table is not tables[0]
        x = torch.empty((8, 393216, 512), dtype=torch.bfloat16, device="cuda")
        actual = root.activations[0](x)
        assert (
            actual.shape == x.shape
            and actual.dtype == x.dtype
            and actual.device == x.device
        )
        layout.restore()
        assert [module.forward for module in root.activations] == methods
        assert all(
            module._cirrus42_gelu_table is table
            for module, table in zip(root.activations, tables)
        )


@pytest.mark.parametrize(
    "invalid", ("training", "count", "owner", "dtype", "shape", "cpu")
)
def test_install_rejects_invalid_parent_before_mutating_other_activations(
    root, monkeypatch, invalid
):
    if invalid == "training":
        root.train()
    elif invalid == "count":
        root.activations[-1].approximate = "none"
    elif invalid == "owner":
        monkeypatch.setattr(gelu_table, "_ORIGINALS", gelu_table._ORIGINALS[:-1])
    else:
        table = root.activations[-1]._cirrus42_gelu_table
        if invalid == "dtype":
            table = table.float()
        elif invalid == "shape":
            table = table[:128]
        else:
            table = table.cpu()
        root.activations[-1]._cirrus42_gelu_table = table
    methods = [module.forward for module in root.activations]
    tables = [module._cirrus42_gelu_table for module in root.activations]
    with pytest.raises(ValueError):
        layout.install(root)
    assert not layout._ORIGINALS
    assert [module.forward for module in root.activations] == methods
    assert all(
        module._cirrus42_gelu_table is table
        for module, table in zip(root.activations, tables)
    )


def test_model_case_requires_live_backbone_scope():
    with pytest.raises(ValueError, match="backbone isolation"):
        decoder_benchmark.run(
            SimpleNamespace(experiment="semantic-gelu-address-layout", scope="decoder")
        )


def test_model_case_installs_confirmed_parent_before_layout(monkeypatch):
    calls = []
    original = decoder_benchmark.apply_experiment

    def install_parent(strata, name):
        calls.append(name)

    monkeypatch.setattr(decoder_benchmark, "apply_experiment", install_parent)
    monkeypatch.setattr(layout, "install", lambda strata: calls.append("layout"))
    original(nn.Module(), "semantic-gelu-address-layout")
    assert calls == ["semantic-normalization-column-schedules", "layout"]
