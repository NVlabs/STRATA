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
"""Validate reversible native-order loop scheduling; Cirrus-42 (AI agent)."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from screamcast import decoder_benchmark
from screamcast import normalization_loop_schedule as loops
from screamcast import normalization_model as parent
from screamcast.forward_stage_benchmark import CachedConditioning


@pytest.fixture
def strata():
    model = nn.Module()
    model.backbone = nn.Module()
    model.backbone.bf16_mixed = True
    model.backbone.blocks = nn.ModuleList()
    model.register_parameter("sentinel", nn.Parameter(torch.ones(1)))
    for index in range(24):
        block = nn.Module()
        block.do_depthwise_attention = bool(index % 2)
        block.norm1 = nn.LayerNorm(1024, elementwise_affine=False, eps=1e-6)
        block.norm2 = nn.LayerNorm(1024, elementwise_affine=False, eps=1e-6)
        model.backbone.blocks.append(block)
    model.eval()
    parent.install(model)
    try:
        yield model
    finally:
        loops.restore()
        parent.restore()


def test_kernel_swap_keeps_methods_modules_parameters_state(strata):
    norms = [
        norm for block in strata.backbone.blocks for norm in (block.norm1, block.norm2)
    ]
    methods = [norm.forward for norm in norms]
    state = tuple(strata.state_dict())
    params = [id(x) for x in strata.parameters()]
    modules = tuple(strata.modules())
    loops.install(strata)
    assert parent.native_order_kernel is loops.unrolled_kernel
    assert [norm.forward for norm in norms] == methods
    assert (
        tuple(strata.state_dict()) == state
        and [id(x) for x in strata.parameters()] == params
    )
    assert tuple(strata.modules()) == modules
    loops.restore()
    assert parent.native_order_kernel is loops._NATIVE_KERNEL
    assert [norm.forward for norm in norms] == methods


@pytest.mark.parametrize("invalid", ("training", "count", "parent", "method"))
def test_invalid_parent_keeps_native_kernel(strata, invalid):
    if invalid == "training":
        strata.train()
    elif invalid == "count":
        strata.backbone.blocks = strata.backbone.blocks[:23]
    elif invalid == "parent":
        parent.restore()
    else:
        strata.backbone.blocks[-1].norm2.forward = parent._ORIGINALS[-1][2]
    with pytest.raises(ValueError):
        loops.install(strata)
    assert parent.native_order_kernel is loops._NATIVE_KERNEL


def test_combined_trial_requires_backbone_scope():
    args = SimpleNamespace(
        scope="decoder",
        experiment="semantic-normalization-column-schedules",
        baseline_experiment="pixel-norm2-modulation",
    )
    with pytest.raises(ValueError, match="backbone isolation"):
        decoder_benchmark.run(args)


def test_combined_trial_accepts_backbone_scope(monkeypatch):
    def stop(*args, **kwargs):
        raise RuntimeError("accepted combined scope")

    monkeypatch.setattr(torch, "load", stop)
    args = SimpleNamespace(
        scope="backbone",
        experiment="semantic-normalization-column-schedules",
        baseline_experiment="pixel-norm2-modulation",
        inputs="unused",
    )
    with pytest.raises(RuntimeError, match="accepted combined scope"):
        decoder_benchmark.run(args)


@pytest.mark.parametrize("cached", (False, True))
def test_combination_inherits_column_parent_and_skips_cached_installer(
    monkeypatch, cached
):
    original = decoder_benchmark.apply_experiment
    parents = []
    installed = []

    def dispatch(model, name):
        if name == "semantic-column-pinned-scale":
            parents.append(name)
        else:
            original(model, name)

    monkeypatch.setattr(decoder_benchmark, "apply_experiment", dispatch)
    monkeypatch.setattr(loops, "install", lambda model: installed.append(model))
    model = nn.Module()
    model.backbone = (
        CachedConditioning(torch.empty(0), (24, 32, 32)) if cached else nn.Module()
    )
    original(model, "semantic-normalization-column-schedules")
    assert parents == ["semantic-column-pinned-scale"]
    assert installed == ([] if cached else [model])
