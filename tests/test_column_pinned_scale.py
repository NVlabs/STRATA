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
"""Pinned-scale column FakeTensor/state/restoration contracts; Cirrus-42 (AI agent)."""

from types import SimpleNamespace

import pytest
import torch
from physicsnemo.experimental.models.strata.layers import Natten3DSelfAttention
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode

from screamcast import column_pinned_scale as two
from screamcast import column_rms_attention as parent
from screamcast import decoder_benchmark
from screamcast.forward_stage_benchmark import CachedConditioning


@pytest.fixture
def strata():
    model = nn.Module()
    with torch.device("meta"):
        model.backbone = nn.ModuleList(
            [
                Natten3DSelfAttention(
                    1024,
                    num_heads=16,
                    qk_norm=True,
                    qk_norm_affine=False,
                    gated_attention=True,
                    do_depthwise_attention=True,
                    attn_kernel=9,
                )
                for _ in range(12)
            ]
        )
    model.eval()
    parent.install(model)
    try:
        yield model
    finally:
        two.restore()
        parent.restore()


def test_fake_pinned_scale_core_keeps_bnc_bf16_output():
    with FakeTensorMode():
        raw = torch.empty((8, 24576, 3072), device="cuda", dtype=torch.bfloat16)
        gate = torch.empty((8, 24576, 1024), device="cuda", dtype=torch.bfloat16)
        output = two.attention(raw, gate)
        assert (
            output.shape == gate.shape
            and output.dtype == gate.dtype
            and output.is_contiguous()
        )


def test_install_restore_only_changes_core_pointer(strata):
    params = [id(p) for p in strata.parameters()]
    state = tuple(strata.state_dict())
    modules = tuple(strata.modules())
    methods = [module.forward for module in strata.backbone]
    projections = [
        (m.qkv, m.gate_proj, m.proj, m.q_norm, m.k_norm) for m in strata.backbone
    ]
    two.install(strata)
    assert parent.normalized_attention_gate is two.attention
    assert params == [id(p) for p in strata.parameters()]
    assert state == tuple(strata.state_dict()) and modules == tuple(strata.modules())
    assert methods == [module.forward for module in strata.backbone]
    assert projections == [
        (m.qkv, m.gate_proj, m.proj, m.q_norm, m.k_norm) for m in strata.backbone
    ]
    two.restore()
    assert parent.normalized_attention_gate is two._NATIVE_CORE
    assert methods == [module.forward for module in strata.backbone]


@pytest.mark.parametrize("invalid", ("training", "count", "parent", "method"))
def test_invalid_parent_rejected_before_replacing_core(strata, invalid):
    if invalid == "training":
        strata.train()
    elif invalid == "count":
        strata.backbone = strata.backbone[:11]
    elif invalid == "parent":
        parent.restore()
    else:
        strata.backbone[-1].forward = parent._ORIGINALS[-1][2]
    with pytest.raises(ValueError):
        two.install(strata)
    assert parent.normalized_attention_gate is two._NATIVE_CORE


def test_column_trial_rejects_decoder_scope():
    args = SimpleNamespace(
        scope="decoder",
        experiment="semantic-column-pinned-scale",
        baseline_experiment="pixel-norm2-modulation",
    )
    with pytest.raises(ValueError, match="backbone isolation"):
        decoder_benchmark.run(args)


def test_column_trial_accepts_backbone_scope_before_loading(monkeypatch):
    def stop(*args, **kwargs):
        raise RuntimeError("accepted column scope")

    monkeypatch.setattr(torch, "load", stop)
    args = SimpleNamespace(
        scope="backbone",
        experiment="semantic-column-pinned-scale",
        baseline_experiment="pixel-norm2-modulation",
        inputs="unused",
    )
    with pytest.raises(RuntimeError, match="accepted column scope"):
        decoder_benchmark.run(args)


@pytest.mark.parametrize("cached", (False, True))
def test_column_case_skips_backbone_installer_for_cached_decoder(monkeypatch, cached):
    original_apply = decoder_benchmark.apply_experiment
    parents = []
    installed = []

    def dispatch(model, name):
        if name == "pixel-norm2-modulation":
            parents.append(name)
        else:
            original_apply(model, name)

    monkeypatch.setattr(decoder_benchmark, "apply_experiment", dispatch)
    monkeypatch.setattr(two, "install", lambda model: installed.append(model))
    model = nn.Module()
    model.backbone = (
        CachedConditioning(torch.empty(0), (24, 32, 32)) if cached else nn.Module()
    )
    original_apply(model, "semantic-column-pinned-scale")
    assert parents == ["pixel-norm2-modulation"]
    assert installed == ([] if cached else [model])
