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
"""Conditioning vector operator contracts; Cirrus-42 (AI agent)."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode

from screamcast import conditioning_norm, conditioning_vector, decoder_benchmark


@pytest.fixture
def strata():
    model = nn.Module()
    model.pixel_blocks = nn.ModuleList()
    for _ in range(4):
        block = nn.Module()
        block.adaln_bilinear_dw_norm = nn.RMSNorm(
            1024, eps=None, elementwise_affine=False, device="meta"
        )
        block.adaln_bilinear_dw_proj = nn.Linear(1024, 768, device="meta")
        model.pixel_blocks.append(block)
    model.eval()
    conditioning_norm.install(model)
    try:
        yield model
    finally:
        conditioning_vector.restore()


def test_fake_norm_returns_native_projection_input():
    with FakeTensorMode():
        a = torch.empty(
            (192, 128, 128, 512), device="cuda", dtype=torch.bfloat16
        ).permute(0, 3, 1, 2)
        bias = torch.empty(1024, device="cuda", dtype=torch.float32)
        out = conditioning_vector.norm_tail(a, a, bias)
        assert out.shape == (8, 393216, 1024) and out.dtype == torch.bfloat16
        assert out.is_contiguous()


def test_install_restore_preserves_baseline_methods_state_and_parameters(strata):
    ids = [id(p) for p in strata.parameters()]
    names = tuple(strata.state_dict())
    methods = [b._compute_bilinear_dw_adaln_params for b in strata.pixel_blocks]
    conditioning_vector.install(strata)
    assert conditioning_norm.norm_tail is conditioning_vector.norm_tail
    assert ids == [id(p) for p in strata.parameters()]
    assert names == tuple(strata.state_dict())
    conditioning_vector.restore()
    assert conditioning_norm.norm_tail is conditioning_vector._NATIVE_NORM
    assert methods == [b._compute_bilinear_dw_adaln_params for b in strata.pixel_blocks]


def test_wrong_epsilon_rejected_before_operator_patch(strata):
    strata.pixel_blocks[-1].adaln_bilinear_dw_norm.eps = 1e-6
    with pytest.raises(ValueError, match="native nonaffine"):
        conditioning_vector.install(strata)
    assert conditioning_norm.norm_tail is conditioning_vector._NATIVE_NORM


@pytest.mark.parametrize(
    "scope,baseline",
    (("backbone", "semantic-rope-packed-residual-layout"), ("decoder", "baseline")),
)
def test_composed_pixel_trial_requires_decoder_and_confirmed_baseline(scope, baseline):
    args = SimpleNamespace(
        scope=scope,
        experiment="pixel-packed-persistent-vector-norm",
        baseline_experiment=baseline,
    )
    with pytest.raises(ValueError, match="decoder isolation"):
        decoder_benchmark.run(args)


@pytest.mark.parametrize(
    "baseline",
    (
        "pixel-qkv-packed",
        "pixel-packed-persistent",
        "pixel-packed-persistent-vector-norm",
        "pixel-fused-conditioning-conv-norm",
        "pixel-norm1-modulation",
        "pixel-norm2-modulation",
    ),
)
def test_pixel_baseline_scope_reaches_data_loading(monkeypatch, baseline):
    def stop_before_loading(*args, **kwargs):
        raise RuntimeError("scope accepted before data loading")

    monkeypatch.setattr(torch, "load", stop_before_loading)
    args = SimpleNamespace(
        scope="decoder",
        experiment="pixel-packed-persistent-vector-norm",
        baseline_experiment=baseline,
        inputs="unused",
    )
    with pytest.raises(RuntimeError, match="scope accepted"):
        decoder_benchmark.run(args)
