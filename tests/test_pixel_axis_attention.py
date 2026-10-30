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
"""Pixel axis ownership and state contracts; Cirrus-42 (AI agent)."""

from types import SimpleNamespace

import pytest
import torch
from physicsnemo.experimental.models.strata.layers import Natten3DSelfAttention
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode

from screamcast import (
    attention_layout,
    decoder_benchmark,
    pixel_persistent,
    pixel_qkv_pack,
)
from screamcast import pixel_axis_attention as axis


@pytest.fixture
def root():
    model = nn.Module()
    model.pixel_blocks = nn.ModuleList(
        [
            Natten3DSelfAttention(
                128,
                num_heads=2,
                qk_norm=True,
                qk_norm_affine=False,
                gated_attention=False,
                attn_kernel=9,
            )
            for _ in range(4)
        ]
    )
    model.eval()
    pixel_qkv_pack.install(model)
    pixel_persistent.install(model)
    try:
        yield model
    finally:
        axis.restore()
        pixel_persistent.restore()
        pixel_qkv_pack.restore()


@pytest.mark.parametrize("last_only", (False, True))
def test_install_restore_preserves_parent_state_and_learned_objects(root, last_only):
    methods = [module.forward for module in root.pixel_blocks]
    modules = tuple(root.modules())
    parameters = [id(p) for p in root.parameters()]
    state = tuple(root.state_dict())
    for _ in range(2):
        axis.install(root, last_only=last_only)
        assert len(axis._ORIGINALS) == (1 if last_only else 4)
        assert tuple(root.modules()) == modules
        assert [id(p) for p in root.parameters()] == parameters
        assert tuple(root.state_dict()) == state
        assert all(
            module.forward.__func__ is axis._forward
            for module in (root.pixel_blocks[-1:] if last_only else root.pixel_blocks)
        )
        if last_only:
            assert [module.forward for module in root.pixel_blocks[:-1]] == methods[:-1]
        assert attention_layout.attention_core is pixel_persistent.persistent_core
        axis.restore()
        assert [module.forward for module in root.pixel_blocks] == methods


@pytest.mark.parametrize(
    "invalid", ("training", "owner", "count", "gated", "kernel", "dilation", "epsilon")
)
def test_invalid_parent_rejected_before_patching(root, monkeypatch, invalid):
    if invalid == "training":
        root.train()
    elif invalid == "owner":
        monkeypatch.setattr(pixel_qkv_pack, "_ORIGINALS", [])
    elif invalid == "count":
        root.pixel_blocks = root.pixel_blocks[:-1]
    elif invalid == "gated":
        root.pixel_blocks[-1].gated_attention = True
    elif invalid == "kernel":
        root.pixel_blocks[-1].attn_kernel = 11
    elif invalid == "dilation":
        root.pixel_blocks[-1].na_dilation = 2
    else:
        root.pixel_blocks[-1].q_norm.eps = 1e-5
    methods = [module.forward for module in root.pixel_blocks]
    with pytest.raises(ValueError):
        axis.install(root)
    assert not axis._ORIGINALS
    assert [module.forward for module in root.pixel_blocks] == methods


def test_registered_fake_operators_preserve_shapes_dtype_device():
    with FakeTensorMode():
        raw = torch.empty((8, 393216, 384), device="cuda", dtype=torch.bfloat16)
        qkv = axis.packed_qkv(raw)
        assert len({id(x) for x in qkv}) == 3
        output = axis.restore_output(axis.attention(*qkv))
        assert output.shape == (8, 393216, 128)
        assert (
            output.dtype == raw.dtype
            and output.device == raw.device
            and output.is_contiguous()
        )


def test_forward_keeps_native_external_geometry_and_amp_guards(root):
    axis.install(root)
    with pytest.raises(ValueError, match="no RoPE"):
        root.pixel_blocks[0](
            torch.empty(1), latent_dhw=(24, 128, 128), rope_tables=(None, None)
        )
    with pytest.raises(ValueError, match="BF16 autocast"):
        root.pixel_blocks[0](torch.empty(1), latent_dhw=(24, 128, 128))


@pytest.mark.parametrize(
    "experiment", ("pixel-attention-axis", "pixel-attention-axis-last")
)
def test_model_trial_requires_decoder_scope(experiment):
    args = SimpleNamespace(
        experiment=experiment,
        scope="backbone",
        baseline_experiment="semantic-normalization-column-schedules",
    )
    with pytest.raises(ValueError, match="decoder isolation"):
        decoder_benchmark.run(args)


@pytest.mark.parametrize(
    "experiment", ("pixel-attention-axis", "pixel-attention-axis-last")
)
def test_decoder_scope_accepts_confirmed_parent_before_loading(monkeypatch, experiment):
    def stop(*args, **kwargs):
        raise RuntimeError("accepted decoder parent")

    monkeypatch.setattr(torch, "load", stop)
    args = SimpleNamespace(
        experiment=experiment,
        scope="decoder",
        baseline_experiment="semantic-normalization-column-schedules",
        inputs="unused",
    )
    with pytest.raises(RuntimeError, match="accepted decoder parent"):
        decoder_benchmark.run(args)


def test_new_case_installs_confirmed_parent(monkeypatch):
    calls = []
    original = decoder_benchmark.apply_experiment
    monkeypatch.setattr(
        decoder_benchmark, "apply_experiment", lambda strata, name: calls.append(name)
    )
    monkeypatch.setattr(axis, "install", lambda strata: calls.append("axis"))
    original(nn.Module(), "pixel-attention-axis")
    assert calls == ["semantic-normalization-column-schedules", "axis"]


def test_last_only_case_installs_confirmed_parent_and_one_block(monkeypatch):
    calls = []
    original = decoder_benchmark.apply_experiment
    monkeypatch.setattr(
        decoder_benchmark, "apply_experiment", lambda strata, name: calls.append(name)
    )
    monkeypatch.setattr(axis, "install", lambda strata, **kwargs: calls.append(kwargs))
    original(nn.Module(), "pixel-attention-axis-last")
    assert calls == ["semantic-normalization-column-schedules", {"last_only": True}]
