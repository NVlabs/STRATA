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
"""State-layout ownership and FakeTensor contracts; Cirrus-42 (AI agent)."""

from types import SimpleNamespace

import pytest
import torch
from physicsnemo.experimental.models.strata.layers import StrataPixel3DBlock
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode

from screamcast import (
    decoder_benchmark,
    pixel_norm2_mod,
    pixel_norm_mod,
    pixel_qkv_pack,
)
from screamcast import pixel_state_layout_model as selected


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
    pixel_norm_mod.install(model)
    pixel_norm2_mod.install(model)
    try:
        yield model
    finally:
        selected.restore()
        pixel_norm2_mod.restore()
        pixel_norm_mod.restore()
        pixel_qkv_pack.restore()


def fake_inputs():
    x = torch.empty((8, 393216, 128), device="cuda", dtype=torch.float32)
    raw = torch.empty((8, 393216, 768), device="cuda", dtype=torch.bfloat16)
    attn = torch.empty(x.shape, device="cuda", dtype=torch.bfloat16)
    return x, raw, attn


def test_fake_normalizations_are_contiguous_bf16():
    with FakeTensorMode():
        x, raw, attn = fake_inputs()
        outputs = [
            selected.norm1(x, raw[:, :, :128], raw[:, :, 128:256]),
            selected.norm2(
                x, attn, raw[:, :, 256:384], raw[:, :, 384:512], raw[:, :, 512:640]
            ),
        ]
        assert all(
            y.shape == x.shape and y.dtype == torch.bfloat16 and y.is_contiguous()
            for y in outputs
        )


@pytest.mark.parametrize(
    "invalid", ("state_stride", "state_dtype", "mod_stride", "attention_stride")
)
def test_fake_invalid_tensor_contract(invalid):
    with FakeTensorMode():
        x, raw, attn = fake_inputs()
        if invalid == "state_stride":
            x = torch.empty((8, 128, 393216), device="cuda").transpose(1, 2)
        elif invalid == "state_dtype":
            x = x.to(torch.bfloat16)
        elif invalid == "mod_stride":
            raw = (
                torch.empty((8, 393216, 768), device="cuda", dtype=torch.bfloat16)
                .transpose(0, 1)
                .contiguous()
                .transpose(0, 1)
            )
        else:
            attn = torch.empty(
                (8, 128, 393216), device="cuda", dtype=torch.bfloat16
            ).transpose(1, 2)
        with pytest.raises(ValueError):
            selected.norm2(
                x, attn, raw[:, :, 256:384], raw[:, :, 384:512], raw[:, :, 512:640]
            )


def test_install_restore_preserves_all_learned_state(strata):
    params = [id(p) for p in strata.parameters()]
    modules = [id(m) for m in strata.modules()]
    keys = tuple(strata.state_dict())
    original = [b.forward for b in strata.pixel_blocks]
    selected.install(strata)
    assert [b.forward.__func__ for b in strata.pixel_blocks] == list(selected._FORWARDS)
    assert params == [id(p) for p in strata.parameters()]
    assert modules == [id(m) for m in strata.modules()]
    assert keys == tuple(strata.state_dict())
    selected.restore()
    assert original == [b.forward for b in strata.pixel_blocks]


@pytest.mark.parametrize(
    "invalid", ("parent", "count", "training", "epsilon", "affine")
)
def test_invalid_owner_or_norm_rejected_before_patching(strata, invalid):
    if invalid == "parent":
        pixel_norm2_mod.restore()
    elif invalid == "count":
        strata.pixel_blocks = strata.pixel_blocks[:3]
    elif invalid == "training":
        strata.train()
    elif invalid == "epsilon":
        strata.pixel_blocks[-1].norm1.eps = 1e-5
    else:
        strata.pixel_blocks[-1].norm2 = nn.LayerNorm(128, eps=1e-6, device="meta")
    original = [b.forward for b in strata.pixel_blocks]
    with pytest.raises(ValueError):
        selected.install(strata)
    assert not selected._ORIGINALS
    assert original == [b.forward for b in strata.pixel_blocks]


def test_four_block_routing_preserves_learned_calls_and_external_stride(
    strata, monkeypatch
):
    counts = dict(attention=0, mlp=0, conditioning=0)
    with FakeTensorMode():
        _, raw, _ = fake_inputs()

        def conditioning(_):
            counts["conditioning"] += 1
            return raw.reshape(8, 393216, 6, 128).unbind(2)

        def attention(y, **kwargs):
            counts["attention"] += 1
            assert y.dtype == torch.bfloat16 and y.is_contiguous()
            return torch.empty(y.shape, device=y.device, dtype=y.dtype)

        def mlp(y):
            counts["mlp"] += 1
            assert y.dtype == torch.bfloat16 and y.is_contiguous()
            return torch.empty(y.shape, device=y.device, dtype=y.dtype)

        for block in strata.pixel_blocks:
            monkeypatch.setattr(
                block, "_compute_bilinear_dw_adaln_params", conditioning
            )
            monkeypatch.setattr(block.attn, "forward", attention)
            monkeypatch.setattr(block.mlp, "forward", mlp)
        selected.install(strata)
        x = torch.empty((8, 128, 393216), device="cuda").transpose(1, 2)
        for index, block in enumerate(strata.pixel_blocks):
            x = block(x, None, (24, 128, 128), (24, 32, 32), s_cond_bilinear=object())
            assert x.stride() == (
                (50331648, 128, 1) if index < 3 else (50331648, 1, 393216)
            )
        assert counts == dict(attention=4, mlp=4, conditioning=4)


def test_new_case_uses_confirmed_attention_parent(monkeypatch):
    calls = []
    original = decoder_benchmark.apply_experiment

    def dispatch(model, name):
        if name == "pixel-state-row-layout":
            return original(model, name)
        calls.append(name)

    monkeypatch.setattr(decoder_benchmark, "apply_experiment", dispatch)
    monkeypatch.setattr(selected, "install", lambda model: calls.append("state"))
    original(SimpleNamespace(), "pixel-state-row-layout")
    assert calls == ["pixel-attention-axis-last", "state"]


@pytest.mark.parametrize("scope", ("decoder", "backbone"))
def test_new_trial_scope(scope, monkeypatch):
    def stop(*args, **kwargs):
        raise RuntimeError("accepted scope")

    monkeypatch.setattr(torch, "load", stop)
    args = SimpleNamespace(
        scope=scope,
        experiment="pixel-state-row-layout",
        baseline_experiment="pixel-attention-axis-last",
        inputs="unused",
    )
    if scope == "decoder":
        with pytest.raises(RuntimeError, match="accepted scope"):
            decoder_benchmark.run(args)
    else:
        with pytest.raises(ValueError):
            decoder_benchmark.run(args)
