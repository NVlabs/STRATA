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
"""Packed projection ownership/Fake contracts; Cirrus-42 (AI agent)."""

from types import SimpleNamespace

import pytest
import torch
from physicsnemo.experimental.models.strata.layers import StrataPixel3DBlock
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode

from screamcast import (
    attention_layout,
    decoder_benchmark,
    pixel_axis_attention,
    pixel_norm2_mod,
    pixel_norm_mod,
    pixel_persistent,
    pixel_qkv_pack,
)
from screamcast import pixel_packed_projection_model as selected
from screamcast import pixel_state_layout_model as parent


@pytest.fixture
def strata(monkeypatch):
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
    monkeypatch.setattr(
        attention_layout, "attention_core", pixel_persistent.persistent_core
    )
    pixel_axis_attention.install(model, last_only=True)
    parent.install(model)
    try:
        yield model
    finally:
        selected.restore()
        parent.restore()
        pixel_axis_attention.restore()
        pixel_norm2_mod.restore()
        pixel_norm_mod.restore()
        pixel_qkv_pack.restore()


def inputs(row_input):
    x = torch.empty(
        (8, 393216, 128) if row_input else (8, 128, 393216),
        device="cuda",
        dtype=torch.float32,
    )
    if not row_input:
        x = x.transpose(1, 2)
    raw = torch.empty((8, 393216, 768), device="cuda", dtype=torch.bfloat16)
    attn = torch.empty(x.shape, device="cuda", dtype=torch.bfloat16)
    return x, raw, attn


@pytest.mark.parametrize("row_input", (False, True))
def test_fake_norm_and_update_preserve_external_contract(row_input):
    with FakeTensorMode():
        x, raw, attn = inputs(row_input)
        norm = selected.norm2(
            x,
            attn,
            raw[:, :, 256:384],
            raw[:, :, 384:512],
            raw[:, :, 512:640],
            False,
            row_input,
        )
        assert (
            norm.shape == x.shape
            and norm.dtype == torch.bfloat16
            and norm.is_contiguous()
        )
        for row_output in (False, True):
            state = selected.update(
                x,
                attn,
                raw[:, :, 256:384],
                raw[:, :, 640:768],
                attn,
                False,
                row_input,
                row_output,
            )
            assert state.shape == x.shape and state.dtype == torch.float32
            assert state.stride() == (
                (50331648, 128, 1) if row_output else (50331648, 1, 393216)
            )


@pytest.mark.parametrize(
    "invalid",
    ("state_stride", "state_dtype", "attention_stride", "mod_stride", "mlp_dtype"),
)
def test_fake_rejects_invalid_metadata_before_kernel(invalid):
    with FakeTensorMode():
        x, raw, attn = inputs(True)
        mlp = attn
        if invalid == "state_stride":
            x = torch.empty((8, 128, 393216), device="cuda").transpose(1, 2)
        elif invalid == "state_dtype":
            x = x.to(torch.bfloat16)
        elif invalid == "attention_stride":
            attn = torch.empty(
                (8, 128, 393216), device="cuda", dtype=torch.bfloat16
            ).transpose(1, 2)
        elif invalid == "mod_stride":
            raw = torch.empty((8, 393216, 128), device="cuda", dtype=torch.bfloat16)
        else:
            mlp = mlp.to(torch.float32)
        gate = raw if invalid == "mod_stride" else raw[:, :, 256:384]
        gate2 = raw if invalid == "mod_stride" else raw[:, :, 640:768]
        with pytest.raises(ValueError):
            selected.update(x, attn, gate, gate2, mlp, False, True, True)


def test_install_restore_preserves_all_learned_state(strata):
    parameters = [id(p) for p in strata.parameters()]
    modules = [id(m) for m in strata.modules()]
    state = tuple(strata.state_dict())
    originals = [(b.forward, b.attn.forward) for b in strata.pixel_blocks]
    selected.install(strata)
    assert [b.forward.__func__ for b in strata.pixel_blocks] == list(
        selected._BLOCK_FORWARDS
    )
    assert [b.attn.forward.__func__ for b in strata.pixel_blocks] == [
        selected._ATTN_FORWARDS[False]
    ] * 3 + [selected._ATTN_FORWARDS[True]]
    assert parameters == [id(p) for p in strata.parameters()]
    assert modules == [id(m) for m in strata.modules()] and state == tuple(
        strata.state_dict()
    )
    selected.restore()
    assert originals == [(b.forward, b.attn.forward) for b in strata.pixel_blocks]


@pytest.mark.parametrize("invalid", ("training", "block_forward", "attention_forward"))
def test_invalid_parent_is_rejected_without_partial_patch(strata, monkeypatch, invalid):
    if invalid == "training":
        strata.train()
    elif invalid == "block_forward":
        monkeypatch.setattr(strata.pixel_blocks[1], "forward", lambda: None)
    else:
        monkeypatch.setattr(strata.pixel_blocks[1].attn, "forward", lambda: None)
    original = [(b.forward, b.attn.forward) for b in strata.pixel_blocks]
    with pytest.raises(ValueError):
        selected.install(strata)
    assert not selected._ORIGINALS
    assert original == [(b.forward, b.attn.forward) for b in strata.pixel_blocks]


def test_four_blocks_route_every_learned_call_and_restore_exit(strata, monkeypatch):
    counts = dict(
        conditioning=0,
        qkv=0,
        projection=0,
        drop=0,
        mlp=0,
        native_pack=0,
        axis_pack=0,
        native_core=0,
        axis_core=0,
    )
    selected.install(strata)
    with FakeTensorMode(), torch.autocast("cuda", dtype=torch.bfloat16):
        x, raw, attn = inputs(False)

        def conditioning(_):
            counts["conditioning"] += 1
            return raw.reshape(8, 393216, 6, 128).unbind(2)

        def qkv(value):
            counts["qkv"] += 1
            return torch.empty(
                (8, 393216, 384), device=value.device, dtype=torch.bfloat16
            )

        def projection(value):
            counts["projection"] += 1
            assert value.shape == (8, 393216, 128) and value.is_contiguous()
            return torch.empty_like(value)

        def drop(value):
            counts["drop"] += 1
            return value

        def mlp(value):
            counts["mlp"] += 1
            assert value.dtype == torch.bfloat16 and value.is_contiguous()
            return torch.empty_like(value)

        def pack(value, axis):
            counts["axis_pack" if axis else "native_pack"] += 1
            return tuple(
                torch.empty((8, 393216, 2, 64), device=value.device, dtype=value.dtype)
                for _ in range(3)
            )

        def core(q, k, v, axis):
            counts["axis_core" if axis else "native_core"] += 1
            return torch.empty_like(q)

        monkeypatch.setattr(
            pixel_qkv_pack, "packed_qkv", lambda value: pack(value, False)
        )
        monkeypatch.setattr(
            pixel_axis_attention, "packed_qkv", lambda value: pack(value, True)
        )
        monkeypatch.setattr(
            pixel_persistent, "persistent_core", lambda q, k, v: core(q, k, v, False)
        )
        monkeypatch.setattr(
            pixel_axis_attention, "attention", lambda q, k, v: core(q, k, v, True)
        )
        for block in strata.pixel_blocks:
            monkeypatch.setattr(
                block, "_compute_bilinear_dw_adaln_params", conditioning
            )
            monkeypatch.setattr(block.attn.qkv, "forward", qkv)
            monkeypatch.setattr(block.attn.proj, "forward", projection)
            monkeypatch.setattr(block.attn.proj_drop, "forward", drop)
            monkeypatch.setattr(block.mlp, "forward", mlp)
        for index, block in enumerate(strata.pixel_blocks):
            x = block(x, None, (24, 128, 128), (24, 32, 32), s_cond_bilinear=object())
            assert x.stride() == (
                (50331648, 128, 1) if index < 3 else (50331648, 1, 393216)
            )
    assert counts == dict(
        conditioning=4,
        qkv=4,
        projection=4,
        drop=4,
        mlp=4,
        native_pack=3,
        axis_pack=1,
        native_core=3,
        axis_core=1,
    )


def test_new_case_inherits_confirmed_row_parent(monkeypatch):
    calls = []
    original = decoder_benchmark.apply_experiment

    def dispatch(model, name):
        if name == "pixel-packed-output-projection":
            return original(model, name)
        calls.append(name)

    monkeypatch.setattr(decoder_benchmark, "apply_experiment", dispatch)
    monkeypatch.setattr(selected, "install", lambda model: calls.append("packed"))
    original(SimpleNamespace(), "pixel-packed-output-projection")
    assert calls == ["pixel-state-row-layout", "packed"]


@pytest.mark.parametrize("scope", ("decoder", "backbone"))
def test_model_trial_requires_decoder_scope(scope, monkeypatch):
    def stop(*args, **kwargs):
        raise RuntimeError("accepted scope")

    monkeypatch.setattr(torch, "load", stop)
    args = SimpleNamespace(
        scope=scope,
        experiment="pixel-packed-output-projection",
        baseline_experiment="pixel-state-row-layout",
        inputs="unused",
    )
    with pytest.raises(RuntimeError if scope == "decoder" else ValueError):
        decoder_benchmark.run(args)
