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
"""Norm2 FakeTensor, installation and scope contracts; Cirrus-42 (AI agent)."""

from types import SimpleNamespace

import pytest
import torch
from physicsnemo.experimental.models.strata.layers import StrataPixel3DBlock
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode

from screamcast import decoder_benchmark, pixel_norm_mod, pixel_qkv_pack
from screamcast import pixel_norm2_mod as norm


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
    try:
        yield model
    finally:
        norm.restore()
        pixel_norm_mod.restore()
        pixel_qkv_pack.restore()


def test_fake_norm2_is_bf16_contiguous():
    with FakeTensorMode():
        x = torch.empty((8, 128, 393216), device="cuda", dtype=torch.float32).transpose(
            1, 2
        )
        attn = torch.empty(x.shape, device="cuda", dtype=torch.bfloat16)
        raw = torch.empty((8, 393216, 768), device="cuda", dtype=torch.bfloat16)
        out = norm.norm2_modulate(
            x, attn, raw[:, :, 256:384], raw[:, :, 384:512], raw[:, :, 512:640]
        )
        assert (
            out.shape == x.shape and out.dtype == torch.bfloat16 and out.is_contiguous()
        )


def test_restore_preserves_norm1_forward_parameters_and_other_operations(strata):
    params = [id(p) for p in strata.parameters()]
    keys = tuple(strata.state_dict())
    forwards = [b.forward for b in strata.pixel_blocks]
    children = [
        (
            b.attn,
            b.norm1,
            b.norm2,
            b.mlp,
            b.adaln_bilinear_dw_proj,
            b._compute_bilinear_dw_adaln_params,
        )
        for b in strata.pixel_blocks
    ]
    norm.install(strata)
    assert len(norm._ORIGINALS) == 4
    assert all(b.forward.__func__ is norm._forward for b in strata.pixel_blocks)
    assert params == [id(p) for p in strata.parameters()]
    assert keys == tuple(strata.state_dict())
    assert children == [
        (
            b.attn,
            b.norm1,
            b.norm2,
            b.mlp,
            b.adaln_bilinear_dw_proj,
            b._compute_bilinear_dw_adaln_params,
        )
        for b in strata.pixel_blocks
    ]
    norm.restore()
    assert forwards == [b.forward for b in strata.pixel_blocks]
    assert all(
        b.forward.__func__ is pixel_norm_mod._forward for b in strata.pixel_blocks
    )


@pytest.mark.parametrize(
    "invalid", ("epsilon", "affine", "count", "training", "parent")
)
def test_invalid_contract_rejected_before_patch(strata, invalid):
    if invalid == "epsilon":
        strata.pixel_blocks[-1].norm2.eps = 1e-5
    elif invalid == "affine":
        strata.pixel_blocks[-1].norm2 = nn.LayerNorm(128, eps=1e-6, device="meta")
    elif invalid == "count":
        strata.pixel_blocks = strata.pixel_blocks[:3]
    elif invalid == "training":
        strata.train()
    else:
        pixel_norm_mod.restore()
    forwards = [b.forward for b in strata.pixel_blocks]
    with pytest.raises(ValueError):
        norm.install(strata)
    assert not norm._ORIGINALS
    assert forwards == [b.forward for b in strata.pixel_blocks]


def test_new_trial_scope_reaches_data_loading(monkeypatch):
    def stop(*args, **kwargs):
        raise RuntimeError("accepted scope")

    monkeypatch.setattr(torch, "load", stop)
    args = SimpleNamespace(
        scope="decoder",
        experiment="pixel-norm2-modulation",
        baseline_experiment="pixel-norm1-modulation",
        inputs="unused",
    )
    with pytest.raises(RuntimeError, match="accepted scope"):
        decoder_benchmark.run(args)
