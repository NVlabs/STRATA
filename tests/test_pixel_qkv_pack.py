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
"""Pixel packing installation and FakeTensor contracts; Cirrus-42 (AI agent)."""

import pytest
import torch
from physicsnemo.experimental.models.strata.layers import Natten3DSelfAttention
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode

from screamcast import attention_layout, pixel_persistent, pixel_qkv_pack


@pytest.fixture
def strata():
    model = nn.Module()
    with torch.device("meta"):
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
    try:
        yield model
    finally:
        pixel_persistent.restore()
        pixel_qkv_pack.restore()


def test_fake_pack_allocates_distinct_native_layout_buffers():
    with FakeTensorMode():
        raw = torch.empty((8, 393216, 384), device="cuda", dtype=torch.bfloat16)
        outputs = pixel_qkv_pack.packed_qkv(raw)
        assert len({id(t) for t in outputs}) == 3
        assert all(
            t.shape == (8, 393216, 2, 64)
            and t.dtype == torch.bfloat16
            and t.is_contiguous()
            for t in outputs
        )


def test_install_restore_preserves_parameters_and_state(strata):
    originals = [m.forward for m in strata.pixel_blocks]
    params = [id(p) for p in strata.parameters()]
    state = tuple(strata.state_dict())
    pixel_qkv_pack.install(strata)
    assert len(pixel_qkv_pack._ORIGINALS) == 4
    assert all(
        m.forward.__func__ is pixel_qkv_pack._forward for m in strata.pixel_blocks
    )
    assert params == [id(p) for p in strata.parameters()]
    assert state == tuple(strata.state_dict())
    pixel_qkv_pack.restore()
    assert originals == [m.forward for m in strata.pixel_blocks]


@pytest.mark.parametrize("invalid", ("gated", "epsilon", "count"))
def test_invalid_module_rejected_before_patching(strata, invalid):
    if invalid == "gated":
        strata.pixel_blocks[-1].gated_attention = True
    elif invalid == "epsilon":
        strata.pixel_blocks[-1].q_norm.eps = 1e-5
    else:
        strata.pixel_blocks = strata.pixel_blocks[:-1]
    with pytest.raises(ValueError):
        pixel_qkv_pack.install(strata)
    assert not pixel_qkv_pack._ORIGINALS


def test_rope_rejected(strata):
    pixel_qkv_pack.install(strata)
    with pytest.raises(ValueError, match="no RoPE"):
        strata.pixel_blocks[0](
            torch.empty(0),
            latent_dhw=(24, 128, 128),
            rope_tables=(torch.empty(0), torch.empty(0)),
        )


def test_persistent_fake_preserves_shape_and_dtype():
    with FakeTensorMode():
        q = torch.empty((8, 393216, 2, 64), device="cuda", dtype=torch.bfloat16)
        out = pixel_persistent.persistent_core(q, q, q)
        assert out.shape == q.shape and out.dtype == q.dtype and out.is_contiguous()


def test_persistent_install_requires_validated_pixel_pack(strata):
    with pytest.raises(ValueError, match="validated packed"):
        pixel_persistent.install(strata)
    assert attention_layout.attention_core is pixel_persistent._NATIVE_CORE


def test_persistent_restore_preserves_packed_methods_and_parameters(strata):
    pixel_qkv_pack.install(strata)
    params = [id(p) for p in strata.parameters()]
    state = tuple(strata.state_dict())
    forwards = [m.forward for m in strata.pixel_blocks]
    pixel_persistent.install(strata)
    assert attention_layout.attention_core is pixel_persistent.persistent_core
    assert params == [id(p) for p in strata.parameters()]
    assert state == tuple(strata.state_dict())
    pixel_persistent.restore()
    assert attention_layout.attention_core is pixel_persistent._NATIVE_CORE
    assert forwards == [m.forward for m in strata.pixel_blocks]
