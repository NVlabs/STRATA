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
"""Fused conditioning installation and shape contracts; Cirrus-42 (AI agent)."""

import pytest
import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode

from screamcast import conditioning_conv_norm_unroll as fused
from screamcast import conditioning_vector


@pytest.fixture
def strata():
    model = nn.Module()
    model.pixel_blocks = nn.ModuleList()
    for _ in range(4):
        block = nn.Module()
        block.adaln_bilinear_dw_conv = nn.Conv2d(
            1024, 1024, 5, groups=1024, device="meta"
        )
        block.adaln_bilinear_dw_norm = nn.RMSNorm(
            1024, eps=None, elementwise_affine=False, device="meta"
        )
        block.adaln_bilinear_dw_proj = nn.Linear(1024, 768, device="meta")
        model.pixel_blocks.append(block)
    model.eval()
    conditioning_vector.install(model)
    try:
        yield model
    finally:
        fused.restore()
        conditioning_vector.restore()


def test_fake_fused_norm_returns_native_projection_input():
    with FakeTensorMode():
        a = torch.empty(
            (192, 132, 132, 512), device="cuda", dtype=torch.bfloat16
        ).permute(0, 3, 1, 2)
        weight = torch.empty((1024, 1, 5, 5), device="cuda", dtype=torch.float32)
        bias = torch.empty(1024, device="cuda", dtype=torch.float32)
        out = fused.fused_norm(a, a, weight, bias)
        assert out.shape == (8, 393216, 1024) and out.dtype == torch.bfloat16
        assert out.is_contiguous()


def test_install_restore_preserves_original_parameters_state_and_projection(strata):
    ids = [id(p) for p in strata.parameters()]
    names = tuple(strata.state_dict())
    methods = [b._compute_bilinear_dw_adaln_params for b in strata.pixel_blocks]
    projections = [b.adaln_bilinear_dw_proj for b in strata.pixel_blocks]
    fused.install(strata)
    assert len(fused._ORIGINALS) == 4
    assert all(
        b._compute_bilinear_dw_adaln_params.__func__ is fused.compute
        for b in strata.pixel_blocks
    )
    assert ids == [id(p) for p in strata.parameters()]
    assert names == tuple(strata.state_dict())
    assert projections == [b.adaln_bilinear_dw_proj for b in strata.pixel_blocks]
    fused.restore()
    assert methods == [b._compute_bilinear_dw_adaln_params for b in strata.pixel_blocks]


@pytest.mark.parametrize("invalid", ("epsilon", "kernel", "count", "training"))
def test_invalid_geometry_rejected_before_any_patch(strata, invalid):
    if invalid == "epsilon":
        strata.pixel_blocks[-1].adaln_bilinear_dw_norm.eps = 1e-6
    elif invalid == "kernel":
        strata.pixel_blocks[-1].adaln_bilinear_dw_conv = nn.Conv2d(
            1024, 1024, 3, groups=1024, device="meta"
        )
    elif invalid == "count":
        strata.pixel_blocks = strata.pixel_blocks[:3]
    else:
        strata.train()
    methods = [b._compute_bilinear_dw_adaln_params for b in strata.pixel_blocks]
    with pytest.raises(ValueError):
        fused.install(strata)
    assert not fused._ORIGINALS
    assert methods == [b._compute_bilinear_dw_adaln_params for b in strata.pixel_blocks]
