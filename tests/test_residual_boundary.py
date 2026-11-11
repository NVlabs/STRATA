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
"""Check residual output layout and restoration; Cirrus-42 (AI agent)."""

import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode

from screamcast import residual_boundary, token_layout


def test_fake_residual_returns_channel_major_fp32_tokens():
    with FakeTensorMode():
        input = torch.empty((8, 24576, 1024), device="cuda", dtype=torch.float32)
        projection = torch.empty(input.shape, device="cuda", dtype=torch.bfloat16)
        output = residual_boundary.residual_channel_major(input, projection, projection)
        assert output.shape == input.shape
        assert output.dtype == torch.float32
        assert output.stride() == (25165824, 1, 24576)
        assert output is not input


def test_install_restore_preserves_weights_and_both_layout_hooks():
    strata = nn.Module()
    strata.backbone = nn.Module()
    strata.backbone.blocks = nn.ModuleList([nn.Identity() for _ in range(24)])
    strata.register_parameter("sentinel", nn.Parameter(torch.ones(1)))
    strata.eval()
    before = tuple(strata.state_dict())
    ids = [id(p) for p in strata.parameters()]
    last = strata.backbone.blocks[-1]
    native = last.forward
    try:
        token_layout.install(strata)
        assert len(token_layout._HANDLES) == 2
        residual_boundary.install(strata)
        assert len(token_layout._HANDLES) == 1
        assert last.forward.__func__ is residual_boundary._last_forward
        assert tuple(strata.state_dict()) == before
        assert [id(p) for p in strata.parameters()] == ids
        residual_boundary.restore()
        assert last.forward == native
        assert len(token_layout._HANDLES) == 2
        assert "forward" not in last.__dict__
    finally:
        residual_boundary.restore()
        token_layout.restore()
