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
"""Check ordered-norm contracts and residual fusion; Cirrus-42 (AI agent)."""

import operator

import pytest
import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx import Graph, GraphModule

from screamcast import normalization_model as ordered


@pytest.mark.parametrize("residual", (False, True))
def test_fake_norm_produces_contiguous_bf16_linear_input(residual):
    with FakeTensorMode():
        x = torch.empty((8, 24576, 1024), device="cuda", dtype=torch.float32)
        if residual:
            projection = torch.empty(x.shape, device="cuda", dtype=torch.bfloat16)
            output = ordered.ordered_norm_add(x, projection)
        else:
            output = ordered.ordered_norm(x, True, False)
        assert output.shape == x.shape
        assert output.dtype == torch.bfloat16
        assert output.is_contiguous()


def test_install_restore_preserves_parameters_and_native_methods():
    strata = nn.Module()
    strata.backbone = nn.Module()
    strata.backbone.bf16_mixed = True
    strata.backbone.blocks = nn.ModuleList()
    strata.register_parameter("sentinel", nn.Parameter(torch.ones(1)))
    for index in range(24):
        block = nn.Module()
        block.do_depthwise_attention = bool(index % 2)
        block.norm1 = nn.LayerNorm(1024, elementwise_affine=False, eps=1e-6)
        block.norm2 = nn.LayerNorm(1024, elementwise_affine=False, eps=1e-6)
        strata.backbone.blocks.append(block)
    strata.eval()
    before = tuple(strata.state_dict())
    ids = [id(p) for p in strata.parameters()]
    try:
        ordered.install(strata)
        assert len(ordered._ORIGINALS) == 48
        assert (
            sum(block.norm1._cirrus42_semantic_norm for block in strata.backbone.blocks)
            == 12
        )
        assert all(block.norm2._cirrus42_norm2 for block in strata.backbone.blocks)
        assert tuple(strata.state_dict()) == before
        assert [id(p) for p in strata.parameters()] == ids
        ordered.restore()
        for block in strata.backbone.blocks:
            for norm in (block.norm1, block.norm2):
                assert "forward" not in norm.__dict__
                assert not hasattr(norm, "_cirrus42_norm2")
    finally:
        ordered.restore()


@pytest.mark.parametrize(
    "norm2, projection_dtype, count",
    ((True, torch.bfloat16, 1), (False, torch.bfloat16, 0), (True, torch.float32, 0)),
)
def test_fusion_requires_norm2_and_native_projection_dtype(
    norm2, projection_dtype, count
):
    graph = Graph()
    x = graph.placeholder("x")
    projection = graph.placeholder("projection")
    x.meta["example_value"] = torch.empty(0, dtype=torch.float32)
    projection.meta["example_value"] = torch.empty(0, dtype=projection_dtype)
    total = graph.call_function(operator.add, (x, projection))
    normalized = graph.call_function(
        torch.ops.cirrus42.ordered_norm.default, (total, False, norm2)
    )
    graph.output(normalized)
    module = GraphModule(nn.Module(), graph)
    assert ordered.fuse_residual_norm(module) == count
    fused = [
        node
        for node in graph.nodes
        if node.op == "call_function"
        and node.target == torch.ops.cirrus42.ordered_norm_add.default
    ]
    assert len(fused) == count
    if count:
        assert fused[0].args == (x, projection)
