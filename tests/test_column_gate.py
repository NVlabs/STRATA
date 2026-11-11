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
"""Validate column gate fusion boundaries; Cirrus-42 (AI agent)."""

import operator

import pytest
import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx import Graph, GraphModule

from screamcast import column_gate


def test_fake_gate_preserves_contiguous_bf16_projection_input():
    with FakeTensorMode():
        q = torch.empty((8192, 16, 24, 64), device="cuda", dtype=torch.bfloat16)
        logits = torch.empty((8, 24576, 1024), device="cuda", dtype=torch.bfloat16)
        result = column_gate.attention_gate(q, q, q, logits)
        assert result.shape == logits.shape
        assert result.dtype == logits.dtype
        assert result.is_contiguous()


@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize(
    "case", ("valid", "arithmetic", "wrong_geometry", "shared_core")
)
def test_fusion_requires_column_layout_gate_and_fixed_geometry(case, reverse):
    graph = Graph()
    q, k, v, logits = [graph.placeholder(name) for name in ("q", "k", "v", "logits")]
    core = graph.call_function(
        torch.ops.cirrus42.column_attention_native_sum.default, (q, k, v)
    )
    output = graph.call_method("reshape", (core, 8, 1024, 16, 24, 64))
    output = graph.call_method("permute", (output, 0, 3, 1, 2, 4))
    output = graph.call_method("reshape", (output, 8, 24576, 1024))
    if case == "arithmetic":
        output = graph.call_function(torch.sin, (output,))
    sigmoid = graph.call_function(torch.sigmoid, (logits,))
    product = graph.call_function(
        operator.mul, (sigmoid, output) if reverse else (output, sigmoid)
    )
    shape = (8, 12288 if case == "wrong_geometry" else 24576, 1024)
    with FakeTensorMode():
        product.meta["example_value"] = torch.empty(
            shape, device="cuda", dtype=torch.bfloat16
        )
    graph.output((product, core) if case == "shared_core" else product)
    module = GraphModule(nn.Module(), graph)
    count = column_gate.fuse_column_gate(module)
    assert count == (1 if case == "valid" else 0)
    fused = [
        node
        for node in graph.nodes
        if node.op == "call_function"
        and node.target == torch.ops.cirrus42.column_attention_gate.default
    ]
    assert len(fused) == count
    if count:
        assert fused[0].args == (q, k, v, logits)
        assert not any(column_gate.is_column_core(node) for node in graph.nodes)
