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
"""Do not fuse column gates into neighborhood layout; Cirrus-42 (AI agent)."""

import operator

import pytest
import torch
from torch import nn
from torch.fx import Graph, GraphModule

from screamcast import (
    column_attention,  # noqa: F401
    column_precision,  # noqa: F401
    semantic_attention_layout,  # noqa: F401
)
from screamcast.semantic_gate_fusion import fuse_output_gate


@pytest.mark.parametrize("persistent", (False, True))
@pytest.mark.parametrize(
    "column_function",
    (
        torch.nn.functional.scaled_dot_product_attention,
        torch.ops.cirrus42.column_attention.default,
        torch.ops.cirrus42.column_attention_native_sum.default,
    ),
    ids=("native_sdpa", "short_column_op", "native_sum_column_op"),
)
def test_only_direct_semantic_output_gate_is_fused(column_function, persistent):
    graph = Graph()
    q, k, v, logits = [graph.placeholder(name) for name in ("q", "k", "v", "logits")]
    core = graph.call_function(
        torch.ops.cirrus42.semantic_attention_core.default, (q, k, v, persistent)
    )
    view = graph.call_method("reshape", (core, (8, 24576, 1024)))
    gate = graph.call_function(torch.sigmoid, (logits,))
    gated = graph.call_function(operator.mul, (view, gate))
    column_input = graph.call_method("reshape", (view, (8192, 16, 24, 64)))
    column = graph.call_function(
        column_function,
        (column_input, column_input, column_input),
    )
    column = graph.call_method("reshape", (column, (8, 24576, 1024)))
    column_gate = graph.call_function(torch.sigmoid, (logits,))
    column_gated = graph.call_function(operator.mul, (column, column_gate))
    graph.output((gated, column_gated))
    module = GraphModule(nn.Module(), graph)
    assert fuse_output_gate(module) == 1
    fused = [
        node
        for node in graph.nodes
        if node.op == "call_function"
        and node.target == torch.ops.cirrus42.semantic_output_gate.default
    ]
    assert len(fused) == 1
    assert fused[0].args == (core, logits)
    assert column_gate.target is torch.sigmoid
    assert column_gated in graph.nodes
