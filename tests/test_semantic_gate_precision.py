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
"""Keep column gates intact when pinning semantic gates; Cirrus-42 (AI agent)."""

import operator

import torch
from torch import nn
from torch.fx import Graph, GraphModule

from screamcast import semantic_attention_layout  # noqa: F401

# Register the semantic core operator referenced by the FX graph.
from screamcast.semantic_gate_precision import pin_gate_rounding


def test_only_semantic_attention_gate_is_pinned():
    graph = Graph()
    q, k, v, logits = [graph.placeholder(name) for name in ("q", "k", "v", "logits")]
    core = graph.call_function(
        torch.ops.cirrus42.semantic_attention_core.default, (q, k, v)
    )
    view = graph.call_method("reshape", (core, (8, 24576, 1024)))
    gate = graph.call_function(torch.sigmoid, (logits,))
    gated = graph.call_function(operator.mul, (view, gate))
    # Even with an upstream semantic dependency, a column-attention result
    # must retain its original sigmoid semantics.
    column_input = graph.call_method("reshape", (view, (8192, 16, 24, 64)))
    column_attention = graph.call_function(
        torch.nn.functional.scaled_dot_product_attention,
        (column_input, column_input, column_input),
    )
    column = graph.call_method("reshape", (column_attention, (8, 24576, 1024)))
    column_gate = graph.call_function(torch.sigmoid, (logits,))
    column_gated = graph.call_function(operator.mul, (column, column_gate))
    graph.output((gated, column_gated))
    module = GraphModule(nn.Module(), graph)
    assert pin_gate_rounding(module) == 1
    assert gate.target == torch.ops.cirrus42.rounded_gate_sigmoid.default
    assert column_gate.target is torch.sigmoid
