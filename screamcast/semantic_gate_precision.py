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
"""Preserve native BF16 semantic sigmoid gates; Cirrus-42 (AI agent)."""

import operator

import torch
import triton
import triton.language as tl
from einops import rearrange
from torch._inductor.compile_fx import compile_fx
from torch.fx import Node


@triton.jit
def sigmoid_kernel(x, out, N: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(x + index, index < N, other=0).to(tl.float32)
    result = tl.sigmoid(value)
    tl.store(out + index, result, index < N)


@torch.library.custom_op("cirrus42::rounded_gate_sigmoid", mutates_args=())
def rounded_sigmoid(x: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("Semantic gate requires contiguous BF16 linear output")
    out = torch.empty_like(x)
    sigmoid_kernel[(triton.cdiv(x.numel(), 1024),)](
        x, out, N=x.numel(), BLOCK=1024, num_warps=4
    )
    return out


@rounded_sigmoid.register_fake
def rounded_sigmoid_fake(x):
    return torch.empty_like(x)


def is_semantic_core(node):
    return (
        isinstance(node, Node)
        and node.op == "call_function"
        and str(node.target)
        in {
            "cirrus42.semantic_attention_core.default",
            "cirrus42.semantic_attention_core",
            "<CustomOpDef(cirrus42::semantic_attention_core)>",
        }
    )


def semantic_output_view(node):
    """Follow layout-only operations; stop at other attention or arithmetic."""
    if not isinstance(node, Node):
        return False
    if is_semantic_core(node):
        return True
    if node.op == "call_method" and node.target in {
        "reshape",
        "view",
        "permute",
        "transpose",
        "contiguous",
        "flatten",
    }:
        return semantic_output_view(node.args[0])
    if node.op == "call_function" and node.target in {
        rearrange,
        torch.reshape,
        torch.permute,
        torch.transpose,
        torch.flatten,
        torch.clone,
        torch.ops.aten.view.default,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.permute.default,
        torch.ops.aten.clone.default,
    }:
        return semantic_output_view(node.args[0])
    return False


def pin_gate_rounding(graph_module):
    """Replace only sigmoid gates directly multiplying semantic attention output."""
    gates = set()
    for node in graph_module.graph.nodes:
        if (
            node.op != "call_function"
            or node.target not in {operator.mul, torch.mul, torch.ops.aten.mul.Tensor}
            or len(node.args) != 2
        ):
            continue
        for gate, output in (node.args, node.args[::-1]):
            if (
                isinstance(gate, Node)
                and gate.op == "call_function"
                and gate.target in {torch.sigmoid, torch.ops.aten.sigmoid.default}
                and semantic_output_view(output)
            ):
                gates.add(gate)
    for gate in gates:
        gate.target = torch.ops.cirrus42.rounded_gate_sigmoid.default
    if gates:
        graph_module.graph.lint()
        graph_module.recompile()
    return len(gates)


def backend(graph_module, example_inputs):
    cores = [node for node in graph_module.graph.nodes if is_semantic_core(node)]
    if cores:
        changed = pin_gate_rounding(graph_module)
        print("Pinned semantic BF16 sigmoid gates:", changed, flush=True)
        if changed != len(cores):
            raise RuntimeError(
                f"Expected {len(cores)} semantic sigmoid gates, pinned {changed}"
            )
    return compile_fx(graph_module, example_inputs)
