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
"""Round sigmoid inside semantic output layout; Cirrus-42 (AI agent)."""

import operator

import torch
import triton
import triton.language as tl
from torch._inductor.compile_fx import compile_fx
from torch.fx import Node

from screamcast.semantic_gate_precision import is_semantic_core, semantic_output_view


@triton.jit
def output_kernel(attention, logits, output, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # Fixed B8,D24,H32,W32,16 heads of 64; native query tile (8,4,8).
    channel = index % 1024
    w = index // 1024 % 32
    h = index // 32768 % 32
    d = index // 1048576 % 24
    batch = index // 25165824
    token = batch * 4 + w // 8
    token = token * 8 + h // 4
    token = token * 3 + d // 8
    token = token * 8 + w % 8
    token = token * 4 + h % 4
    token = token * 8 + d % 8
    value = tl.load(attention + token * 1024 + channel).to(tl.float32)
    gate = tl.load(logits + index).to(tl.float32)
    # This materialized rounding is necessary to reproduce the native boundary.
    gate = tl.sigmoid(gate).to(tl.bfloat16).to(tl.float32)
    tl.store(output + index, value * gate)


@torch.library.custom_op("cirrus42::semantic_output_gate", mutates_args=())
def output_gate(attention: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    if (
        attention.shape != (8, 24576, 16, 64)
        or logits.shape != (8, 24576, 1024)
        or attention.dtype != torch.bfloat16
        or logits.dtype != torch.bfloat16
        or not attention.is_contiguous()
        or not logits.is_contiguous()
    ):
        raise ValueError("Semantic gate fusion requires fixed contiguous BF16 operands")
    output = torch.empty_like(logits)
    output_kernel[(triton.cdiv(logits.numel(), 1024),)](
        attention,
        logits,
        output,
        BLOCK=1024,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output


@output_gate.register_fake
def output_gate_fake(attention, logits):
    return torch.empty_like(logits)


def fuse_output_gate(graph_module):
    replacements = []
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
                source = output
                while not is_semantic_core(source):
                    source = source.args[0]
                replacements.append((node, source, gate.args[0]))
    for node, source, logits in replacements:
        with graph_module.graph.inserting_before(node):
            fused = graph_module.graph.call_function(
                torch.ops.cirrus42.semantic_output_gate.default, (source, logits)
            )
            fused.meta = node.meta.copy()
        node.replace_all_uses_with(fused)
        graph_module.graph.erase_node(node)
    if replacements:
        graph_module.graph.eliminate_dead_code()
        graph_module.graph.lint()
        graph_module.recompile()
    return len(replacements)


def backend(graph_module, example_inputs):
    cores = [node for node in graph_module.graph.nodes if is_semantic_core(node)]
    if cores:
        changed = fuse_output_gate(graph_module)
        print("Fused rounded semantic output gates:", changed, flush=True)
        if changed != len(cores):
            raise RuntimeError(f"Expected {len(cores)} semantic gates, fused {changed}")
    return compile_fx(graph_module, example_inputs)
