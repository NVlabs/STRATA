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
"""Fuse native-order norm with BF16 linear inputs; Cirrus-42 (AI agent)."""

import operator
from types import MethodType

import torch
import triton
from torch.fx import Node

from screamcast import semantic_gate_fusion
from screamcast.normalization_order import native_order_kernel

_ORIGINALS = []


def _validate(input):
    if (
        input.shape != (8, 24576, 1024)
        or input.dtype != torch.float32
        or not input.is_cuda
        or not input.is_contiguous()
    ):
        raise ValueError(
            "Ordered norm requires fixed contiguous CUDA FP32 backbone tokens"
        )


def _execute(a, b, semantic, residual, coalesced):
    _validate(a)
    if residual and (
        b.shape != a.shape
        or b.dtype != torch.bfloat16
        or b.device != a.device
        or not b.is_contiguous()
    ):
        raise ValueError(
            "Ordered residual norm requires contiguous BF16 projection output"
        )
    output = torch.empty(a.shape, device=a.device, dtype=torch.bfloat16)
    native_order_kernel[(triton.cdiv(196608, 64 if semantic else 8),)](
        a,
        b,
        output,
        N=196608,
        SEMANTIC=semantic,
        RESIDUAL=residual,
        COALESCED=coalesced,
        num_warps=16,
        num_stages=1,
        enable_fp_fusion=True,
    )
    return output


@torch.library.custom_op("cirrus42::ordered_norm", mutates_args=())
def ordered_norm(input: torch.Tensor, semantic: bool, norm2: bool) -> torch.Tensor:
    return _execute(input, input, semantic, False, norm2)


@ordered_norm.register_fake
def ordered_norm_fake(input, semantic, norm2):
    return torch.empty(input.shape, device=input.device, dtype=torch.bfloat16)


@torch.library.custom_op("cirrus42::ordered_norm_add", mutates_args=())
def ordered_norm_add(input: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
    return _execute(input, projection, False, True, True)


@ordered_norm_add.register_fake
def ordered_norm_add_fake(input, projection):
    return torch.empty(input.shape, device=input.device, dtype=torch.bfloat16)


def _forward(self, input):
    if (
        not torch.is_autocast_enabled("cuda")
        or torch.get_autocast_dtype("cuda") != torch.bfloat16
    ):
        raise ValueError(
            "Ordered norm fuses the immediate BF16-autocast linear input cast"
        )
    return ordered_norm(input, self._cirrus42_semantic_norm, self._cirrus42_norm2)


def restore():
    for module, had_forward, original in _ORIGINALS:
        if had_forward:
            module.forward = original
        else:
            del module.forward
        del module._cirrus42_semantic_norm
        del module._cirrus42_norm2
    _ORIGINALS.clear()


def install(strata):
    restore()
    if (
        strata.training
        or len(strata.backbone.blocks) != 24
        or not strata.backbone.bf16_mixed
    ):
        raise ValueError(
            "Ordered norm requires the released 24-block BF16 eval backbone"
        )
    norms = []
    for block in strata.backbone.blocks:
        for name in ("norm1", "norm2"):
            norm = getattr(block, name)
            if (
                not isinstance(norm, torch.nn.LayerNorm)
                or norm.normalized_shape != (1024,)
                or norm.eps != 1e-6
                or norm.weight is not None
                or norm.bias is not None
            ):
                raise ValueError(
                    "Ordered norm requires non-affine 1024-channel LayerNorm with epsilon1e-6"
                )
            norms.append(
                (
                    norm,
                    name == "norm1" and not block.do_depthwise_attention,
                    name == "norm2",
                )
            )
    for module, semantic, norm2 in norms:
        _ORIGINALS.append((module, "forward" in module.__dict__, module.forward))
        module._cirrus42_semantic_norm = semantic
        module._cirrus42_norm2 = norm2
        module.forward = MethodType(_forward, module)


def _dtype(node):
    if not isinstance(node, Node):
        return None
    value = node.meta.get("example_value", node.meta.get("val"))
    return value.dtype if isinstance(value, torch.Tensor) else None


def fuse_residual_norm(graph_module):
    changed = 0
    for node in list(graph_module.graph.nodes):
        if (
            node.op != "call_function"
            or node.target != torch.ops.cirrus42.ordered_norm.default
            or len(node.args) != 3
            or node.args[2] is not True
        ):
            continue
        value = node.args[0]
        if (
            not isinstance(value, Node)
            or value.op != "call_function"
            or value.target not in {operator.add, torch.add, torch.ops.aten.add.Tensor}
            or len(value.args) != 2
            or value.kwargs.get("alpha", 1) != 1
        ):
            continue
        a, b = value.args
        if _dtype(a) == torch.bfloat16 and _dtype(b) == torch.float32:
            a, b = b, a
        if _dtype(a) != torch.float32 or _dtype(b) != torch.bfloat16:
            continue
        with graph_module.graph.inserting_before(node):
            fused = graph_module.graph.call_function(
                torch.ops.cirrus42.ordered_norm_add.default, (a, b)
            )
            fused.meta = node.meta.copy()
        node.replace_all_uses_with(fused)
        graph_module.graph.erase_node(node)
        changed += 1
    if changed:
        graph_module.graph.eliminate_dead_code()
        graph_module.graph.lint()
        graph_module.recompile()
    return changed


def backend(graph_module, example_inputs):
    expected = sum(
        node.op == "call_function"
        and node.target == torch.ops.cirrus42.ordered_norm.default
        and node.args[2] is True
        for node in graph_module.graph.nodes
    )
    changed = fuse_residual_norm(graph_module)
    if expected:
        print("Ordered residual norm fusions:", changed, "of", expected, flush=True)
    return semantic_gate_fusion.backend(graph_module, example_inputs)
