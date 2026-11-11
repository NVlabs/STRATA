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
"""Fuse exact column attention, output layout, and gate; Cirrus-42 (AI agent)."""

import json
import operator
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import torch
import triton
import triton.language as tl
from einops import rearrange
from torch.fx import Node
from triton.language.extra.cuda import libdevice

from screamcast import normalization_model
from screamcast.column_precision import fragment_sum, native_attention, precision_kernel
from screamcast.column_precision_replay import load_operands
from screamcast.cuda_timing import measure


@triton.jit
def gate_kernel(
    Q,
    K,
    V,
    OUT,
    GATE,
    MODE: tl.constexpr,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    Q2: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    K2: tl.constexpr,
    V0: tl.constexpr,
    V1: tl.constexpr,
    V2: tl.constexpr,
):
    group = tl.program_id(0)
    column, head = group // 16, group % 16
    level = tl.arange(0, 32)
    channel = tl.arange(0, 64)
    q = tl.load(
        Q + column * Q0 + head * Q1 + level[:, None] * Q2 + channel[None, :],
        level[:, None] < 24,
        other=0,
    )
    k = tl.load(
        K + column * K0 + head * K1 + level[None, :] * K2 + channel[:, None],
        level[None, :] < 24,
        other=0,
    )
    value = tl.load(
        V + column * V0 + head * V1 + level[:, None] * V2 + channel[None, :],
        level[:, None] < 24,
        other=0,
    )
    scores = tl.dot(q, k) * 0.125
    scores = tl.where(level[None, :] < 24, scores, -float("inf"))
    if MODE == 2:
        maximum = tl.max(scores, 1)
        probability = tl.exp2(
            tl.fma(scores, 1.4426950408889634, -maximum[:, None] * 1.4426950408889634)
        )
    elif MODE == 3:
        probability = libdevice.exp(scores - tl.max(scores, 1)[:, None])
    else:
        scaled = scores * 1.4426950408889634
        maximum_scaled = tl.max(scaled, 1)
        if MODE == 14:
            maximum_scaled = tl.maximum(maximum_scaled, 0.0)
        probability = tl.exp2(scaled - maximum_scaled[:, None])
    if MODE == 7:
        denominator = tl.sum(probability.to(tl.float64), 1).to(tl.float32)
    elif MODE >= 8 and MODE <= 11:
        denominator = fragment_sum(probability, MODE)
    else:
        denominator = tl.sum(probability, 1)
    output = tl.dot(probability.to(tl.bfloat16), value)
    if MODE == 4:
        reciprocal = tl.div_rn(1.0, denominator)
        output = output * reciprocal[:, None]
    elif MODE == 5:
        reciprocal = tl.inline_asm_elementwise(
            "rcp.approx.ftz.f32 $0, $1;",
            constraints="=f,f",
            args=[denominator],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        output = output * reciprocal[:, None]
    elif MODE == 6:
        output = tl.div_rn(output, denominator[:, None])
    elif MODE == 12:
        output = output * tl.exp2(-tl.log2(denominator))[:, None]
    elif MODE == 13:
        log_sum = maximum_scaled + tl.log2(denominator)
        output = output * tl.exp2(maximum_scaled - log_sum)[:, None]
    else:
        output = output / denominator[:, None]
    index = (
        (column // 1024 * 24576 + level[:, None] * 1024 + column % 1024) * 1024
        + head * 64
        + channel[None, :]
    )
    logits = tl.load(GATE + index, level[:, None] < 24, other=0).to(tl.float32)
    rounded = output.to(tl.bfloat16).to(tl.float32)
    tl.store(OUT + index, rounded * tl.sigmoid(logits), level[:, None] < 24)


@triton.jit
def reference_post(ATTENTION, LOGITS, OUT, BLOCK: tl.constexpr):
    # Indexing and FP32 sigmoid boundary from the native generated column gate.
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    channel = index % 1024
    token = index // 1024 % 24576
    batch = index // 25165824
    source = (
        batch * 25165824
        + token % 1024 * 24576
        + channel // 64 * 1536
        + token // 1024 * 64
        + channel % 64
    )
    attention = tl.load(ATTENTION + source).to(tl.float32)
    logits = tl.load(LOGITS + index).to(tl.float32)
    tl.store(OUT + index, attention * tl.sigmoid(logits))


def attention_launch(kernel, q, k, v, out, gate=None):
    operands = (q, k, v, out) if gate is None else (q, k, v, out, gate)
    kernel[(131072,)](
        *operands,
        MODE=9,
        Q0=q.stride(0),
        Q1=q.stride(1),
        Q2=q.stride(2),
        K0=k.stride(0),
        K1=k.stride(1),
        K2=k.stride(2),
        V0=v.stride(0),
        V1=v.stride(1),
        V2=v.stride(2),
        num_warps=4,
    )


def reference_call(q, k, v, gate, attention, out):
    attention_launch(precision_kernel, q, k, v, attention)
    reference_post[(196608,)](
        attention, gate, out, BLOCK=1024, num_warps=4, enable_fp_fusion=True
    )


def candidate_call(q, k, v, gate, out):
    attention_launch(gate_kernel, q, k, v, out, gate)


def benchmark(commit, manifest_path, results):
    manifest = json.loads(Path(manifest_path).read_text())
    if (
        manifest["canonical_backbone_error"]["max_absolute_difference"] != 0
        or len(manifest["operand_files"]) != 12
    ):
        raise ValueError("Requires verified twelve-block canonical column corpus")
    torch.manual_seed(20261020)
    result = dict(
        agent="Cirrus-42 (AI agent)",
        time=datetime.now(timezone.utc).isoformat(),
        commit=commit,
        experiment="column-attention-fused-gate-component",
        corpus_manifest=manifest_path,
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        warmup=5,
        repeats=50,
        cases=[],
        scope="Twelve real QKV blocks and synthetic BF16 gate logits; attention BF16 rounding then native FP32 sigmoid; preallocated outputs; two-kernel control vs fused kernel; no full-network inference claim",
    )
    Path(results).write_text(json.dumps(result, indent=2) + "\n")
    with torch.inference_mode():
        gate = torch.randn((8, 24576, 1024), device="cuda", dtype=torch.bfloat16)
        out = torch.empty_like(gate)
        expected = torch.empty_like(gate)
        for index, entry in enumerate(manifest["operand_files"]):
            q, k, v, captured = load_operands(entry, index)
            native = native_attention(q, k, v)
            native_diff = int(
                (native.view(torch.int16) != captured.view(torch.int16)).sum()
            )
            if native_diff:
                raise RuntimeError(f"Native corpus replay differs: {native_diff}")
            attention = torch.empty_like(captured)
            control = partial(reference_call, q, k, v, gate, attention, expected)
            control()
            reference_post[(196608,)](
                captured, gate, out, BLOCK=1024, num_warps=4, enable_fp_fusion=True
            )
            control_bits = int(
                (out.view(torch.int16) != expected.view(torch.int16)).sum()
            )
            candidate = partial(candidate_call, q, k, v, gate, out)
            candidate()
            bits = int((out.view(torch.int16) != expected.view(torch.int16)).sum())
            row = dict(
                column_block=index,
                native_replay_different_bits=native_diff,
                control_different_bits=control_bits,
                candidate_different_bits=bits,
            )
            row["baseline_before"] = measure(control, 5, 50)
            row["candidate"] = measure(candidate, 5, 50)
            row["baseline_after"] = measure(control, 5, 50)
            result["cases"].append(row)
            Path(results).write_text(json.dumps(result, indent=2) + "\n")
            print(
                "COLUMN_GATE",
                index,
                control_bits,
                bits,
                {
                    k: row[k]["median_ms"]
                    for k in ("baseline_before", "candidate", "baseline_after")
                },
                flush=True,
            )
            del q, k, v, captured, native, attention


@torch.library.custom_op("cirrus42::column_attention_gate", mutates_args=())
def attention_gate(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, logits: torch.Tensor
) -> torch.Tensor:
    if (
        logits.shape != (8, 24576, 1024)
        or logits.dtype != torch.bfloat16
        or not logits.is_cuda
        or not logits.is_contiguous()
    ):
        raise ValueError("Column gate requires fixed contiguous CUDA BF16 gate logits")
    for operand in (q, k, v):
        if (
            operand.shape != (8192, 16, 24, 64)
            or operand.dtype != torch.bfloat16
            or operand.device != logits.device
            or operand.stride(-1) != 1
        ):
            raise ValueError("Column gate requires released BF16 column QKV geometry")
    output = torch.empty_like(logits)
    candidate_call(q, k, v, logits, output)
    return output


@attention_gate.register_fake
def attention_gate_fake(q, k, v, logits):
    return torch.empty_like(logits)


def is_column_core(node):
    return (
        isinstance(node, Node)
        and node.op == "call_function"
        and str(node.target)
        in {
            "cirrus42.column_attention_native_sum.default",
            "cirrus42.column_attention_native_sum",
            "<CustomOpDef(cirrus42::column_attention_native_sum)>",
        }
    )


def column_source(node):
    """Follow the native depth-output layout chain; no arithmetic may intervene."""
    while isinstance(node, Node):
        if is_column_core(node):
            return node
        method = node.op == "call_method" and node.target in {
            "reshape",
            "view",
            "permute",
            "transpose",
            "contiguous",
            "flatten",
        }
        function = node.op == "call_function" and node.target in {
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
        }
        if not (method or function) or not node.args or len(node.users) != 1:
            return None
        node = node.args[0]
    return None


def fuse_column_gate(graph_module):
    replacements = []
    for node in list(graph_module.graph.nodes):
        if (
            node.op != "call_function"
            or node.target not in {operator.mul, torch.mul, torch.ops.aten.mul.Tensor}
            or len(node.args) != 2
        ):
            continue
        for gate, output in (node.args, node.args[::-1]):
            if (
                not isinstance(gate, Node)
                or gate.op != "call_function"
                or gate.target not in {torch.sigmoid, torch.ops.aten.sigmoid.default}
            ):
                continue
            source = column_source(output)
            if source is not None and len(source.args) == 3 and len(source.users) == 1:
                value = node.meta.get("example_value", node.meta.get("val"))
                if (
                    not isinstance(value, torch.Tensor)
                    or value.shape != (8, 24576, 1024)
                    or value.dtype != torch.bfloat16
                ):
                    continue
                replacements.append((node, source, gate.args[0]))
    for node, source, logits in replacements:
        with graph_module.graph.inserting_before(node):
            fused = graph_module.graph.call_function(
                torch.ops.cirrus42.column_attention_gate.default, (*source.args, logits)
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
    expected = sum(is_column_core(node) for node in graph_module.graph.nodes)
    changed = fuse_column_gate(graph_module)
    if expected:
        print("Fused exact column output gates:", changed, "of", expected, flush=True)
        if changed != expected:
            raise RuntimeError(
                f"Expected{expected} column output gates, fused{changed}"
            )
    return normalization_model.backend(graph_module, example_inputs)
