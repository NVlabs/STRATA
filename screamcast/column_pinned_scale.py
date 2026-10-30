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
"""Pin the native column scale rounding boundary; Cirrus-42 (AI agent)."""

import argparse
import json
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import torch
import triton
import triton.language as tl
from physicsnemo.experimental.models.strata import layers
from triton.language.extra.cuda import libdevice

from screamcast import column_rms_attention as selected
from screamcast.benchmark_statistics import latency_gate
from screamcast.column_precision import fragment_sum
from screamcast.column_rms_attention import evaluate as native_evaluate
from screamcast.cuda_timing import measure

_NATIVE_CORE = selected.normalized_attention_gate


@triton.jit
def pinned_scale_kernel(INPUT, OUT, GATE, MODE: tl.constexpr):
    group = tl.program_id(0)
    column, head = group // 16, group % 16
    level = tl.arange(0, 32)
    channel = tl.arange(0, 64)
    token = column // 1024 * 24576 + level * 1024 + column % 1024
    source = token[:, None] * 3072 + head * 64 + channel[None, :]
    raw_q = tl.load(INPUT + source, level[:, None] < 24, other=0).to(tl.float32)
    raw_k = tl.load(INPUT + source + 1024, level[:, None] < 24, other=0).to(tl.float32)
    value = tl.load(INPUT + source + 2048, level[:, None] < 24, other=0)
    qsum = tl.sum(raw_q * raw_q, 1)
    ksum = tl.sum(raw_k * raw_k, 1)
    qscale = libdevice.rsqrt(qsum / 64.0 + 1e-6)
    kscale = libdevice.rsqrt(ksum / 64.0 + 1e-6)
    q = (raw_q * qscale[:, None]).to(tl.bfloat16)
    k = (raw_k * kscale[:, None]).to(tl.bfloat16).trans(1, 0)
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
        scaled = tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[scores, 1.4426950408889634],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
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


def evaluate(raw, gate, warps):
    output = torch.empty_like(gate)
    pinned_scale_kernel[(131072,)](
        raw, output, gate, MODE=9, num_warps=warps, enable_fp_fusion=True
    )
    return output


@torch.library.custom_op("cirrus42::normalized_column_pinned_scale", mutates_args=())
def attention(raw: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    selected.validate_operands(raw, logits)
    return evaluate(raw, logits, 2)


@attention.register_fake
def attention_fake(raw, logits):
    return torch.empty_like(logits)


def restore():
    selected.normalized_attention_gate = _NATIVE_CORE


def install(strata):
    restore()
    if strata.training or len(selected._ORIGINALS) != 12:
        raise ValueError(
            "Pinned-scale columns require twelve validated eval fused-RMS modules"
        )
    modules = [
        module
        for module in strata.backbone.modules()
        if isinstance(module, layers.Natten3DSelfAttention)
        and module.do_depthwise_attention
    ]
    registered = [module for module, _, _ in selected._ORIGINALS]
    if len(modules) != 12 or any(
        module.training
        or module not in registered
        or module.forward.__func__ is not selected._forward
        for module in modules
    ):
        raise ValueError(
            "Pinned-scale columns require this model's validated fused-RMS methods"
        )
    selected.normalized_attention_gate = attention


def run(commit, path, registered=False):
    result = dict(
        agent="Cirrus-42 (AI agent)",
        time=datetime.now(timezone.utc).isoformat(),
        commit=commit,
        experiment="registered-column-pinned-scale"
        if registered
        else "column-two-warp-pinned-scale",
        warmup=5,
        repeats=200,
        cases=[],
        scope="Synthetic BF16 projected QKV and gates at released B8/T128 backbone geometry; native original-denominator four-warp column baseline. Candidate only pins scaled-score multiply to explicit mul.rn.f32 to retain the four-warp rounded product before maximum/subtract. Other FP32 fusion and all native denominator arithmetic unchanged. Per-call allocations included; two fresh seeds and both four-/two-warp explicit-scale variants checked bitwise before component timing.",
    )
    if registered:
        result[
            "scope"
        ] = "Actual registered CUDA operators under BF16 autocast: selected four-warp baseline and explicit-scale two-warp candidate, fresh seeds20261026/27,200 samples and five warmups per neighboring control bracket; every BF16 output bit and all finite checks, allocations included. Original denominator, RMS and gate retained. Full-model gates and independent confirmation remain required."
    for seed in (20261026, 20261027) if registered else (20261024, 20261025):
        torch.manual_seed(seed)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            raw = torch.randn((8, 24576, 3072), device="cuda", dtype=torch.bfloat16)
            gate = torch.randn((8, 24576, 1024), device="cuda", dtype=torch.bfloat16)
            control = (
                partial(_NATIVE_CORE, raw, gate)
                if registered
                else partial(native_evaluate, raw, gate, 4)
            )
            expected = control()
            for warps in (2,) if registered else (4, 2):
                candidate = (
                    partial(attention, raw, gate)
                    if registered
                    else partial(evaluate, raw, gate, warps)
                )
                actual = candidate()
                delta = actual.float() - expected.float()
                feature = delta.square().mean(
                    (0, 1)
                ).sqrt() / expected.float().square().mean((0, 1)).sqrt().clamp_min(
                    1e-12
                )
                row = dict(
                    seed=seed,
                    warps=warps,
                    different_bits=int(
                        (actual.view(torch.int16) != expected.view(torch.int16)).sum()
                    ),
                    finite=bool(actual.isfinite().all()),
                    relative_rms=float(
                        delta.square().mean().sqrt()
                        / expected.float().square().mean().sqrt()
                    ),
                    max_feature_relative_rms=float(feature.max()),
                    max_absolute_difference=float(delta.abs().max()),
                )
                del actual, delta, feature
                row["baseline_before"] = measure(control, 5, 200)
                row["candidate"] = measure(candidate, 5, 200)
                row["baseline_after"] = measure(control, 5, 200)
                result["cases"].append(row)
                Path(path).write_text(json.dumps(result, indent=2) + "\n")
                print(
                    "PINNED_SCALE",
                    seed,
                    warps,
                    row["different_bits"],
                    row["relative_rms"],
                    row["candidate"]["median_ms"],
                    row["baseline_before"]["median_ms"],
                    row["baseline_after"]["median_ms"],
                    flush=True,
                )
            del raw, gate, expected

    rows = [row for row in result["cases"] if row["warps"] == 2]
    stats = latency_gate(
        *[
            dict(
                before=row["baseline_before"],
                candidate=row["candidate"],
                after=row["baseline_after"],
            )
            for row in rows
        ],
        isolated_scope=f"seed{rows[0]['seed']}",
    )
    stats["comparisons"] = {
        (f"seed{rows[1]['seed']}_" + key[5:] if key.startswith("full_") else key): value
        for key, value in stats["comparisons"].items()
    }
    stats["scope"] = "Two synthetic seed brackets; full-model gates remain required"
    result["registered"] = registered
    result["component_gate"] = stats
    result["eligible_for_model_test"] = stats["significant"] and all(
        row["finite"] and row["different_bits"] == 0 for row in rows
    )
    Path(path).write_text(json.dumps(result, indent=2) + "\n")
    print("PINNED_COMPONENT_GATE", result["eligible_for_model_test"], stats, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--registered", action="store_true")
    args = parser.parse_args()
    run(args.commit, args.results, args.registered)


if __name__ == "__main__":
    main()
