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
"""Compose native-order normalization loop schedules; Cirrus-42 (AI agent)."""

import argparse
import json
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, SliceLayout
from triton.experimental.gluon.language.extra import libdevice

from screamcast import normalization_model as selected
from screamcast.cuda_timing import measure
from screamcast.normalization_order import _combine, _update

_NATIVE_KERNEL = selected.native_order_kernel


@gluon.jit
def unrolled_kernel(
    A,
    B,
    OUT,
    N: gl.constexpr,
    SEMANTIC: gl.constexpr,
    RESIDUAL: gl.constexpr,
    COALESCED: gl.constexpr,
):
    if SEMANTIC:
        X: gl.constexpr = 64
        R: gl.constexpr = 64
        COMPUTE: gl.constexpr = BlockedLayout([4, 1], [16, 2], [1, 16], [0, 1])
        LOAD: gl.constexpr = BlockedLayout([1, 2], [1, 32], [16, 1], [1, 0])
    else:
        X: gl.constexpr = 8
        R: gl.constexpr = 256
        LOAD: gl.constexpr = BlockedLayout([1, 4], [1, 32], [8, 2], [1, 0])
        if COALESCED:
            COMPUTE: gl.constexpr = LOAD
        else:
            COMPUTE: gl.constexpr = BlockedLayout([4, 1], [2, 16], [1, 16], [0, 1])
    row = gl.program_id(0) * X + gl.arange(0, X, layout=SliceLayout(1, LOAD))
    channel = gl.arange(0, R, layout=SliceLayout(0, LOAD))
    offset = row[:, None] * 1024 + channel[None, :]
    loaded = gl.load(A + offset, row[:, None] < N, other=0)
    if RESIDUAL:
        loaded += gl.load(B + offset, row[:, None] < N, other=0).to(gl.float32)
    mean = gl.convert_layout(loaded, COMPUTE)
    m2 = gl.zeros_like(mean)
    weight = gl.full_like(mean, 1.0)
    if SEMANTIC:
        for start in range(R, 1024, R):
            loaded = gl.load(A + offset + start, row[:, None] < N, other=0)
            if RESIDUAL:
                loaded += gl.load(B + offset + start, row[:, None] < N, other=0).to(
                    gl.float32
                )
            current = gl.convert_layout(loaded, COMPUTE)
            mean, m2, weight = _update(current, mean, m2, weight, False)
    else:
        for start in gl.static_range(R, 1024, R):
            loaded = gl.load(A + offset + start, row[:, None] < N, other=0)
            if RESIDUAL:
                loaded += gl.load(B + offset + start, row[:, None] < N, other=0).to(
                    gl.float32
                )
            current = gl.convert_layout(loaded, COMPUTE)
            mean, m2, weight = _update(current, mean, m2, weight, False)
    average, moment, count = gl.reduce((mean, m2, weight), 1, _combine)
    scale = libdevice.rsqrt(moment / 1024.0 + 1e-6)
    for start in gl.static_range(0, 1024, R):
        loaded = gl.load(A + offset + start, row[:, None] < N, other=0)
        if RESIDUAL:
            loaded += gl.load(B + offset + start, row[:, None] < N, other=0).to(
                gl.float32
            )
        current = gl.convert_layout(loaded, COMPUTE)
        result = (current - average[:, None]) * scale[:, None]
        result = gl.convert_layout(result, LOAD)
        gl.store(OUT + offset + start, result, row[:, None] < N)


def restore():
    selected.native_order_kernel = _NATIVE_KERNEL


def install(strata):
    restore()
    if (
        strata.training
        or len(selected._ORIGINALS) != 48
        or len(strata.backbone.blocks) != 24
        or not strata.backbone.bf16_mixed
    ):
        raise ValueError("Loop schedule requires48 validated eval native-order norms")
    modules = [
        norm for block in strata.backbone.blocks for norm in (block.norm1, block.norm2)
    ]
    registered = [norm for norm, _, _ in selected._ORIGINALS]
    if any(
        norm.training
        or norm not in registered
        or norm.forward.__func__ is not selected._forward
        for norm in modules
    ):
        raise ValueError(
            "Loop schedule requires this model's native-order norm methods"
        )
    selected.native_order_kernel = unrolled_kernel


def execute(kernel, a, b, semantic, residual, coalesced, dtype):
    output = torch.empty(a.shape, device=a.device, dtype=dtype)
    kernel[(triton.cdiv(196608, 64 if semantic else 8),)](
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


def run(commit, path):
    result = dict(
        agent="Cirrus-42 (AI agent)",
        time=datetime.now(timezone.utc).isoformat(),
        commit=commit,
        experiment="native-order-selective-loop-current-preflight",
        warmup=5,
        repeats=100,
        cases=[],
        scope="Selective loop kernel from9583616d/b7b5093b: preserve native Welford update/reduction, semantic statistics rolled, column statistics unrolled, output passes unrolled. Current selected native_order_kernel control, four actual semantic/column/residual/presummed modes on fresh seeds20261032/33. Compare every FP32/BF16 output bit and finite, per-call BF16 output allocations included. Compose only after this current-stack preflight, original400-sample backbone/full gates and independent confirmation unchanged.",
    )
    for seed in (20261032, 20261033):
        torch.manual_seed(seed)
        with torch.inference_mode():
            a = torch.randn((8, 24576, 1024), device="cuda", dtype=torch.float32)
            b = torch.randn(a.shape, device="cuda", dtype=torch.bfloat16) * 0.1
            summed = a + b.float()
            for name, semantic, residual, coalesced, input in (
                ("semantic_norm1", True, False, False, a),
                ("column_norm1", False, False, False, a),
                ("residual_norm2", False, True, True, a),
                ("presummed_norm2", False, False, True, summed),
            ):
                args = (input, b, semantic, residual, coalesced)
                expected32 = execute(_NATIVE_KERNEL, *args, torch.float32)
                actual32 = execute(unrolled_kernel, *args, torch.float32)
                expected16 = execute(_NATIVE_KERNEL, *args, torch.bfloat16)
                actual16 = execute(unrolled_kernel, *args, torch.bfloat16)
                row = dict(
                    seed=seed,
                    mode=name,
                    fp32_different_bits=int(
                        (
                            actual32.view(torch.int32) != expected32.view(torch.int32)
                        ).sum()
                    ),
                    bf16_different_bits=int(
                        (
                            actual16.view(torch.int16) != expected16.view(torch.int16)
                        ).sum()
                    ),
                    finite=bool(
                        actual32.isfinite().all() and actual16.isfinite().all()
                    ),
                )
                del expected32, actual32, expected16, actual16
                baseline = partial(execute, _NATIVE_KERNEL, *args, torch.bfloat16)
                candidate = partial(execute, unrolled_kernel, *args, torch.bfloat16)
                row["baseline_before"] = measure(baseline, 5, 100)
                row["candidate"] = measure(candidate, 5, 100)
                row["baseline_after"] = measure(baseline, 5, 100)
                result["cases"].append(row)
                Path(path).write_text(json.dumps(result, indent=2) + "\n")
                print(
                    "NATIVE_LOOP_CURRENT",
                    seed,
                    name,
                    row["fp32_different_bits"],
                    row["bf16_different_bits"],
                    *[
                        row[k]["median_ms"]
                        for k in ("baseline_before", "candidate", "baseline_after")
                    ],
                    flush=True
                )
            del a, b, summed
    result["all_bits_exact"] = all(
        x["finite"] and x["fp32_different_bits"] == x["bf16_different_bits"] == 0
        for x in result["cases"]
    )
    Path(path).write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--results", required=True)
    args = parser.parse_args()
    run(args.commit, args.results)


if __name__ == "__main__":
    main()
