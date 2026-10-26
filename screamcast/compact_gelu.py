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
"""Compact exact native BF16 GELU table; Cirrus-42 (AI agent)."""

from datetime import datetime, timezone
from functools import partial

import torch
import triton
import triton.language as tl

from screamcast import gelu_table
from screamcast.benchmark_statistics import latency_gate
from screamcast.cuda_timing import measure
from screamcast.gemm_schedule_validation import save


@triton.jit
def compact_value(value, TABLE):
    bits = value.to(tl.uint16, bitcast=True).to(tl.int32)
    magnitude = bits & 32767
    sign = bits >> 15
    middle = (magnitude >= 14976) & (magnitude < 16640)
    small = magnitude < 256
    exceptional = magnitude >= 32640
    use_table = middle | small | exceptional
    index = tl.where(
        middle,
        magnitude - 14976 + sign * 1664,
        tl.where(
            small, 3328 + magnitude + sign * 256, 3840 + magnitude - 32640 + sign * 128
        ),
    )
    looked_up = tl.load(TABLE + index, use_table, other=0)
    fallback_bits = tl.where(
        magnitude < 14976, bits - 128, tl.where(sign != 0, 32768, bits)
    ).to(tl.uint16)
    fallback = fallback_bits.to(tl.bfloat16, bitcast=True)
    return tl.where(use_table, looked_up, fallback)


@triton.jit
def kernel(INPUT, TABLE, OUT, N: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(INPUT + index, index < N, other=0)
    result = compact_value(value, TABLE)
    tl.store(OUT + index, result, index < N)


def prepare_table(full):
    return torch.cat(
        (
            full[14976:16640],
            full[47744:49408],
            full[:256],
            full[32768:33024],
            full[32640:32768],
            full[65408:65536],
        )
    )


def evaluate(input, table, block=4096, warps=4):
    output = torch.empty_like(input)
    kernel[(triton.cdiv(input.numel(), block),)](
        input,
        table,
        output,
        N=input.numel(),
        BLOCK=block,
        num_warps=warps,
    )
    return output


def run(args):
    result = dict(
        agent="Cirrus-42 (AI agent)",
        source_commit=args.commit,
        time=datetime.now(timezone.utc).isoformat(),
        experiment="compact-native-bf16-gelu",
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        full_table_entries=65536,
        compact_table_entries=4096,
        compact_rule="Store both signs for magnitudes2^-10<=abs(x)<8, exponent0/1, and exponent255; other tiny normals use exact exponent decrement; large finite positives return input and negatives return negative zero. Verify all65536 native patterns before timing.",
        warmup=5,
        repeats=100,
        seeds=[20261044, 20261045],
        cases=[],
        scope="Both released GELU activation shapes with synthetic BF16 input; selected full-table4096-element/four-warp kernel versus compact exact table. No model acceptance from components.",
    )
    save(args.results, result)
    with torch.inference_mode():
        full = gelu_table.prepare_table(torch.device("cuda:0"))
        compact = prepare_table(full)
        patterns = (
            torch.arange(65536, device="cuda", dtype=torch.int32)
            .to(torch.int16)
            .view(torch.bfloat16)
        )
        actual = evaluate(patterns, compact)
        different = actual.view(torch.int16) != full.view(torch.int16)
        result["all_pattern_different_bf16_bits"] = int(different.sum())
        result["first_different_patterns"] = (
            different.nonzero().flatten()[:32].cpu().tolist()
        )
        save(args.results, result)
        if result["all_pattern_different_bf16_bits"]:
            raise RuntimeError(
                f"Compact GELU differs on {result['all_pattern_different_bf16_bits']} patterns: {result['first_different_patterns']}"
            )
        del patterns, actual, different
        for shape in ((8, 24576, 4096), (8, 393216, 512)):
            for block, warps in ((4096, 4), (8192, 4), (8192, 8), (16384, 8)):
                case = dict(shape=list(shape), block=block, warps=warps, samples=[])
                result["cases"].append(case)
                try:
                    for seed in result["seeds"]:
                        torch.manual_seed(seed)
                        x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
                        baseline = partial(gelu_table.lookup, x, full)
                        candidate = partial(evaluate, x, compact, block, warps)
                        expected, actual = baseline(), candidate()
                        sample = dict(
                            seed=seed,
                            different_bf16_bits=int(
                                (
                                    actual.view(torch.int16)
                                    != expected.view(torch.int16)
                                ).sum()
                            ),
                        )
                        del actual, expected
                        sample["before"] = measure(baseline, 5, 100)
                        sample["candidate"] = measure(candidate, 5, 100)
                        sample["after"] = measure(baseline, 5, 100)
                        case["samples"].append(sample)
                        save(args.results, result)
                        print(
                            "COMPACT_GELU_RESULT",
                            list(shape),
                            block,
                            warps,
                            seed,
                            sample["different_bf16_bits"],
                            {
                                k: sample[k]["median_ms"]
                                for k in ("before", "candidate", "after")
                            },
                            flush=True,
                        )
                        del baseline, candidate, x
                    gate = latency_gate(*case["samples"], isolated_scope="seed20261044")
                    gate["comparisons"] = {
                        key.replace("full_", "seed20261045_"): value
                        for key, value in gate["comparisons"].items()
                    }
                    case["gate"] = gate
                    case["component_eligible"] = gate["significant"] and all(
                        sample["different_bf16_bits"] == 0 for sample in case["samples"]
                    )
                    case["status"] = "complete"
                except Exception as error:
                    case.update(status="error", error=repr(error))
                    print("COMPACT_GELU_ERROR", repr(error), flush=True)
                save(args.results, result)
    result["status"] = "complete"
    save(args.results, result)
