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
"""Pixel projection with rounded GELU lookup; Cirrus-42 (AI agent)."""

import itertools
from datetime import datetime, timezone
from functools import partial

import torch
import triton
import triton.language as tl

from screamcast import gelu_table
from screamcast.benchmark_statistics import latency_gate
from screamcast.cuda_timing import measure
from screamcast.gemm_schedule_validation import difference, linear, save


@triton.jit
def fused_kernel(X, W, BIAS, TABLE, OUT, BM: tl.constexpr, BN: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    columns = tl.program_id(1) * BN + tl.arange(0, BN)
    channels = tl.arange(0, 128)
    x = tl.load(X + rows[:, None] * 128 + channels[None, :])
    w = tl.load(W + columns[None, :] * 128 + channels[:, None])
    accum = tl.dot(x, w, allow_tf32=False)
    bias = tl.load(BIAS + columns).to(tl.float32)
    rounded = (accum + bias[None, :]).to(tl.bfloat16)
    bits = rounded.to(tl.uint16, bitcast=True).to(tl.int32)
    result = tl.load(TABLE + bits)
    tl.store(OUT + rows[:, None] * 512 + columns[None, :], result)


def evaluate(x, weight, bias, table, bm, bn, warps):
    packed_weight = weight.to(torch.bfloat16)
    packed_bias = bias.to(torch.bfloat16)
    output = torch.empty((3145728, 512), device=x.device, dtype=torch.bfloat16)
    fused_kernel[(3145728 // bm, 512 // bn)](
        x,
        packed_weight,
        packed_bias,
        table,
        output,
        BM=bm,
        BN=bn,
        num_warps=warps,
        num_stages=2,
        enable_fp_fusion=False,
    )
    return output


def reference(compiled, x, weight, bias, table):
    projected = compiled(x, weight, bias).view(8, 393216, 512)
    return gelu_table.lookup(projected, table).view(3145728, 512)


def run(args):
    state = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False, mmap=True
    )["network"]
    key = "_pixel_blocks.0.mlp.fwd.0.weight"
    with torch.inference_mode(False):
        weight = torch.nn.Parameter(state[key].cuda(), requires_grad=True)
        bias = torch.nn.Parameter(
            state[key.removesuffix("weight") + "bias"].cuda(), requires_grad=True
        )
    result = dict(
        agent="Cirrus-42 (AI agent)",
        source_commit=args.commit,
        time=datetime.now(timezone.utc).isoformat(),
        experiment="pixel-gemm-rounded-gelu-fusion",
        checkpoint=args.checkpoint,
        weight_key=key,
        input_shape=[3145728, 128],
        output_shape=[3145728, 512],
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        warmup=5,
        repeats=100,
        seeds=[20261040, 20261041],
        cases=[],
        scope="Released first pixel MLP weights, synthetic BF16 normalized inputs at actual B8/T128 geometry. Native BF16 projection followed by confirmed exact GELU lookup versus one projection+BF16-round+lookup kernel. Per-call FP32 weight/bias conversion and output allocation included; no full-model acceptance.",
    )
    save(args.results, result)
    with torch.inference_mode():
        table = gelu_table.prepare_table(torch.device("cuda:0"))
        compiled = torch.compile(linear)
        for bm, bn, warps in itertools.product((64, 128, 256), (64, 128), (4, 8)):
            case = dict(bm=bm, bn=bn, warps=warps, samples=[])
            result["cases"].append(case)
            print("GEMM_GELU_START", bm, bn, warps, flush=True)
            try:
                for seed in result["seeds"]:
                    torch.manual_seed(seed)
                    x = torch.randn((3145728, 128), device="cuda", dtype=torch.bfloat16)
                    baseline = partial(reference, compiled, x, weight, bias, table)
                    candidate = partial(evaluate, x, weight, bias, table, bm, bn, warps)
                    expected = baseline()
                    actual = candidate()
                    sample = dict(seed=seed, error=difference(actual, expected))
                    del actual, expected
                    sample["before"] = measure(baseline, 5, 100)
                    sample["candidate"] = measure(candidate, 5, 100)
                    sample["after"] = measure(baseline, 5, 100)
                    case["samples"].append(sample)
                    save(args.results, result)
                    print(
                        "GEMM_GELU_RESULT",
                        bm,
                        bn,
                        warps,
                        seed,
                        sample["error"],
                        {
                            k: sample[k]["median_ms"]
                            for k in ("before", "candidate", "after")
                        },
                        flush=True,
                    )
                    del baseline, candidate, x
                gate = latency_gate(*case["samples"], isolated_scope="seed20261040")
                gate["comparisons"] = {
                    key.replace("full_", "seed20261041_"): value
                    for key, value in gate["comparisons"].items()
                }
                case["gate"] = gate
                case["component_eligible"] = gate["significant"] and all(
                    sample["error"]["finite"]
                    and sample["error"]["relative_rms"] <= 0.005
                    and sample["error"]["max_channel_relative_rms"] <= 0.01
                    for sample in case["samples"]
                )
                case["status"] = "complete"
            except Exception as error:
                case.update(status="error", error=repr(error))
                print("GEMM_GELU_ERROR", bm, bn, warps, repr(error), flush=True)
            save(args.results, result)
    result["status"] = "complete"
    save(args.results, result)
