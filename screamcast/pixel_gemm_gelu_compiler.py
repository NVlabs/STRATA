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
"""Trace exact GELU lookup into GEMM templates; Cirrus-42 (AI agent)."""

from datetime import datetime, timezone
from functools import partial

import torch
from torch.nn import functional as F

from screamcast import gelu_table
from screamcast.benchmark_statistics import latency_gate
from screamcast.cuda_timing import measure
from screamcast.gemm_schedule_validation import difference, linear, save
from screamcast.pixel_gemm_gelu import reference

MODES = {
    "trace_default": {},
    "trace_tma": {
        "max_autotune_gemm": True,
        "max_autotune_gemm_backends": "ATEN,TRITON",
        "triton.enable_persistent_tma_matmul": True,
        "benchmark_epilogue_fusion": True,
        "max_epilogue_benchmarked_choices": 5,
        "epilogue_fusion_first": True,
    },
    "trace_tma_only": {
        "max_autotune_gemm": True,
        "max_autotune_gemm_backends": "TRITON",
        "triton.enable_persistent_tma_matmul": True,
        "benchmark_epilogue_fusion": True,
        "max_epilogue_benchmarked_choices": 5,
        "epilogue_fusion_first": True,
    },
}


def traced_lookup(projected, table):
    bits = projected.view(torch.int16).to(torch.int32) & 65535
    return table[bits]


def projection_lookup(x, weight, bias, table):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        projected = F.linear(x, weight, bias)
    return traced_lookup(projected, table)


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
        experiment="pixel-gemm-traceable-gelu-epilogue",
        checkpoint=args.checkpoint,
        weight_key=key,
        input_shape=[3145728, 128],
        output_shape=[3145728, 512],
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        warmup=5,
        repeats=100,
        seeds=[20261042, 20261043],
        modes=MODES,
        cases=[],
        scope="Actual first pixel MLP weights with synthetic BF16 normalized inputs. Confirmed native projection plus opaque exact GELU lookup versus traceable BF16-rounded lookup with native or persistent-TMA compiler epilogues. Weight conversions and allocations included. Component evidence only.",
    )
    save(args.results, result)
    with torch.inference_mode():
        table = gelu_table.prepare_table(torch.device("cuda:0"))
        patterns = (
            torch.arange(65536, device="cuda", dtype=torch.int32)
            .to(torch.int16)
            .view(torch.bfloat16)
        )
        lookup_actual = torch.compile(traced_lookup)(patterns, table)
        result["all_pattern_different_bf16_bits"] = int(
            (lookup_actual.view(torch.int16) != table.view(torch.int16)).sum()
        )
        if result["all_pattern_different_bf16_bits"]:
            raise RuntimeError("Traceable lookup changed native GELU table values")
        save(args.results, result)
        del patterns, lookup_actual
        for mode, options in MODES.items():
            case = dict(mode=mode, samples=[])
            result["cases"].append(case)
            print("TRACE_GELU_START", mode, flush=True)
            try:
                compiled = torch.compile(linear)
                fused = torch.compile(projection_lookup, options=options)
                for seed in result["seeds"]:
                    torch.manual_seed(seed)
                    x = torch.randn((3145728, 128), device="cuda", dtype=torch.bfloat16)
                    baseline = partial(reference, compiled, x, weight, bias, table)
                    candidate = partial(fused, x, weight, bias, table)
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
                        "TRACE_GELU_RESULT",
                        mode,
                        seed,
                        sample["error"],
                        {
                            k: sample[k]["median_ms"]
                            for k in ("before", "candidate", "after")
                        },
                        flush=True,
                    )
                    del baseline, candidate, x
                gate = latency_gate(*case["samples"], isolated_scope="seed20261042")
                gate["comparisons"] = {
                    key.replace("full_", "seed20261043_"): value
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
                print("TRACE_GELU_ERROR", mode, repr(error), flush=True)
            save(args.results, result)
            torch._dynamo.reset()
    result["status"] = "complete"
    save(args.results, result)
