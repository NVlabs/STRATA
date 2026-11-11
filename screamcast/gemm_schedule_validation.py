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
"""Screen released linear shapes and weights; Cirrus-42 (AI agent)."""

import json
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import torch
from torch.nn import functional as F

from screamcast.benchmark_statistics import latency_gate
from screamcast.cuda_timing import measure

CASES = (
    ("semantic_mlp_first", 196608, 4096, 1024),
    ("semantic_mlp_second", 196608, 1024, 4096),
    ("pixel_mlp_first", 3145728, 512, 128),
    ("conditioning_projection", 3145728, 768, 1024),
)
MODES = {
    "triton_tma": {
        "max_autotune_gemm": True,
        "max_autotune_gemm_backends": "ATEN,TRITON",
        "triton.enable_persistent_tma_matmul": True,
    },
    "cutedsl": {
        "max_autotune_gemm": True,
        "max_autotune_gemm_backends": "ATEN,CUTEDSL",
    },
}


def linear(x, weight, bias):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return F.linear(x, weight, bias)


def save(path, result):
    destination = Path(path)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(result) + "\n")
    temporary.replace(destination)


def difference(actual, expected):
    delta = actual.float() - expected.float()
    expected_float = expected.float()
    channel = delta.square().mean(0).sqrt() / expected_float.square().mean(
        0
    ).sqrt().clamp_min(1e-12)
    return dict(
        finite=bool(actual.isfinite().all()),
        different_bf16_bits=int(
            (
                actual.contiguous().view(torch.int16)
                != expected.contiguous().view(torch.int16)
            ).sum()
        ),
        relative_rms=float(
            delta.square().mean().sqrt()
            / expected_float.square().mean().sqrt().clamp_min(1e-12)
        ),
        max_channel_relative_rms=float(channel.max()),
        max_abs=float(delta.abs().max()),
    )


def run(args):
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    state = checkpoint["network"]
    result = dict(
        agent="Cirrus-42 (AI agent)",
        source_commit=args.commit,
        time=datetime.now(timezone.utc).isoformat(),
        experiment="released-linear-compiler-schedules",
        checkpoint=args.checkpoint,
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        warmup=5,
        repeats=100,
        seeds=[20261038, 20261039],
        modes=MODES,
        scope="Released FP32 weights/bias with synthetic BF16 normalized inputs at actual flattened B8/T128 shapes; native internal BF16 AMP; output allocation and per-call weight casts included. No full-model acceptance.",
        cases=[],
    )
    save(args.results, result)
    for name, rows, outputs, features in CASES:
        key = next(
            k
            for k, value in state.items()
            if k.endswith(".weight") and tuple(value.shape) == (outputs, features)
        )
        bias_key = key.removesuffix("weight") + "bias"
        with torch.inference_mode(False):
            weight = torch.nn.Parameter(state[key].cuda(), requires_grad=True)
            bias = torch.nn.Parameter(state[bias_key].cuda(), requires_grad=True)
        for mode, options in MODES.items():
            case = dict(
                name=name,
                mode=mode,
                weight_key=key,
                bias_key=bias_key,
                shape=[rows, outputs, features],
                samples=[],
            )
            result["cases"].append(case)
            print("GEMM_START", name, mode, key, flush=True)
            try:
                baseline = torch.compile(linear)
                candidate = torch.compile(linear, options=options)
                with torch.inference_mode():
                    for seed in result["seeds"]:
                        torch.manual_seed(seed)
                        x = torch.randn(
                            (rows, features), device="cuda", dtype=torch.bfloat16
                        )
                        reference = partial(baseline, x, weight, bias)
                        changed = partial(candidate, x, weight, bias)
                        expected = reference()
                        start = time.perf_counter()
                        actual = changed()
                        torch.cuda.synchronize()
                        compile_seconds = time.perf_counter() - start
                        sample = dict(
                            seed=seed,
                            compile_seconds=compile_seconds,
                            error=difference(actual, expected),
                        )
                        del actual, expected
                        sample["before"] = measure(reference, 5, 100)
                        sample["candidate"] = measure(changed, 5, 100)
                        sample["after"] = measure(reference, 5, 100)
                        case["samples"].append(sample)
                        save(args.results, result)
                        print(
                            "GEMM_RESULT",
                            name,
                            mode,
                            seed,
                            sample["error"],
                            {
                                k: sample[k]["median_ms"]
                                for k in ("before", "candidate", "after")
                            },
                            flush=True,
                        )
                        del reference, changed, x
                gate = latency_gate(*case["samples"], isolated_scope="seed20261038")
                gate["comparisons"] = {
                    key.replace("full_", "seed20261039_"): value
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
                print("GEMM_ERROR", name, mode, repr(error), flush=True)
            save(args.results, result)
            torch._dynamo.reset()
        del weight, bias
    result["status"] = "complete"
    save(args.results, result)
