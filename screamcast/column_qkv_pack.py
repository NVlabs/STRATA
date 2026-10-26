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
"""Fuse column QK RMS statistics and QKV layout; Cirrus-42 (AI agent)."""

import json
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from screamcast.cuda_timing import measure


@triton.jit
def pack_kernel(INPUT, Q, K, V, ROWS: tl.constexpr):
    index = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    channel = tl.arange(0, 64)
    depth = index % 24
    head = index // 24 % 16
    column = index // 384 % 1024
    batch = index // 393216
    token = batch * 24576 + depth * 1024 + column
    source = token[:, None] * 3072 + head[:, None] * 64 + channel[None, :]
    q = tl.load(INPUT + source).to(tl.float32)
    k = tl.load(INPUT + source + 1024).to(tl.float32)
    v = tl.load(INPUT + source + 2048)
    qsum = tl.sum(q * q, 1)
    ksum = tl.sum(k * k, 1)
    qscale = libdevice.rsqrt(qsum / 64.0 + 1e-6)
    kscale = libdevice.rsqrt(ksum / 64.0 + 1e-6)
    destination = index[:, None] * 64 + channel[None, :]
    tl.store(Q + destination, q * qscale[:, None])
    tl.store(K + destination, k * kscale[:, None])
    tl.store(V + destination, v)


def reference(input):
    qkv = input.reshape(8, 24576, 3, 16, 64).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        q = torch.nn.functional.rms_norm(q, (64,), None, 1e-6).to(v.dtype)
        k = torch.nn.functional.rms_norm(k, (64,), None, 1e-6).to(v.dtype)
    return tuple(
        t.reshape(8, 16, 24, 1024, 64)
        .permute(0, 3, 1, 2, 4)
        .contiguous()
        .reshape(8192, 16, 24, 64)
        for t in (q, k, v)
    )


def evaluate(input, rows, warps):
    outputs = [
        torch.empty((8192, 16, 24, 64), device=input.device, dtype=torch.bfloat16)
        for _ in range(3)
    ]
    pack_kernel[(3145728 // rows,)](
        input, *outputs, ROWS=rows, num_warps=warps, num_stages=1, enable_fp_fusion=True
    )
    return tuple(outputs)


def benchmark(commit, results):
    torch.manual_seed(20261020)
    result = dict(
        agent="Cirrus-42 (AI agent)",
        time=datetime.now(timezone.utc).isoformat(),
        commit=commit,
        experiment="column-qk-rms-qkv-pack-component",
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        repeats=50,
        warmup=5,
        cases=[],
        scope="Synthetic BF16 projected QKV8x24576x3072; native compiled nonaffine RMSNorm epsilon1e-6 under BF16 AMP, then native depth-column layout; output allocations included; no checkpoint or full-model claim",
    )
    Path(results).write_text(json.dumps(result, indent=2) + "\n")
    with torch.inference_mode():
        input = torch.randn((8, 24576, 3072), device="cuda", dtype=torch.bfloat16)
        compiled = torch.compile(reference)
        control = partial(compiled, input)
        expected = control()
        for rows in (16, 32, 64, 128):
            for warps in (4, 8):
                row = dict(rows=rows, warps=warps)
                candidate = partial(evaluate, input, rows, warps)
                try:
                    actual = candidate()
                    row.update(
                        status="complete",
                        different_bits=[
                            int((a.view(torch.int16) != b.view(torch.int16)).sum())
                            for a, b in zip(actual, expected)
                        ],
                    )
                    del actual
                    row["baseline_before"] = measure(control, 5, 50)
                    row["candidate"] = measure(candidate, 5, 50)
                    row["baseline_after"] = measure(control, 5, 50)
                    print(
                        "QKV_PACK",
                        rows,
                        warps,
                        row["different_bits"],
                        {
                            k: row[k]["median_ms"]
                            for k in ("baseline_before", "candidate", "baseline_after")
                        },
                        flush=True,
                    )
                except Exception as error:
                    row.update(status="error", error=repr(error))
                    print("QKV_ERROR", rows, warps, str(error)[:500], flush=True)
                result["cases"].append(row)
                Path(results).write_text(json.dumps(result, indent=2) + "\n")
