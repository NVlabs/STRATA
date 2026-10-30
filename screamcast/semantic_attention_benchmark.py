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
"""Synthetic semantic attention layout screen; Cirrus-42 (AI agent)."""

import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from natten.functional import na3d

from screamcast import semantic_attention_layout
from screamcast.cuda_timing import measure


def attention(q, k, v, optimized):
    operands = [
        t.reshape(8, 16, 24, 32, 32, 64).permute(0, 2, 3, 4, 1, 5) for t in (q, k, v)
    ]
    function = semantic_attention_layout.na3d if optimized else na3d
    return function(*operands, kernel_size=(9, 9, 9), backend="blackwell-fna").reshape(
        8, 24576, 1024
    )


def run(args):
    torch.manual_seed(20261020)
    with torch.inference_mode():
        operands = [
            torch.randn((8, 16, 24576, 64), dtype=torch.bfloat16, device="cuda")
            for _ in range(3)
        ]
        torch._dynamo.reset()
        baseline = torch.compile(lambda: attention(*operands, False))
        expected = baseline()
        before = measure(baseline, args.warmup, args.repeats)
        torch._dynamo.reset()
        candidate = torch.compile(lambda: attention(*operands, True))
        actual = candidate()
        difference = actual.float() - expected.float()
        error = (
            (difference.square().mean() / expected.float().square().mean())
            .sqrt()
            .item()
        )
        finite = bool(actual.isfinite().all())
        max_abs = difference.abs().max().item()
        del actual, difference
        timing = measure(candidate, args.warmup, args.repeats)
        torch._dynamo.reset()
        baseline = torch.compile(lambda: attention(*operands, False))
        after = measure(baseline, args.warmup, args.repeats)
    record = dict(
        agent="Cirrus-42 (AI agent)",
        time=datetime.now(timezone.utc).isoformat(),
        commit=args.commit,
        experiment="semantic-attention-layout-screen",
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        seed=20261020,
        scope="One semantic neighborhood block; B8,D24,H=W32,16heads64,BF16; synthetic contiguous heads-first QKV; includes input/output layout; excludes QKV/output projections,norm,RoPE; each phase independently compiled",
        warmup=args.warmup,
        repeats=args.repeats,
        baseline_before=before,
        candidate=timing,
        baseline_after=after,
        finite=finite,
        relative_rms=error,
        max_absolute_difference=max_abs,
    )
    Path(args.results).write_text(json.dumps(record, indent=2) + "\n")
    print(
        "RESULT",
        before["median_ms"],
        timing["median_ms"],
        after["median_ms"],
        error,
        max_abs,
        flush=True,
    )
