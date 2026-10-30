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
"""Synthetic short-column attention screen; Cirrus-42 (AI agent)."""

import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.nn.functional import scaled_dot_product_attention

from screamcast.column_attention import attention
from screamcast.cuda_timing import measure


def run(args):
    torch.manual_seed(20261020)
    with torch.inference_mode():
        q, k, v = [
            torch.randn((8192, 16, 24, 64), dtype=torch.bfloat16, device="cuda")
            for _ in range(3)
        ]
        baseline = torch.compile(
            lambda: scaled_dot_product_attention(q, k, v, scale=0.125)
        )
        expected = baseline()
        before = measure(baseline, args.warmup, args.repeats)
        trials = []
        for normalized in (False, True):
            actual = attention(q, k, v, normalized)
            delta = actual.float() - expected.float()
            rms = float(
                (delta.square().mean() / expected.float().square().mean()).sqrt()
            )
            max_abs = float(delta.abs().max())
            timing = measure(
                lambda: attention(q, k, v, normalized), args.warmup, args.repeats
            )
            trials.append(
                dict(
                    normalize_before_value_dot=normalized,
                    timing=timing,
                    relative_rms=rms,
                    max_absolute_difference=max_abs,
                    finite=bool(actual.isfinite().all()),
                )
            )
            print(
                "CANDIDATE", normalized, timing["median_ms"], rms, max_abs, flush=True
            )
        after = measure(baseline, args.warmup, args.repeats)
    row = dict(
        agent="Cirrus-42 (AI agent)",
        time=datetime.now(timezone.utc).isoformat(),
        commit=args.commit,
        experiment="short-column-attention-screen",
        seed=20261020,
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        warmup=args.warmup,
        repeats=args.repeats,
        scope="One semantic depth-column attention;8192 columns,16heads,24levels,64channels,BF16; synthetic contiguous QKV; scale0.125; no dropout; excludes projections,norm and gates",
        baseline_before=before,
        trials=trials,
        baseline_after=after,
    )
    Path(args.results).write_text(json.dumps(row, indent=2) + "\n")
    print(
        "RESULT",
        before["median_ms"],
        after["median_ms"],
        flush=True,
    )
