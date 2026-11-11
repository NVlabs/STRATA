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
"""Screen rounded semantic output gate fusion; Cirrus-42 (AI agent)."""

import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from natten.token_permute.torch_impl import token_unpermute_torch

from screamcast import semantic_gate_fusion, semantic_gate_precision
from screamcast.cuda_timing import measure


def reference(attention, logits):
    gate = semantic_gate_precision.rounded_sigmoid(logits)
    output = token_unpermute_torch(
        attention, (24, 32, 32), (8, 4, 8), (1, 1, 1), True
    ).reshape(8, 24576, 1024)
    return output * gate


def run(args):
    torch.manual_seed(20261020)
    with torch.inference_mode():
        attention = torch.randn((8, 24576, 16, 64), device="cuda", dtype=torch.bfloat16)
        logits = torch.randn((8, 24576, 1024), device="cuda", dtype=torch.bfloat16)
        baseline = torch.compile(reference)
        expected = baseline(attention, logits)
        before = measure(lambda: baseline(attention, logits), args.warmup, args.repeats)
        actual = semantic_gate_fusion.output_gate(attention, logits)
        count = int((actual.view(torch.int16) != expected.view(torch.int16)).sum())
        max_absolute = float((actual.float() - expected.float()).abs().max())
        candidate = measure(
            lambda: semantic_gate_fusion.output_gate(attention, logits),
            args.warmup,
            args.repeats,
        )
        after = measure(lambda: baseline(attention, logits), args.warmup, args.repeats)
    row = dict(
        agent="Cirrus-42 (AI agent)",
        time=datetime.now(timezone.utc).isoformat(),
        experiment="semantic-output-rounded-gate-screen",
        commit=args.commit,
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        scope="One semantic output unpermute and sigmoid gate; B8,D24,H=W32,16heads64; fixed synthetic BF16 inputs",
        seed=20261020,
        warmup=args.warmup,
        repeats=args.repeats,
        baseline_before=before,
        candidate=candidate,
        baseline_after=after,
        different_output_bit_patterns=count,
        max_absolute_difference=max_absolute,
    )
    Path(args.results).write_text(json.dumps(row, indent=2) + "\n")
    print(
        "RESULT",
        before["median_ms"],
        candidate["median_ms"],
        after["median_ms"],
        count,
        max_absolute,
        flush=True,
    )
