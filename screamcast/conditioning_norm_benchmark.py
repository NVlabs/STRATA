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
"""Screen persistent conditioning norm against compiled native math."""

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from screamcast.conditioning_norm import execute
from screamcast.cuda_timing import measure


def native(a, b, bias):
    s = torch.cat(
        (
            a.float() + bias[:512][None, :, None, None],
            b.float() + bias[512:][None, :, None, None],
        ),
        dim=1,
    )
    return (
        F.gelu(F.rms_norm(s.permute(0, 2, 3, 1), (1024,)))
        .reshape(8, 24 * 128 * 128, 1024)
        .to(torch.bfloat16)
    )


def run(args):
    torch.manual_seed(20261020)
    with torch.inference_mode():
        a, b = [
            torch.randn(
                (192, 128, 128, 512), dtype=torch.bfloat16, device="cuda"
            ).permute(0, 3, 1, 2)
            for _ in range(2)
        ]
        bias = torch.randn(1024, device="cuda") * 0.1
        reference = torch.compile(native)
        expected = reference(a, b, bias)
        record = dict(
            agent="Cirrus-42 (AI agent)",
            commit=args.commit,
            experiment="conditioning-norm-screen",
            gpu=torch.cuda.get_device_name(),
            torch=torch.__version__,
            seed=20261020,
            scope="synthetic BF16 contiguous-NHWC 192x128x128x1024 native convolution outputs, FP32 bias/RMSNorm/GELU, BF16 projection input; all channels",
            trials=[],
        )
        record["baseline_before"] = measure(
            lambda: reference(a, b, bias), args.warmup, args.repeats
        )
        for rows in (1, 2, 4, 8):
            for warps in (4, 8):
                print("START", rows, warps, flush=True)
                actual = execute(a, b, bias, rows, warps)
                ds, rs = [], []
                for aa, bb in zip(actual.split(1), expected.split(1)):
                    ds.append((aa.float() - bb.float()).square().sum())
                    rs.append(bb.float().square().sum())
                error = (torch.stack(ds).sum() / torch.stack(rs).sum()).sqrt().item()
                finite = bool(actual.isfinite().all())
                del actual
                timing = measure(
                    lambda: execute(a, b, bias, rows, warps), args.warmup, args.repeats
                )
                row = dict(
                    rows=rows,
                    warps=warps,
                    relative_rms=error,
                    finite=finite,
                    timing=timing,
                )
                record["trials"].append(row)
                print("RESULT", rows, warps, error, timing["median_ms"], flush=True)
                Path(args.results).write_text(json.dumps(record, indent=2) + "\n")
        record["baseline_after"] = measure(
            lambda: reference(a, b, bias), args.warmup, args.repeats
        )
        Path(args.results).write_text(json.dumps(record, indent=2) + "\n")
