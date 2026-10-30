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
"""Validate fused conditioning on released weights and real input; Cirrus-42 (AI agent)."""

import hashlib
import json
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import torch

from screamcast import conditioning_spatial_reuse
from screamcast.conditioning_conv_norm_unroll import evaluate, native
from screamcast.cuda_timing import measure


def run(args):
    with open(args.inputs, "rb") as stream:
        input_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    if input_sha != "221cdae124bd6d9785a1c3a456d332e585eb61b90f0d117b7377b0717779dca7":
        raise ValueError("Fixed benchmark input checksum changed")
    cached = torch.load(args.inputs, map_location="cpu", weights_only=True, mmap=True)
    if (cached["batch_size"], cached["tile_size"]) != (8, 128) or tuple(
        cached["latent_dhw"]
    ) != (24, 32, 32):
        raise ValueError("Released B8/T128 geometry required")
    if cached["checkpoint"] != args.checkpoint:
        raise ValueError("Checkpoint differs from fixed input provenance")
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True, mmap=True
    )
    network = checkpoint["network"]
    result = dict(
        agent="Cirrus-42 (AI agent)",
        time=datetime.now(timezone.utc).isoformat(),
        commit=args.commit,
        experiment="released-conditioning-conv-rms-gelu-component",
        inputs=args.inputs,
        input_sha256=input_sha,
        checkpoint=args.checkpoint,
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        warmup=args.warmup,
        repeats=args.repeats,
        rows=8,
        loop_mode=1,
        warps=4,
        cases=[],
        scope="Fixed real conditioning from released benchmark input; confirmed native coarse preparation; four original FP32 checkpoint filters/biases, memory-mapped network state exactly as pipeline.load_checkpoint consumes it. Native BF16 cuDNN convolution plus accepted vector normalization versus selected fused schedule. Per-call BF16 weight packing and output allocations included; require every BF16 projection-input bit equal for all four filters.",
    )
    Path(args.results).write_text(json.dumps(result, indent=2) + "\n")
    with torch.inference_mode():
        conditioning = cached["conditioning"].cuda()
        conditioning_spatial_reuse.initialize()
        a, b = conditioning_spatial_reuse.fused_prepare(conditioning)
        for index in range(4):
            weight_key = f"_pixel_blocks.{index}.adaln_bilinear_dw_conv.weight"
            bias_key = f"_pixel_blocks.{index}.adaln_bilinear_dw_conv.bias"
            raw_weight, raw_bias = network[weight_key], network[bias_key]
            row = dict(
                block=index,
                weight_key=weight_key,
                bias_key=bias_key,
                weight_sha256=hashlib.sha256(raw_weight.numpy().tobytes()).hexdigest(),
                bias_sha256=hashlib.sha256(raw_bias.numpy().tobytes()).hexdigest(),
            )
            weight, bias = raw_weight.cuda(), raw_bias.cuda()
            baseline = partial(native, a, b, weight, bias)
            candidate = partial(evaluate, a, b, weight, bias, 8, 1)
            expected = baseline()
            actual = candidate()
            different = 0
            for aa, bb in zip(actual.split(1), expected.split(1)):
                different += int((aa.view(torch.int16) != bb.view(torch.int16)).sum())
            row.update(different_bits=different, finite=bool(actual.isfinite().all()))
            del actual, expected
            row["baseline_before"] = measure(baseline, args.warmup, args.repeats)
            row["candidate"] = measure(candidate, args.warmup, args.repeats)
            row["baseline_after"] = measure(baseline, args.warmup, args.repeats)
            result["cases"].append(row)
            Path(args.results).write_text(json.dumps(result, indent=2) + "\n")
            print(
                "REAL_CONV_NORM",
                index,
                different,
                {
                    k: row[k]["median_ms"]
                    for k in ("baseline_before", "candidate", "baseline_after")
                },
                flush=True,
            )
            if different or not row["finite"]:
                raise RuntimeError(
                    "Released conditioning BF16 projection inputs differ"
                )
