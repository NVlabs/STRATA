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
"""Preserve conditioning reduction with vector memory layouts; Cirrus-42 (AI agent)."""

import json
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, SliceLayout
from triton.experimental.gluon.language.extra import libdevice

from screamcast import conditioning_norm
from screamcast.conditioning_norm import execute
from screamcast.cuda_timing import measure


@gluon.jit
def vector_kernel(A, B, BIAS, OUT, ROWS: gl.constexpr, VEC: gl.constexpr):
    LOAD: gl.constexpr = BlockedLayout([1, VEC], [1, 32], [1, 4], [1, 0])
    COMPUTE: gl.constexpr = BlockedLayout([1, 8], [1, 32], [1, 4], [1, 0])
    row = gl.program_id(0).to(gl.int64) * ROWS + gl.arange(
        0, ROWS, layout=SliceLayout(1, LOAD)
    )
    channel = gl.arange(0, 1024, layout=SliceLayout(0, LOAD))
    av = gl.load(
        A + row[:, None] * 512 + channel[None, :], channel[None, :] < 512, other=0
    ).to(gl.float32)
    bv = gl.load(
        B + row[:, None] * 512 + channel[None, :] - 512,
        channel[None, :] >= 512,
        other=0,
    ).to(gl.float32)
    value = gl.where(channel[None, :] < 512, av, bv) + gl.load(BIAS + channel)[None, :]
    value = gl.convert_layout(value, COMPUTE)
    inv = libdevice.rsqrt(gl.sum(value * value, 1) / 1024.0 + 1.1920928955078125e-7)
    normalized = value * inv[:, None]
    gelu = (normalized * 0.5) * (1.0 + libdevice.erf(normalized * 0.7071067811865476))
    gelu = gl.convert_layout(gelu, LOAD)
    gl.store(OUT + row[:, None] * 1024 + channel[None, :], gelu)


def evaluate(a, b, bias, rows, vec):
    out = torch.empty((8, 393216, 1024), device=a.device, dtype=torch.bfloat16)
    vector_kernel[(3145728 // rows,)](
        a, b, bias, out, ROWS=rows, VEC=vec, num_warps=4, enable_fp_fusion=False
    )
    return out


def benchmark(commit, results):
    torch.manual_seed(20261020)
    result = dict(
        agent="Cirrus-42 (AI agent)",
        commit=commit,
        time=datetime.now(timezone.utc).isoformat(),
        experiment="conditioning-vector-memory-component",
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        seed=20261020,
        warmup=5,
        repeats=50,
        cases=[],
        scope="Synthetic released BF16 NHWC convolution outputs and FP32 bias, same FP32 RMS1024 reduction layout and erf GELU as accepted one-row four-warp kernel; output allocations included",
    )
    Path(results).write_text(json.dumps(result, indent=2) + "\n")
    with torch.inference_mode():
        a, b = [
            torch.randn(
                (192, 128, 128, 512), device="cuda", dtype=torch.bfloat16
            ).permute(0, 3, 1, 2)
            for _ in range(2)
        ]
        bias = torch.randn(1024, device="cuda") * 0.1
        baseline = partial(execute, a, b, bias, 1, 4)
        expected = baseline()
        for rows in (1, 2, 4, 8):
            for vec in (4, 8, 16):
                row = dict(rows=rows, vector=vec)
                try:
                    candidate = partial(evaluate, a, b, bias, rows, vec)
                    actual = candidate()
                    different = 0
                    for aa, bb in zip(actual.split(1), expected.split(1)):
                        different += int(
                            (aa.view(torch.int16) != bb.view(torch.int16)).sum()
                        )
                    row.update(
                        status="complete",
                        different_bits=different,
                        finite=bool(actual.isfinite().all()),
                    )
                    del actual
                    row["baseline_before"] = measure(baseline, 5, 50)
                    row["candidate"] = measure(candidate, 5, 50)
                    row["baseline_after"] = measure(baseline, 5, 50)
                    print(
                        "CONDITIONING_VECTOR",
                        rows,
                        vec,
                        different,
                        {
                            k: row[k]["median_ms"]
                            for k in ("baseline_before", "candidate", "baseline_after")
                        },
                        flush=True,
                    )
                except Exception as error:
                    row.update(status="error", error=repr(error))
                    print(
                        "CONDITIONING_VECTOR_ERROR", rows, vec, repr(error), flush=True
                    )
                    result["cases"].append(row)
                    Path(results).write_text(json.dumps(result, indent=2) + "\n")
                    raise
                result["cases"].append(row)
                Path(results).write_text(json.dumps(result, indent=2) + "\n")


_NATIVE_NORM = conditioning_norm.norm_tail


@torch.library.custom_op("cirrus42::conditioning_vector_norm", mutates_args=())
def norm_tail(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    for tensor in (a, b):
        if (
            tensor.shape != (192, 512, 128, 128)
            or tensor.dtype != torch.bfloat16
            or not tensor.is_cuda
            or not tensor.is_contiguous(memory_format=torch.channels_last)
        ):
            raise ValueError(
                "Conditioning vector norm requires released CUDA BF16 NHWC convolution outputs"
            )
    if (
        bias.shape != (1024,)
        or bias.dtype != torch.float32
        or bias.device != a.device
        or not bias.is_contiguous()
        or b.device != a.device
    ):
        raise ValueError(
            "Conditioning vector norm requires native FP32 bias on the input device"
        )
    return evaluate(a, b, bias, 4, 4)


@norm_tail.register_fake
def norm_tail_fake(a, b, bias):
    return torch.empty((8, 393216, 1024), device=a.device, dtype=torch.bfloat16)


def restore():
    conditioning_norm.norm_tail = _NATIVE_NORM


def install(strata):
    if strata.training or len(strata.pixel_blocks) != 4:
        raise ValueError(
            "Conditioning vector norm requires four released eval pixel blocks"
        )
    for block in strata.pixel_blocks:
        norm = block.adaln_bilinear_dw_norm
        if (
            norm.weight is not None
            or norm.eps is not None
            or norm.normalized_shape != (1024,)
        ):
            raise ValueError(
                "Conditioning vector norm requires native nonaffine RMS1024 epsilon"
            )
    conditioning_norm.install(strata)
    conditioning_norm.norm_tail = norm_tail
