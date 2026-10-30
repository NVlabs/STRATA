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
"""Fuse final residual addition with decoder layout; Cirrus-42 (AI agent)."""

import json
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from types import MethodType

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, SliceLayout

from screamcast import token_layout
from screamcast.cuda_timing import measure
from screamcast.token_layout import restore_tokens


@gluon.jit
def boundary_kernel(
    A, B, C, OUT, ROWS: gl.constexpr, CHANNELS: gl.constexpr, WARPS: gl.constexpr
):
    LOAD: gl.constexpr = BlockedLayout([1, 4], [4, 8], [WARPS // 2, 2], [1, 0])
    STORE: gl.constexpr = BlockedLayout([4, 1], [8, 4], [2, WARPS // 2], [0, 1])
    row = gl.program_id(0) * ROWS + gl.arange(0, ROWS, layout=SliceLayout(1, LOAD))
    channel = gl.program_id(1) * CHANNELS + gl.arange(
        0, CHANNELS, layout=SliceLayout(0, LOAD)
    )
    source = row[:, None] * 1024 + channel[None, :]
    a = gl.load(A + source)
    b = gl.load(B + source).to(gl.float32)
    c = gl.load(C + source).to(gl.float32)
    value = (a + b) + c
    value = gl.convert_layout(value, STORE)
    out_row = gl.program_id(0) * ROWS + gl.arange(0, ROWS, layout=SliceLayout(1, STORE))
    out_channel = gl.program_id(1) * CHANNELS + gl.arange(
        0, CHANNELS, layout=SliceLayout(0, STORE)
    )
    destination = (
        out_row[:, None] // 24576 * 25165824
        + out_channel[None, :] * 24576
        + out_row[:, None] % 24576
    )
    gl.store(OUT + destination, value)


def launch(a, b, c, out, rows, channels, warps):
    boundary_kernel[(196608 // rows, 1024 // channels)](
        a,
        b,
        c,
        out,
        ROWS=rows,
        CHANNELS=channels,
        WARPS=warps,
        num_warps=warps,
        enable_fp_fusion=False,
    )
    return out


def evaluate(a, b, c, rows, channels, warps):
    output = torch.empty(
        (8, 1024, 24576), device=a.device, dtype=torch.float32
    ).transpose(1, 2)
    return launch(a, b, c, output, rows, channels, warps)


def native_add(a, b, c):
    return (a + b) + c


def reference_call(compiled, a, b, c):
    return restore_tokens(compiled(a, b, c))


def benchmark(commit, results):
    torch.manual_seed(20261020)
    result = dict(
        agent="Cirrus-42 (AI agent)",
        time=datetime.now(timezone.utc).isoformat(),
        commit=commit,
        experiment="final-residual-channel-major-store",
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        repeats=50,
        warmup=5,
        cases=[],
        scope="Synthetic released final-block shapes: FP32 residual plus two BF16 projections; preserve (a+b)+c; compiled native addition followed by actual opaque channel-major boundary copy vs fused write; both paths include their normal output allocations",
    )
    Path(results).write_text(json.dumps(result, indent=2) + "\n")
    with torch.inference_mode():
        a = torch.randn((8, 24576, 1024), device="cuda", dtype=torch.float32)
        b = torch.randn(a.shape, device="cuda", dtype=torch.bfloat16)
        c = torch.randn(a.shape, device="cuda", dtype=torch.bfloat16)
        compiled = torch.compile(native_add)
        reference = partial(reference_call, compiled, a, b, c)
        expected = reference()
        for rows in (32, 64):
            for channels in (32, 64):
                for warps in (4, 8):
                    row = dict(rows=rows, channels=channels, warps=warps)
                    candidate = partial(evaluate, a, b, c, rows, channels, warps)
                    try:
                        actual = candidate()
                        row.update(
                            status="complete",
                            fp32_different_bits=int(
                                (
                                    actual.view(torch.int32)
                                    != expected.view(torch.int32)
                                ).sum()
                            ),
                        )
                        row["baseline_before"] = measure(reference, 5, 50)
                        row["candidate"] = measure(candidate, 5, 50)
                        row["baseline_after"] = measure(reference, 5, 50)
                        print(
                            "BOUNDARY",
                            rows,
                            channels,
                            warps,
                            row["fp32_different_bits"],
                            {
                                k: row[k]["median_ms"]
                                for k in (
                                    "baseline_before",
                                    "candidate",
                                    "baseline_after",
                                )
                            },
                            flush=True,
                        )
                    except Exception as error:
                        row.update(status="error", error=repr(error))
                        print(
                            "BOUNDARY_ERROR",
                            rows,
                            channels,
                            warps,
                            str(error)[:500],
                            flush=True,
                        )
                    result["cases"].append(row)
                    Path(results).write_text(json.dumps(result, indent=2) + "\n")


_ORIGINALS = []


@torch.library.custom_op("cirrus42::residual_channel_major", mutates_args=())
def residual_channel_major(
    input: torch.Tensor, attention: torch.Tensor, mlp: torch.Tensor
) -> torch.Tensor:
    if (
        input.shape != (8, 24576, 1024)
        or input.dtype != torch.float32
        or not input.is_cuda
        or not input.is_contiguous()
    ):
        raise ValueError(
            "Residual layout fusion requires contiguous released FP32 backbone tokens"
        )
    for projection in (attention, mlp):
        if (
            projection.shape != input.shape
            or projection.dtype != torch.bfloat16
            or projection.device != input.device
            or not projection.is_contiguous()
        ):
            raise ValueError(
                "Residual layout fusion requires contiguous BF16 attention and MLP projections"
            )
    return evaluate(input, attention, mlp, 32, 64, 4)


@residual_channel_major.register_fake
def residual_channel_major_fake(input, attention, mlp):
    return torch.empty(
        (8, 1024, 24576), device=input.device, dtype=input.dtype
    ).transpose(1, 2)


def _last_forward(self, input, latent_dhw=None, rope_tables=None):
    attention = self.attn(
        self.norm1(input), latent_dhw=latent_dhw, rope_tables=rope_tables
    )
    middle = input + attention
    mlp = self.mlp(self.norm2(middle))
    return residual_channel_major(input, attention, mlp)


def restore():
    for block, had_forward, original in _ORIGINALS:
        if had_forward:
            block.forward = original
        else:
            del block.forward
        token_layout._HANDLES.append(
            block.register_forward_hook(token_layout._after_last_block)
        )
    _ORIGINALS.clear()


def install(strata):
    restore()
    if (
        strata.training
        or len(strata.backbone.blocks) != 24
        or len(token_layout._HANDLES) != 2
    ):
        raise ValueError(
            "Boundary fusion requires released eval backbone and installed token-layout hooks"
        )
    block = strata.backbone.blocks[-1]
    _ORIGINALS.append((block, "forward" in block.__dict__, block.forward))
    token_layout._HANDLES.pop().remove()
    block.forward = MethodType(_last_forward, block)
