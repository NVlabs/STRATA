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
"""Schedule fused depthwise/RMSNorm/GELU loops; Cirrus-42 (AI agent)."""

import json
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as F
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, SliceLayout
from triton.experimental.gluon.language.extra import libdevice

from screamcast import conditioning_vector
from screamcast.cuda_timing import measure


@gluon.jit
def convolution_step(pointer, W, channel, acc, kh, kw):
    value = gl.load(pointer + (kh * 132 + kw) * 512).to(gl.float32)
    weight = gl.load(W + (kh * 5 + kw) * 1024 + channel).to(gl.float32)
    return gl.fma(value, weight[None, :], acc)


@gluon.jit
def fused_kernel(A, B, W, BIAS, OUT, ROWS: gl.constexpr, MODE: gl.constexpr):
    LAYOUT: gl.constexpr = BlockedLayout([1, 8], [1, 32], [1, 4], [1, 0])
    row = gl.program_id(0).to(gl.int64) * ROWS + gl.arange(
        0, ROWS, layout=SliceLayout(1, LAYOUT)
    )
    channel = gl.arange(0, 1024, layout=SliceLayout(0, LAYOUT))
    plane = row // 16384
    h = row // 128 % 128
    w = row % 128
    base = (plane[:, None] * 17424 + h[:, None] * 132 + w[:, None]) * 512 + channel[
        None, :
    ] % 512
    pointer = gl.where(channel[None, :] < 512, A + base, B + base)
    acc = gl.full((ROWS, 1024), 0, gl.float32, layout=LAYOUT)
    if MODE == 0:
        for kh in range(5):
            for kw in range(5):
                acc = convolution_step(pointer, W, channel, acc, kh, kw)
    elif MODE == 1:
        for kh in range(5):
            for kw in gl.static_range(5):
                acc = convolution_step(pointer, W, channel, acc, kh, kw)
    else:
        for kh in gl.static_range(5):
            for kw in gl.static_range(5):
                acc = convolution_step(pointer, W, channel, acc, kh, kw)
    value = acc.to(gl.bfloat16).to(gl.float32) + gl.load(BIAS + channel)[None, :]
    inv = libdevice.rsqrt(gl.sum(value * value, 1) / 1024.0 + 1.1920928955078125e-7)
    normalized = value * inv[:, None]
    result = (normalized * 0.5) * (1.0 + libdevice.erf(normalized * 0.7071067811865476))
    gl.store(OUT + row[:, None] * 1024 + channel[None, :], result)


def native(a, b, weight, bias):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        aa = F.conv2d(a, weight[:512], groups=512)
        bb = F.conv2d(b, weight[512:], groups=512)
    return conditioning_vector.evaluate(aa, bb, bias, 4, 4)


def evaluate(a, b, weight, bias, rows, mode):
    packed = weight.reshape(1024, 25).transpose(0, 1).contiguous().to(torch.bfloat16)
    out = torch.empty((8, 393216, 1024), device=a.device, dtype=torch.bfloat16)
    fused_kernel[(3145728 // rows,)](
        a,
        b,
        packed,
        bias,
        out,
        ROWS=rows,
        MODE=mode,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out


def benchmark(commit, results):
    result = dict(
        agent="Cirrus-42 (AI agent)",
        commit=commit,
        time=datetime.now(timezone.utc).isoformat(),
        experiment="conditioning-conv-rms-gelu-loop-component",
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        warmup=5,
        repeats=30,
        cases=[],
        scope="Synthetic paired BF16 NHWC192x132x132x512 padded input, one FP32 depthwise5x5 filter and FP32 bias. Native two groups512 cuDNN convolutions plus accepted four-row/vector4 normalization versus fused row-major FP32 FMA, explicit BF16 convolution round, FP32 bias/RMS1024 epsilon1.1920928955078125e-7/erf GELU, BF16 projection input. Native reduction compute layoutvector8/four warps retained. Candidate BF16 tap-major weight packing and normal output allocation included. Int64 output offsets; Nested kh/kw loops with full rolling, five-column unrolling or full25-term unrolling; pointer-select memory. Two seeds.",
    )
    Path(results).write_text(json.dumps(result, indent=2) + "\n")
    with torch.inference_mode():
        for seed in (20261020, 20261021):
            torch.manual_seed(seed)
            a, b = [
                torch.randn(
                    (192, 132, 132, 512), device="cuda", dtype=torch.bfloat16
                ).permute(0, 3, 1, 2)
                for _ in range(2)
            ]
            weight = torch.randn((1024, 1, 5, 5), device="cuda") * 0.1
            bias = torch.randn(1024, device="cuda") * 0.1
            baseline = partial(native, a, b, weight, bias)
            expected = baseline()
            for rows in (2, 4, 8, 16):
                for mode in (0, 1, 2):
                    row = dict(seed=seed, rows=rows, loop_mode=mode, warps=4)
                    try:
                        candidate = partial(evaluate, a, b, weight, bias, rows, mode)
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
                        row["baseline_before"] = measure(baseline, 5, 30)
                        row["candidate"] = measure(candidate, 5, 30)
                        row["baseline_after"] = measure(baseline, 5, 30)
                        print(
                            "FUSED_CONV_NORM_LOOP",
                            seed,
                            rows,
                            mode,
                            different,
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
                        result["cases"].append(row)
                        Path(results).write_text(json.dumps(result, indent=2) + "\n")
                        raise
                    result["cases"].append(row)
                    Path(results).write_text(json.dumps(result, indent=2) + "\n")
            del baseline, expected, a, b, weight, bias, candidate


_ORIGINALS = []


@torch.library.custom_op("cirrus42::conditioning_conv_norm_fused", mutates_args=())
def fused_norm(
    a: torch.Tensor, b: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    for tensor in (a, b):
        if (
            tensor.shape != (192, 512, 132, 132)
            or tensor.dtype != torch.bfloat16
            or not tensor.is_cuda
            or not tensor.is_contiguous(memory_format=torch.channels_last)
        ):
            raise ValueError(
                "Fused conditioning requires released CUDA BF16 padded NHWC inputs"
            )
    if (
        weight.shape != (1024, 1, 5, 5)
        or weight.dtype != torch.float32
        or not weight.is_contiguous()
        or bias.shape != (1024,)
        or bias.dtype != torch.float32
        or not bias.is_contiguous()
        or any(t.device != a.device for t in (b, weight, bias))
    ):
        raise ValueError(
            "Fused conditioning requires original FP32 contiguous5x5 weights and bias"
        )
    return evaluate(a, b, weight, bias, 8, 1)


@fused_norm.register_fake
def fused_norm_fake(a, b, weight, bias):
    return torch.empty((8, 393216, 1024), device=a.device, dtype=torch.bfloat16)


def compute(self, padded):
    conv = self.adaln_bilinear_dw_conv
    z = fused_norm(padded[0], padded[1], conv.weight, conv.bias)
    return self.adaln_bilinear_dw_proj(z).reshape(8, 393216, 6, 128).unbind(-2)


def restore():
    for module, had_compute, original in _ORIGINALS:
        if had_compute:
            module._compute_bilinear_dw_adaln_params = original
        else:
            del module._compute_bilinear_dw_adaln_params
    _ORIGINALS.clear()


def install(strata):
    restore()
    if strata.training or len(strata.pixel_blocks) != 4:
        raise ValueError("Fused conditioning requires four released eval pixel blocks")
    for module in strata.pixel_blocks:
        conv = module.adaln_bilinear_dw_conv
        norm = module.adaln_bilinear_dw_norm
        if (
            conv.weight.shape != (1024, 1, 5, 5)
            or conv.weight.dtype != torch.float32
            or conv.bias is None
            or conv.bias.shape != (1024,)
            or conv.bias.dtype != torch.float32
            or norm.normalized_shape != (1024,)
            or norm.weight is not None
            or norm.eps is not None
        ):
            raise ValueError(
                "Fused conditioning requires released FP32 filter/bias and native nonaffine RMS1024"
            )
    for module in strata.pixel_blocks:
        _ORIGINALS.append(
            (
                module,
                "_compute_bilinear_dw_adaln_params" in module.__dict__,
                module._compute_bilinear_dw_adaln_params,
            )
        )
        module._compute_bilinear_dw_adaln_params = MethodType(compute, module)
