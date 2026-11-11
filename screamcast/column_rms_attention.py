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
"""Fuse column QK RMS normalization, attention and gate; Cirrus-42 (AI agent)."""

import json
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from types import MethodType

import torch
import triton
import triton.language as tl
from physicsnemo.experimental.models.strata import layers
from triton.language.extra.cuda import libdevice

from screamcast.column_gate import attention_gate
from screamcast.column_precision import (
    fragment_sum,
    native_attention,
    selected_attention,
)
from screamcast.column_qkv_pack import reference as qkv_reference
from screamcast.cuda_timing import measure


@triton.jit
def rms_attention_kernel(INPUT, OUT, GATE, MODE: tl.constexpr):
    group = tl.program_id(0)
    column, head = group // 16, group % 16
    level = tl.arange(0, 32)
    channel = tl.arange(0, 64)
    token = column // 1024 * 24576 + level * 1024 + column % 1024
    source = token[:, None] * 3072 + head * 64 + channel[None, :]
    raw_q = tl.load(INPUT + source, level[:, None] < 24, other=0).to(tl.float32)
    raw_k = tl.load(INPUT + source + 1024, level[:, None] < 24, other=0).to(tl.float32)
    value = tl.load(INPUT + source + 2048, level[:, None] < 24, other=0)
    qsum = tl.sum(raw_q * raw_q, 1)
    ksum = tl.sum(raw_k * raw_k, 1)
    qscale = libdevice.rsqrt(qsum / 64.0 + 1e-6)
    kscale = libdevice.rsqrt(ksum / 64.0 + 1e-6)
    q = (raw_q * qscale[:, None]).to(tl.bfloat16)
    k = (raw_k * kscale[:, None]).to(tl.bfloat16).trans(1, 0)
    scores = tl.dot(q, k) * 0.125
    scores = tl.where(level[None, :] < 24, scores, -float("inf"))
    if MODE == 2:
        maximum = tl.max(scores, 1)
        probability = tl.exp2(
            tl.fma(scores, 1.4426950408889634, -maximum[:, None] * 1.4426950408889634)
        )
    elif MODE == 3:
        probability = libdevice.exp(scores - tl.max(scores, 1)[:, None])
    else:
        scaled = scores * 1.4426950408889634
        maximum_scaled = tl.max(scaled, 1)
        if MODE == 14:
            maximum_scaled = tl.maximum(maximum_scaled, 0.0)
        probability = tl.exp2(scaled - maximum_scaled[:, None])
    if MODE == 7:
        denominator = tl.sum(probability.to(tl.float64), 1).to(tl.float32)
    elif MODE >= 8 and MODE <= 11:
        denominator = fragment_sum(probability, MODE)
    else:
        denominator = tl.sum(probability, 1)
    output = tl.dot(probability.to(tl.bfloat16), value)
    if MODE == 4:
        reciprocal = tl.div_rn(1.0, denominator)
        output = output * reciprocal[:, None]
    elif MODE == 5:
        reciprocal = tl.inline_asm_elementwise(
            "rcp.approx.ftz.f32 $0, $1;",
            constraints="=f,f",
            args=[denominator],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        output = output * reciprocal[:, None]
    elif MODE == 6:
        output = tl.div_rn(output, denominator[:, None])
    elif MODE == 12:
        output = output * tl.exp2(-tl.log2(denominator))[:, None]
    elif MODE == 13:
        log_sum = maximum_scaled + tl.log2(denominator)
        output = output * tl.exp2(maximum_scaled - log_sum)[:, None]
    else:
        output = output / denominator[:, None]
    index = (
        (column // 1024 * 24576 + level[:, None] * 1024 + column % 1024) * 1024
        + head * 64
        + channel[None, :]
    )
    logits = tl.load(GATE + index, level[:, None] < 24, other=0).to(tl.float32)
    rounded = output.to(tl.bfloat16).to(tl.float32)
    tl.store(OUT + index, rounded * tl.sigmoid(logits), level[:, None] < 24)


def baseline(compiled, input, gate):
    q, k, v = compiled(input)
    return attention_gate(q, k, v, gate)


def evaluate(input, gate, warps):
    output = torch.empty_like(gate)
    rms_attention_kernel[(131072,)](
        input, output, gate, MODE=9, num_warps=warps, enable_fp_fusion=True
    )
    return output


def benchmark(commit, results):
    torch.manual_seed(20261020)
    result = dict(
        agent="Cirrus-42 (AI agent)",
        time=datetime.now(timezone.utc).isoformat(),
        commit=commit,
        experiment="column-rms-attention-gate-component",
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        warmup=5,
        repeats=50,
        cases=[],
        scope="Synthetic projected BF16 QKV8x24576x3072 and BF16 logits8x24576x1024; native compiled nonaffine RMS64 epsilon1e-6 underBF16AMP, column gate fusion source62412558 after native QKV packing; native-order attention and FP32 gate boundary; output allocations included",
    )
    Path(results).write_text(json.dumps(result, indent=2) + "\n")
    with torch.inference_mode():
        input = torch.randn((8, 24576, 3072), device="cuda", dtype=torch.bfloat16)
        gate = torch.randn((8, 24576, 1024), device="cuda", dtype=torch.bfloat16)
        compiled = torch.compile(qkv_reference)
        q, k, v = compiled(input)
        native = native_attention(q, k, v)
        accepted = selected_attention(q, k, v)
        result["native_attention_replay_different_bits"] = int(
            (native.view(torch.int16) != accepted.view(torch.int16)).sum()
        )
        del q, k, v, native, accepted
        control = partial(baseline, compiled, input, gate)
        expected = control()
        for warps in (2, 4, 8):
            row = dict(warps=warps)
            candidate = partial(evaluate, input, gate, warps)
            try:
                actual = candidate()
                row.update(
                    status="complete",
                    different_bits=int(
                        (actual.view(torch.int16) != expected.view(torch.int16)).sum()
                    ),
                    finite=bool(actual.isfinite().all()),
                    max_absolute_difference=float(
                        (actual.float() - expected.float()).abs().max()
                    ),
                )
                del actual
                row["baseline_before"] = measure(control, 5, 50)
                row["candidate"] = measure(candidate, 5, 50)
                row["baseline_after"] = measure(control, 5, 50)
                print(
                    "RMS_ATTENTION",
                    warps,
                    row["different_bits"],
                    row["max_absolute_difference"],
                    {
                        k: row[k]["median_ms"]
                        for k in ("baseline_before", "candidate", "baseline_after")
                    },
                    flush=True,
                )
            except Exception as error:
                row.update(status="error", error=repr(error))
                print("RMS_ATTENTION_ERROR", warps, str(error)[:500], flush=True)
            result["cases"].append(row)
            Path(results).write_text(json.dumps(result, indent=2) + "\n")


_ORIGINALS = []


def validate_operands(raw: torch.Tensor, logits: torch.Tensor) -> None:
    if (
        raw.shape != (8, 24576, 3072)
        or raw.dtype != torch.bfloat16
        or not raw.is_cuda
        or not raw.is_contiguous()
    ):
        raise ValueError(
            "Normalized column attention requires released contiguous CUDA BF16 QKV projection"
        )
    if (
        logits.shape != (8, 24576, 1024)
        or logits.dtype != torch.bfloat16
        or logits.device != raw.device
        or not logits.is_contiguous()
    ):
        raise ValueError(
            "Normalized column attention requires fixed contiguous BF16 gate logits"
        )


@torch.library.custom_op("cirrus42::normalized_column_attention_gate", mutates_args=())
def normalized_attention_gate(raw: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    validate_operands(raw, logits)
    return evaluate(raw, logits, 4)


@normalized_attention_gate.register_fake
def normalized_attention_gate_fake(raw, logits):
    return torch.empty_like(logits)


def _forward(self, input, latent_dhw=None, rope_tables=None):
    if latent_dhw != (24, 32, 32) or rope_tables is not None:
        raise ValueError(
            "Normalized column attention requires released depth geometry without RoPE"
        )
    if (
        not torch.is_autocast_enabled("cuda")
        or torch.get_autocast_dtype("cuda") != torch.bfloat16
    ):
        raise ValueError("Normalized column attention requires native BF16 autocast")
    logits = self.gate_proj(input)
    raw = self.qkv(input)
    output = normalized_attention_gate(raw, logits)
    return self.proj_drop(self.proj(output))


def restore():
    for module, had_forward, original in _ORIGINALS:
        if had_forward:
            module.forward = original
        else:
            del module.forward
    _ORIGINALS.clear()


def install(strata):
    restore()
    if strata.training:
        raise ValueError("Normalized column attention requires eval inference")
    selected = []
    for module in strata.backbone.modules():
        if (
            isinstance(module, layers.Natten3DSelfAttention)
            and module.do_depthwise_attention
        ):
            if (
                module.training
                or module.num_heads != 16
                or module.head_dim != 64
                or module.scale != 0.125
                or not module.gated_attention
            ):
                raise ValueError(
                    "Normalized column attention requires released gated16heads64geometry"
                )
            for norm in (module.q_norm, module.k_norm):
                if (
                    not isinstance(norm, torch.nn.RMSNorm)
                    or norm.normalized_shape != (64,)
                    or norm.eps != 1e-6
                    or norm.weight is not None
                ):
                    raise ValueError(
                        "Normalized column attention requires nonaffine RMS64 epsilon1e-6"
                    )
            selected.append(module)
    if len(selected) != 12:
        raise ValueError(
            "Normalized column attention requires twelve released depth-attention modules"
        )
    for module in selected:
        _ORIGINALS.append((module, "forward" in module.__dict__, module.forward))
        module.forward = MethodType(_forward, module)
