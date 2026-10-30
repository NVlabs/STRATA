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
"""Real-operand column precision diagnostics; Cirrus-42 (AI agent)."""

from pathlib import Path

import torch
import triton
import triton.language as tl
from physicsnemo.experimental.models.strata import layers
from triton.language.extra.cuda import libdevice

from screamcast import column_attention

records = []
operand_directory: Path | None = None


@triton.jit
def fragment_sum(probability, mode: tl.constexpr):
    lane = tl.arange(0, 4)[None, :]
    indices = tl.broadcast_to(2 * lane, (32, 4))
    p0 = tl.gather(probability, indices, 1)
    p1 = tl.gather(probability, indices + 1, 1)
    p8 = tl.gather(probability, indices + 8, 1)
    p9 = tl.gather(probability, indices + 9, 1)
    p16 = tl.gather(probability, indices + 16, 1)
    p17 = tl.gather(probability, indices + 17, 1)
    if mode == 10:
        partial = ((p0 + p1) + (p8 + p9)) + (p16 + p17)
    elif mode == 11:
        partial = ((((p0 + p8) + p16) + p1) + p9) + p17
    else:
        partial = ((((p0 + p1) + p8) + p9) + p16) + p17
    s0 = tl.gather(partial, tl.full((32, 1), 0, tl.int32), 1).reshape((32,))
    s1 = tl.gather(partial, tl.full((32, 1), 1, tl.int32), 1).reshape((32,))
    s2 = tl.gather(partial, tl.full((32, 1), 2, tl.int32), 1).reshape((32,))
    s3 = tl.gather(partial, tl.full((32, 1), 3, tl.int32), 1).reshape((32,))
    if mode == 9:
        return (s0 + s1) + (s2 + s3)
    return (s0 + s2) + (s1 + s3)


@triton.jit
def precision_kernel(
    Q,
    K,
    V,
    OUT,
    MODE: tl.constexpr,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    Q2: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    K2: tl.constexpr,
    V0: tl.constexpr,
    V1: tl.constexpr,
    V2: tl.constexpr,
):
    group = tl.program_id(0)
    column, head = group // 16, group % 16
    level = tl.arange(0, 32)
    channel = tl.arange(0, 64)
    q = tl.load(
        Q + column * Q0 + head * Q1 + level[:, None] * Q2 + channel[None, :],
        level[:, None] < 24,
        other=0,
    )
    k = tl.load(
        K + column * K0 + head * K1 + level[None, :] * K2 + channel[:, None],
        level[None, :] < 24,
        other=0,
    )
    value = tl.load(
        V + column * V0 + head * V1 + level[:, None] * V2 + channel[None, :],
        level[:, None] < 24,
        other=0,
    )
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
    tl.store(
        OUT + group * 1536 + level[:, None] * 64 + channel[None, :],
        output,
        level[:, None] < 24,
    )


def native_attention(q, k, v):
    return torch.ops.aten._scaled_dot_product_cudnn_attention.default(
        q, k, v, None, False, scale=0.125
    )[0]


def error_metrics(actual, expected):
    """Summarize BF16 outputs in the captured four-dimensional column layout."""
    delta = actual.float() - expected.float()
    feature = delta.square().mean((0, 2)).sqrt() / expected.float().square().mean(
        (0, 2)
    ).sqrt().clamp_min(1e-12)
    return dict(
        finite=bool(actual.isfinite().all()),
        relative_rms=float(
            (delta.square().mean() / expected.float().square().mean()).sqrt()
        ),
        max_feature_relative_rms=float(feature.max()),
        max_absolute_difference=float(delta.abs().max()),
        different_output_bits=int(
            (actual.view(torch.int16) != expected.view(torch.int16)).sum()
        ),
    )


def variant(q, k, v, mode):
    if mode == 0:
        return column_attention.attention(q, k, v)
    output = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    precision_kernel[(131072,)](
        q,
        k,
        v,
        output,
        MODE=mode,
        Q0=q.stride(0),
        Q1=q.stride(1),
        Q2=q.stride(2),
        K0=k.stride(0),
        K1=k.stride(1),
        K2=k.stride(2),
        V0=v.stride(0),
        V1=v.stride(1),
        V2=v.stride(2),
        num_warps=4,
    )
    return output


@torch.library.custom_op("cirrus42::column_precision_diagnostic", mutates_args=())
def diagnostic(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    expected = native_attention(q, k, v)
    if len(records) < 12:
        row = dict(
            column_block=len(records),
            input_strides=[list(x.stride()) for x in (q, k, v)],
            output_stride=list(expected.stride()),
            trials=[],
        )
        if operand_directory is not None:
            saved = {
                "agent": "Cirrus-42 (AI agent)",
                "column_block": len(records),
                "repeats": 8,
                "full_shape": [8192, 16, 24, 64],
            }
            for name, operand in zip(("q", "k", "v", "expected"), (q, k, v, expected)):
                if operand.stride() != (24576, 1536, 64, 1):
                    raise ValueError(
                        "Operand cache requires observed contiguous column strides"
                    )
                batches = operand.view(torch.int16).reshape(8, 1024, 16, 24, 64)
                if not torch.equal(batches, batches[:1].expand_as(batches)):
                    raise ValueError(
                        "Operand cache requires eight bitwise-identical batches"
                    )
                saved[name] = operand[:1024].cpu()
            destination = operand_directory / f"column-{len(records):02d}.pt"
            temporary = destination.with_suffix(".pt.tmp")
            torch.save(saved, temporary)
            temporary.replace(destination)
            row["operand_path"] = str(destination)
            row["saved_shape"] = [1024, 16, 24, 64]
            row["verified_batch_repeats"] = 8
        for mode, label in [
            (0, "exp_shift"),
            (1, "exp2_scaled_scores_div_full"),
            (4, "exp2_reciprocal_rn"),
            (5, "exp2_rcp_approx"),
            (6, "exp2_div_rn"),
        ]:
            actual = variant(q, k, v, mode)
            item = dict(mode=mode, label=label, **error_metrics(actual, expected))
            row["trials"].append(item)
            print("COLUMN", row["column_block"], item, flush=True)
            del actual
        records.append(row)
    return expected


@diagnostic.register_fake
def diagnostic_fake(q, k, v):
    return torch.empty_like(q)


def sdpa(
    q,
    k,
    v,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    *,
    scale=None,
    enable_gqa=False,
):
    if (
        q.shape == (8192, 16, 24, 64)
        and q.dtype == torch.bfloat16
        and attn_mask is None
        and dropout_p == 0.0
        and not is_causal
        and scale == 0.125
        and not enable_gqa
    ):
        return diagnostic(q, k, v)
    return column_attention.sdpa(
        q, k, v, attn_mask, dropout_p, is_causal, scale=scale, enable_gqa=enable_gqa
    )


def install(strata):
    records.clear()
    column_attention.install(strata)
    layers.F.scaled_dot_product_attention = sdpa


@torch.library.custom_op("cirrus42::column_attention_native_sum", mutates_args=())
def selected_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    if not q.is_cuda:
        raise ValueError("Native-sum column attention requires CUDA operands")
    for operand in (q, k, v):
        if (
            operand.shape != (8192, 16, 24, 64)
            or operand.dtype != torch.bfloat16
            or operand.stride(-1) != 1
            or operand.device != q.device
        ):
            raise ValueError(
                "Native-sum column attention requires BF16 fixed geometry and contiguous channels"
            )
    return variant(q, k, v, 9)


@selected_attention.register_fake
def selected_attention_fake(q, k, v):
    return torch.empty(q.shape, device=q.device, dtype=q.dtype)


def selected_sdpa(
    q,
    k,
    v,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    *,
    scale=None,
    enable_gqa=False,
):
    if (
        q.is_cuda
        and q.shape == (8192, 16, 24, 64)
        and q.dtype == torch.bfloat16
        and attn_mask is None
        and dropout_p == 0.0
        and not is_causal
        and scale == 0.125
        and not enable_gqa
    ):
        return selected_attention(q, k, v)
    return column_attention._ORIGINAL_SDPA(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )


def install_selected(strata):
    column_attention.install(strata)
    layers.F.scaled_dot_product_attention = selected_sdpa
