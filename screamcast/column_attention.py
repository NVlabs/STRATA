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
"""Fixed 24-level column attention; Cirrus-42 (AI agent)."""

import torch
import triton
import triton.language as tl
from physicsnemo.experimental.models.strata import layers

_ORIGINAL_SDPA = layers.F.scaled_dot_product_attention


@triton.jit
def column_kernel(
    Q,
    K,
    V,
    OUT,
    NORMALIZED: tl.constexpr,
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
    probability = tl.exp(scores - tl.max(scores, 1)[:, None])
    denominator = tl.sum(probability, 1)
    if NORMALIZED:
        probability = probability / denominator[:, None]
    output = tl.dot(probability.to(tl.bfloat16), value)
    if not NORMALIZED:
        output = output / denominator[:, None]
    tl.store(
        OUT + group * 1536 + level[:, None] * 64 + channel[None, :],
        output,
        level[:, None] < 24,
    )


@torch.library.custom_op("cirrus42::column_attention", mutates_args=())
def attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, normalized: bool = False
) -> torch.Tensor:
    for operand in (q, k, v):
        if (
            operand.shape != (8192, 16, 24, 64)
            or operand.dtype != torch.bfloat16
            or operand.stride(-1) != 1
        ):
            raise ValueError(
                "Column attention requires BF16 8192 columns,16 heads,24 levels,64 channels"
            )
    output = torch.empty((8192, 16, 24, 64), device=q.device, dtype=q.dtype)
    column_kernel[(131072,)](
        q,
        k,
        v,
        output,
        NORMALIZED=normalized,
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


@attention.register_fake
def attention_fake(q, k, v, normalized=False):
    return torch.empty(q.shape, device=q.device, dtype=q.dtype)


def sdpa(
    q,
    k,
    v,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    *,
    scale=None,
    enable_gqa=False
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
        return attention(q, k, v)
    return _ORIGINAL_SDPA(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )


def restore():
    layers.F.scaled_dot_product_attention = _ORIGINAL_SDPA


def install(strata):
    if strata.training:
        raise ValueError("Short-column attention requires eval inference")
    count = 0
    for module in strata.backbone.modules():
        if (
            isinstance(module, layers.Natten3DSelfAttention)
            and module.do_depthwise_attention
        ):
            if module.num_heads != 16 or module.head_dim != 64 or module.scale != 0.125:
                raise ValueError(
                    "Short-column attention requires sixteen heads of64,scale0.125"
                )
            count += 1
    if count != 12:
        raise ValueError(
            "Short-column attention requires twelve depth-attention blocks"
        )
    layers.F.scaled_dot_product_attention = sdpa
