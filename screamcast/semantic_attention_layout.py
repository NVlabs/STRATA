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
"""Fixed semantic neighborhood-attention layout; Cirrus-42 (AI agent)."""

import torch
from natten._libnatten import blackwell_na3d_forward
from natten.token_permute.torch_impl import token_permute_torch, token_unpermute_torch
from physicsnemo.experimental.models.strata import layers

from screamcast import attention_layout

_LAYOUT = (24, 32, 32)
_Q_TILE = (8, 4, 8)
_KV_TILE = (4, 4, 8)
_PERSISTENT = False


@torch.library.custom_op("cirrus42::semantic_attention_core", mutates_args=())
def attention_core(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, persistent: bool = False
) -> torch.Tensor:
    output, _ = blackwell_na3d_forward(
        q,
        k,
        v,
        (9, 9, 9),
        (1, 1, 1),
        (1, 1, 1),
        (False, False, False),
        0.125,
        _LAYOUT,
        _LAYOUT,
        _LAYOUT,
        _Q_TILE,
        _KV_TILE,
        persistent,
    )
    return output


@attention_core.register_fake
def attention_core_fake(q, k, v, persistent=False):
    return torch.empty(q.shape, dtype=q.dtype, device=q.device)


def na3d(q, k, v, kernel_size, dilation=1, is_causal=False, backend=None):
    if (
        q.shape != (8, 24, 32, 32, 16, 64)
        or kernel_size != (9, 9, 9)
        or dilation != 1
        or is_causal
        or backend not in (None, "blackwell-fna")
    ):
        return attention_layout.na3d(
            q,
            k,
            v,
            kernel_size,
            dilation=dilation,
            is_causal=is_causal,
            backend=backend,
        )
    q = token_permute_torch(q, _Q_TILE, (1, 1, 1), True)
    k = token_permute_torch(k, _KV_TILE, (1, 1, 1), True)
    v = token_permute_torch(v, _KV_TILE, (1, 1, 1), True)
    output = attention_core(q, k, v, _PERSISTENT)
    return token_unpermute_torch(output, _LAYOUT, _Q_TILE, (1, 1, 1), True)


def install(strata, *, persistent=False):
    global _PERSISTENT
    if strata.training:
        raise ValueError("Semantic attention layout requires eval inference")
    count = 0
    for module in strata.backbone.modules():
        if (
            isinstance(module, layers.Natten3DSelfAttention)
            and not module.do_depthwise_attention
        ):
            if (
                module.attn_kernel != 9
                or module.num_heads != 16
                or module.head_dim != 64
                or module.na_dilation != 1
            ):
                raise ValueError(
                    "Semantic layout requires released 9-cube/16heads64/dilation1 geometry"
                )
            count += 1
    if count != 12:
        raise ValueError(
            "Semantic layout requires twelve neighborhood-attention blocks"
        )
    _PERSISTENT = persistent
    layers._na3d_func = na3d
