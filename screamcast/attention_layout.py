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
"""Fixed B8/T128 inference attention layout experiment; Cirrus-42 (AI agent)."""

import torch
from natten._libnatten import blackwell_na3d_forward
from natten.token_permute.torch_impl import token_permute_torch, token_unpermute_torch
from physicsnemo.experimental.models.strata import layers

_NATIVE_NA3D = layers._na3d_func
_Q_TILE = (8, 4, 8)
_KV_TILE = (4, 4, 8)
_LAYOUT = (24, 128, 128)


@torch.library.custom_op("cirrus42::attention_core", mutates_args=())
def attention_core(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
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
        False,
    )
    return output


@attention_core.register_fake
def attention_core_fake(q, k, v):
    return torch.empty(q.shape, dtype=q.dtype, device=q.device)


def na3d(q, k, v, kernel_size, dilation=1, is_causal=False, backend=None):
    if (
        q.shape != (8, 24, 128, 128, 2, 64)
        or kernel_size != (9, 9, 9)
        or dilation != 1
        or is_causal
        or backend not in (None, "blackwell-fna")
    ):
        return _NATIVE_NA3D(
            q,
            k,
            v,
            kernel_size,
            dilation=dilation,
            is_causal=is_causal,
            backend=backend,
        )
    # Traceable split/permute/copy accepts the original strides, so compilation
    # can combine normalization/RoPE with the tile stores. The opaque boundary
    # retains the exact native BF16 attention core and original scale/window.
    q = token_permute_torch(q, _Q_TILE, (1, 1, 1), True)
    k = token_permute_torch(k, _KV_TILE, (1, 1, 1), True)
    v = token_permute_torch(v, _KV_TILE, (1, 1, 1), True)
    output = attention_core(q, k, v)
    # Native also untile-converts logsumexp for backward; inference needs output.
    return token_unpermute_torch(output, _LAYOUT, _Q_TILE, (1, 1, 1), True)


def restore():
    layers._na3d_func = _NATIVE_NA3D


def install(strata):
    if strata.training:
        raise ValueError("Attention layout experiment supports eval inference only")
    layers._na3d_func = na3d
