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
import torch
import torch.nn as nn
from physicsnemo.experimental.models.strata.layers import _build_chunked_depthwise_conv

_MAX_CONV_ELEMENTS = 2**31 - 1


def _chunked_depthwise_conv(conv: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    """Run Strata's conditioning convolution with automatically sized vmap chunks."""
    h, w = x.shape[-2:]
    ph, pw = h + 2 * conv.padding[0], w + 2 * conv.padding[1]
    oh = (ph - conv.dilation[0] * (conv.kernel_size[0] - 1) - 1) // conv.stride[0] + 1
    ow = (pw - conv.dilation[1] * (conv.kernel_size[1] - 1) - 1) // conv.stride[1] + 1
    area = max(h * w, oh * ow, ph * pw if conv.padding_mode != "zeros" else 0)
    batch_chunk = min(x.shape[0], max(1, _MAX_CONV_ELEMENTS // (x.shape[1] * area)))
    channel_chunk = min(x.shape[1], _MAX_CONV_ELEMENTS // (batch_chunk * area))
    if channel_chunk < 1:
        raise ValueError("one conditioning plane exceeds the 32-bit indexing limit")
    apply = _build_chunked_depthwise_conv(
        conv, channel_chunk if channel_chunk < x.shape[1] else None
    )
    # Strata's bilinear conditioning convolutions always have a bias.
    if batch_chunk == x.shape[0]:
        return apply(x, conv.weight, conv.bias)
    return torch.cat(
        [apply(part, conv.weight, conv.bias) for part in x.split(batch_chunk)]
    )
