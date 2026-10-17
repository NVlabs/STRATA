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
from functools import partial
from typing import Literal

import torch
import torch.nn as nn

_MAX_CONV_ELEMENTS = 2**31 - 1


def _chunked_depthwise_conv(conv: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    """Apply grouped conv in the largest index-safe batch or channel chunks.

    Include explicit padding and the output in the element bound. Bias stays
    outside autocast convolution, matching the existing vmap operator.
    """
    h, w = x.shape[-2:]
    ph, pw = h + 2 * conv.padding[0], w + 2 * conv.padding[1]
    oh = (ph - conv.dilation[0] * (conv.kernel_size[0] - 1) - 1) // conv.stride[0] + 1
    ow = (pw - conv.dilation[1] * (conv.kernel_size[1] - 1) - 1) // conv.stride[1] + 1
    area = max(h * w, oh * ow, ph * pw if conv.padding_mode != "zeros" else 0)
    dim, chunk_size = 0, _MAX_CONV_ELEMENTS // (x.shape[1] * area)
    if chunk_size < 1:
        dim, chunk_size = 1, _MAX_CONV_ELEMENTS // (x.shape[0] * area)
        if chunk_size < 1:
            raise ValueError(
                "one conditioning channel exceeds the 32-bit indexing limit"
            )
    outputs = []
    for i, part in enumerate(x.split(chunk_size, dim=dim)):
        weight, bias = conv.weight, conv.bias
        if dim == 1:
            start = i * chunk_size
            weight = weight[start : start + part.shape[1]]
            if bias is not None:
                bias = bias[start : start + part.shape[1]]
        part = part.contiguous()
        padding = conv.padding
        if conv.padding_mode != "zeros":
            part = nn.functional.pad(
                part, conv._reversed_padding_repeated_twice, mode=conv.padding_mode
            )
            padding = (0, 0)
        output = nn.functional.conv2d(
            part,
            weight,
            None,
            conv.stride,
            padding,
            conv.dilation,
            groups=part.shape[1],
        )
        if bias is not None:
            output = output + bias[None, :, None, None]
        outputs.append(output)
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=dim)


def _wrap_large_depthwise_conv(conv: nn.Conv2d, chunk_size: int = 4):
    """Work around the conv2d element-count restriction for depthwise convs."""
    if conv.groups != conv.out_channels:
        raise ValueError("only works with depthwise convolution")

    func = torch.vmap(
        partial(
            _apply_conv2d,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            padding_mode=conv.padding_mode,
        ),
        chunk_size=chunk_size,
    )
    bias = conv.bias
    if bias is None:
        bias = torch.zeros(
            conv.weight.shape[0], device=conv.weight.device, dtype=conv.weight.dtype
        )

    def apply(x: torch.Tensor) -> torch.Tensor:
        return func(x, conv.weight, bias)

    return torch.vmap(apply)


def _apply_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    stride,
    padding,
    dilation,
    padding_mode,
) -> torch.Tensor:
    x = x.unsqueeze(0)
    w = weight.unsqueeze(0)
    bias = bias.unsqueeze(0)
    if padding_mode != "zeros":
        pad_h, pad_w = padding
        x = torch.nn.functional.pad(x, (pad_w, pad_w, pad_h, pad_h), mode=padding_mode)
        padding = (0, 0)
    return torch.nn.functional.conv2d(
        x,
        w,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )[0]


class DepthwiseConv(torch.nn.Conv2d):
    """Depthwise conv: automatic safe chunks, explicit channel chunks, or None."""

    def __init__(
        self,
        channels: int,
        *args,
        chunk_size: int | None | Literal["auto"] = "auto",
        **kwargs,
    ):
        if "groups" in kwargs:
            raise ValueError("DepthwiseConv does not accept a groups argument")
        super().__init__(channels, channels, *args, **kwargs, groups=channels)
        self.chunk_size = chunk_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chunk_size == "auto":
            return _chunked_depthwise_conv(self, x)
        if self.chunk_size is not None:
            return _wrap_large_depthwise_conv(self, self.chunk_size)(x)
        return super().forward(x)
