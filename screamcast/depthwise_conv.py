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
import math
import warnings
from functools import partial

import torch
import torch.nn as nn


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
    """Depthwise conv that chunks large vmapped inputs when needed."""

    def __init__(self, channels: int, *args, chunk_size: int | None = None, **kwargs):
        if "groups" in kwargs:
            raise ValueError("DepthwiseConv does not accept a groups argument")
        super().__init__(channels, channels, *args, **kwargs, groups=channels)
        self.chunk_size = chunk_size
        if chunk_size is not None:
            self.forward = _wrap_large_depthwise_conv(self, chunk_size)

    def forward(self, x, *args, **kwargs):
        n_chunks_size = 1
        if x is not None and self.chunk_size:
            n_chunks_size = max(1, x.numel() // self.chunk_size)

        size_of_chunk = math.ceil(x.numel() * x.dtype.itemsize / n_chunks_size)
        if size_of_chunk > 2**32:
            warnings.warn(
                f"Convolution {size_of_chunk=} larger than 2^32 so conv2d will revert to slow implementation or error out. Decrease {self.chunk_size=} option."
            )

        return super().forward(x, *args, **kwargs)
