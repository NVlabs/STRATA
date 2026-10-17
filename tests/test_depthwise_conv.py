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
import pytest
import torch
from torch import nn

import screamcast.depthwise_conv as depthwise
from screamcast.depthwise_conv import DepthwiseConv, _wrap_large_depthwise_conv


def test_wrap_depthwise_conv():
    conv = nn.Conv2d(10, 10, 5, padding=2, groups=10)
    conv_wrapped = _wrap_large_depthwise_conv(conv)
    x = torch.ones(5, 10, 32, 32)
    assert conv_wrapped(x).shape == (5, 10, 32, 32)
    torch.testing.assert_close(conv(x), conv_wrapped(x))


def test_wrap_depthwise_conv_without_bias():
    conv = nn.Conv2d(10, 10, 5, padding=2, groups=10, bias=False)
    conv_wrapped = _wrap_large_depthwise_conv(conv)
    x = torch.ones(5, 10, 32, 32)
    assert conv_wrapped(x).shape == (5, 10, 32, 32)
    torch.testing.assert_close(conv(x), conv_wrapped(x))


@pytest.mark.parametrize("chunk_size", ["auto", 2, None])
def test_depthwise_conv(chunk_size):
    conv = DepthwiseConv(10, chunk_size=chunk_size, padding=2, kernel_size=5)
    x = torch.ones(10, 10, 32, 32)
    assert conv(x).shape == (10, 10, 32, 32)


def test_depthwise_conv_without_chunking():
    conv = DepthwiseConv(10, chunk_size=None, padding=2, kernel_size=5)
    x = torch.ones(10, 10, 32, 32)
    assert conv(x).shape == (10, 10, 32, 32)


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("limit,shape", [(144, (5, 2, 4, 4)), (72, (2, 4, 4, 4))])
def test_chunked_matches_conv2d_outputs_and_gradients(monkeypatch, bias, limit, shape):
    torch.manual_seed(7)
    monkeypatch.setattr(depthwise, "_MAX_CONV_ELEMENTS", limit)
    reference = nn.Conv2d(
        shape[1],
        shape[1],
        3,
        padding=1,
        groups=shape[1],
        padding_mode="replicate",
        bias=bias,
    ).double()
    conv = DepthwiseConv(
        shape[1], 3, padding=1, padding_mode="replicate", bias=bias
    ).double()
    conv.load_state_dict(reference.state_dict(), strict=True)
    x = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    x_reference = x.detach().clone().requires_grad_()
    expected = reference(x_reference)
    original_pad = nn.functional.pad
    calls = []

    def check_limit(part, *args, **kwargs):
        padded = original_pad(part, *args, **kwargs)
        assert part.numel() <= limit and padded.numel() <= limit
        calls.append(part.shape)
        return padded

    monkeypatch.setattr(nn.functional, "pad", check_limit)
    actual = conv(x)
    torch.testing.assert_close(actual, expected)
    assert len(calls) > 1
    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    torch.testing.assert_close(x.grad, x_reference.grad)
    torch.testing.assert_close(conv.weight.grad, reference.weight.grad)
    if bias:
        torch.testing.assert_close(conv.bias.grad, reference.bias.grad)
