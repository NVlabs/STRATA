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
import copy
from functools import partial

import pytest
import torch
from torch import nn

import screamcast.depthwise_conv as depthwise


@pytest.mark.parametrize(
    "limit,shape",
    [
        (256, (5, 2, 4, 4)),  # Batch chunks; two padded samples fit.
        (128, (2, 4, 4, 4)),  # One sample also needs channel chunks.
        (1000, (2, 4, 4, 4)),  # Everything fits without splitting.
        (64, (2, 4, 4, 4)),  # Exactly one padded channel fits.
    ],
)
def test_chunked_matches_conv2d_outputs_and_gradients(monkeypatch, limit, shape):
    torch.manual_seed(7)
    monkeypatch.setattr(depthwise, "_MAX_CONV_ELEMENTS", limit)
    reference = nn.Conv2d(
        shape[1], shape[1], 5, padding=2, groups=shape[1], padding_mode="replicate"
    ).double()
    conv = copy.deepcopy(reference)
    conv.forward = partial(depthwise._chunked_depthwise_conv, conv)
    x = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    x_reference = x.detach().clone().requires_grad_()
    expected = reference(x_reference)
    original_build = depthwise._build_chunked_depthwise_conv
    calls = []

    def checked_build(conv, channel_chunk):
        apply = original_build(conv, channel_chunk)

        def checked_apply(part, weight, bias):
            channels = shape[1] if channel_chunk is None else channel_chunk
            # The production 5x5 replicate-padded kernel expands 4x4 to 8x8.
            assert part.shape[0] * channels * 64 <= limit
            calls.append((part.shape[0], channel_chunk))
            return apply(part, weight, bias)

        return checked_apply

    monkeypatch.setattr(depthwise, "_build_chunked_depthwise_conv", checked_build)
    actual = conv(x)
    torch.testing.assert_close(actual, expected)
    if limit == 1000:
        assert calls == [(shape[0], None)]
    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    torch.testing.assert_close(x.grad, x_reference.grad)
    torch.testing.assert_close(conv.weight.grad, reference.weight.grad)
    torch.testing.assert_close(conv.bias.grad, reference.bias.grad)


def test_adapter_deepcopy_uses_own_parameters():
    conv = nn.Conv2d(2, 2, 5, padding=2, groups=2, padding_mode="replicate")
    conv.forward = partial(depthwise._chunked_depthwise_conv, conv)
    clone = copy.deepcopy(conv).double()
    with torch.no_grad():
        clone.weight.zero_()
        clone.bias.fill_(3)
    x = torch.randn(2, 2, 4, 4, dtype=torch.float64)
    torch.testing.assert_close(clone(x), torch.full_like(x, 3))


def test_adapter_rejects_plane_over_indexing_limit(monkeypatch):
    monkeypatch.setattr(depthwise, "_MAX_CONV_ELEMENTS", 63)
    conv = nn.Conv2d(2, 2, 5, padding=2, groups=2, padding_mode="replicate")
    with pytest.raises(ValueError, match="one conditioning plane"):
        depthwise._chunked_depthwise_conv(conv, torch.ones(2, 2, 4, 4))
