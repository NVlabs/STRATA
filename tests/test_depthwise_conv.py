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
from torch import nn

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


def test_depthwise_conv():
    conv = DepthwiseConv(10, chunk_size=2, padding=2, kernel_size=5)
    x = torch.ones(10, 10, 32, 32)
    assert conv(x).shape == (10, 10, 32, 32)


def test_depthwise_conv_without_chunking():
    conv = DepthwiseConv(10, chunk_size=None, padding=2, kernel_size=5)
    x = torch.ones(10, 10, 32, 32)
    assert conv(x).shape == (10, 10, 32, 32)
