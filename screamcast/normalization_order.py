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
"""Preserve native Welford order with coalesced loads; Cirrus-42 (AI agent)."""

from types import FunctionType

from torch._inductor.runtime import triton_helpers
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, SliceLayout
from triton.experimental.gluon.language.extra import libdevice


def _gluon_helper(function):
    # Reuse the installed native formulas with Gluon tensor primitives.
    original = function.fn
    namespace = dict(original.__globals__, tl=gl)
    rebound = FunctionType(
        original.__code__,
        namespace,
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    return gluon.jit(rebound)


_update = _gluon_helper(triton_helpers.welford_reduce)
_combine = _gluon_helper(triton_helpers.welford_combine)


@gluon.jit
def native_order_kernel(
    A,
    B,
    OUT,
    N: gl.constexpr,
    SEMANTIC: gl.constexpr,
    RESIDUAL: gl.constexpr,
    COALESCED: gl.constexpr,
):
    if SEMANTIC:
        X: gl.constexpr = 64
        R: gl.constexpr = 64
        COMPUTE: gl.constexpr = BlockedLayout([4, 1], [16, 2], [1, 16], [0, 1])
        LOAD: gl.constexpr = BlockedLayout([1, 2], [1, 32], [16, 1], [1, 0])
    else:
        X: gl.constexpr = 8
        R: gl.constexpr = 256
        LOAD: gl.constexpr = BlockedLayout([1, 4], [1, 32], [8, 2], [1, 0])
        if COALESCED:
            COMPUTE: gl.constexpr = LOAD
        else:
            COMPUTE: gl.constexpr = BlockedLayout([4, 1], [2, 16], [1, 16], [0, 1])
    row = gl.program_id(0) * X + gl.arange(0, X, layout=SliceLayout(1, LOAD))
    channel = gl.arange(0, R, layout=SliceLayout(0, LOAD))
    offset = row[:, None] * 1024 + channel[None, :]
    loaded = gl.load(A + offset, row[:, None] < N, other=0)
    if RESIDUAL:
        loaded += gl.load(B + offset, row[:, None] < N, other=0).to(gl.float32)
    mean = gl.convert_layout(loaded, COMPUTE)
    m2 = gl.zeros_like(mean)
    weight = gl.full_like(mean, 1.0)
    for start in range(R, 1024, R):
        loaded = gl.load(A + offset + start, row[:, None] < N, other=0)
        if RESIDUAL:
            loaded += gl.load(B + offset + start, row[:, None] < N, other=0).to(
                gl.float32
            )
        current = gl.convert_layout(loaded, COMPUTE)
        mean, m2, weight = _update(current, mean, m2, weight, False)
    average, moment, count = gl.reduce((mean, m2, weight), 1, _combine)
    scale = libdevice.rsqrt(moment / 1024.0 + 1e-6)
    for start in range(0, 1024, R):
        loaded = gl.load(A + offset + start, row[:, None] < N, other=0)
        if RESIDUAL:
            loaded += gl.load(B + offset + start, row[:, None] < N, other=0).to(
                gl.float32
            )
        current = gl.convert_layout(loaded, COMPUTE)
        result = (current - average[:, None]) * scale[:, None]
        result = gl.convert_layout(result, LOAD)
        gl.store(OUT + offset + start, result, row[:, None] < N)
