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
"""Synchronized CUDA timing without model imports; Cirrus-42 (AI agent)."""

import statistics
import time

import torch


def measure(call, warmup, repeats, *, before_each=None):
    for _ in range(warmup):
        if before_each is not None:
            before_each()
        call()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    cuda_ms, wall_ms = [], []
    for _ in range(repeats):
        if before_each is not None:
            before_each()
            torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        wall = time.perf_counter()
        start.record()
        out = call()
        end.record()
        end.synchronize()
        cuda_ms.append(start.elapsed_time(end))
        wall_ms.append(1000 * (time.perf_counter() - wall))
        del out
    return dict(
        median_ms=statistics.median(cuda_ms),
        cuda_ms=cuda_ms,
        wall_ms=wall_ms,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
    )
