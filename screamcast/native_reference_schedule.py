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
"""Fix the historical native reference reduction; Cirrus-42 (AI agent).

The compiled native residual LayerNorm can autotune between distinct Welford
reduction trees. The optimized backbone preserves the historical X8/R256 tree.
This scope makes that reference choice explicit without changing the candidate.
"""

from contextlib import contextmanager
from unittest.mock import patch

from torch._inductor.runtime import triton_heuristics
from triton import Config


@contextmanager
def native_reference_schedule():
    """Pin only the B8/T128 native backbone residual normalization schedule."""
    original = triton_heuristics.cached_autotune
    observed = {"native_residual_norm_kernels": 0}

    def scheduled(size_hints, configs, triton_meta, heuristic_type, **kwargs):
        metadata = kwargs.get("inductor_meta") or {}
        signature = triton_meta["signature"]
        if (
            size_hints == {"x": 262144, "r0_": 1024}
            and metadata.get("kernel_name", "").startswith(
                "triton_red_fused__to_copy_add_native_layer_norm_"
            )
            and signature.get("in_ptr0") == "*fp32"
            and signature.get("in_ptr1") == "*bf16"
            and signature.get("out_ptr2") == "*bf16"
        ):
            # The original autotuner reaches R256 by dynamically shrinking an
            # X8/R512 candidate; it need not occur in the initial config list.
            configs = [
                Config({"XBLOCK": 8, "R0_BLOCK": 256}, num_warps=16, num_stages=1)
            ]
            observed["native_residual_norm_kernels"] += 1
        return original(size_hints, configs, triton_meta, heuristic_type, **kwargs)

    with patch.object(triton_heuristics, "cached_autotune", scheduled):
        yield observed
