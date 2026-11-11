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
"""Native-reference schedule scope contracts; Cirrus-42 (AI agent)."""

from types import SimpleNamespace

import pytest
from torch._inductor.runtime import triton_heuristics

from screamcast.native_reference_schedule import native_reference_schedule


def config(x, r):
    return SimpleNamespace(
        kwargs={"XBLOCK": x, "R0_BLOCK": r}, num_warps=16, num_stages=1
    )


@pytest.mark.parametrize("unrelated", (None, "shape", "name", "dtype"))
def test_only_native_backbone_residual_reference_is_pinned(monkeypatch, unrelated):
    calls = []

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return "compiled"

    monkeypatch.setattr(triton_heuristics, "cached_autotune", original)
    hints = {"x": 262144, "r0_": 1024}
    meta = {"signature": {"in_ptr0": "*fp32", "in_ptr1": "*bf16", "out_ptr2": "*bf16"}}
    name = "triton_red_fused__to_copy_add_native_layer_norm_0"
    if unrelated == "shape":
        hints["r0_"] = 128
    elif unrelated == "name":
        name = "triton_red_fused_native_layer_norm_0"
    elif unrelated == "dtype":
        meta["signature"]["in_ptr0"] = "*bf16"
    historical, alternative = config(8, 256), config(64, 64)
    configs = [alternative, historical]
    with native_reference_schedule() as observed:
        result = triton_heuristics.cached_autotune(
            hints, configs, meta, "reduction", inductor_meta={"kernel_name": name}
        )
    assert result == "compiled"
    received = calls[0][0][1]
    if unrelated is None:
        assert len(received) == 1
        assert received[0].kwargs == historical.kwargs
        assert received[0].num_warps == 16 and received[0].num_stages == 1
    else:
        assert received is configs
    assert calls[0][0][0] is hints and calls[0][0][2] is meta
    assert observed["native_residual_norm_kernels"] == (unrelated is None)
    assert triton_heuristics.cached_autotune is original
    assert configs == [alternative, historical]


@pytest.mark.parametrize("compiler_fails", (False, True))
def test_explicit_schedule_without_initial_candidate_restores_scope(
    monkeypatch, compiler_fails
):
    received = []

    def original(size_hints, configs, *args, **kwargs):
        received.extend(configs)
        if compiler_fails:
            raise RuntimeError("compiler failed")
        return "compiled"

    monkeypatch.setattr(triton_heuristics, "cached_autotune", original)

    def compile_reference():
        with native_reference_schedule():
            return triton_heuristics.cached_autotune(
                {"x": 262144, "r0_": 1024},
                [config(8, 512), config(64, 64)],
                {
                    "signature": {
                        "in_ptr0": "*fp32",
                        "in_ptr1": "*bf16",
                        "out_ptr2": "*bf16",
                    }
                },
                "reduction",
                inductor_meta={
                    "kernel_name": "triton_red_fused__to_copy_add_native_layer_norm_0"
                },
            )

    if compiler_fails:
        with pytest.raises(RuntimeError, match="compiler failed"):
            compile_reference()
    else:
        assert compile_reference() == "compiled"
    assert len(received) == 1
    assert received[0].kwargs == {"XBLOCK": 8, "R0_BLOCK": 256}
    assert received[0].num_warps == 16 and received[0].num_stages == 1
    assert triton_heuristics.cached_autotune is original
