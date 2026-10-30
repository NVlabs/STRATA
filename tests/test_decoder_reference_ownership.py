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
"""Freeze canonical output before later calls; Cirrus-42 (AI agent)."""

from types import SimpleNamespace

import pytest
import torch

from screamcast import decoder_benchmark as benchmark


@pytest.mark.parametrize("candidate_correct", (True, False))
def test_canonical_reference_owns_storage_across_later_calls(
    monkeypatch, candidate_correct
):
    shared = torch.ones((1, 1, 1, 1, 1))
    state = {}
    monkeypatch.setattr(torch._dynamo, "reset", lambda: None)
    monkeypatch.setattr(benchmark, "restore_precompute", lambda: None)
    monkeypatch.setattr(
        benchmark, "apply_experiment", lambda model, name: state.update(name=name)
    )

    def compiler(module, *, backend):
        native = state["name"] == "cuda-fused-prepare-reuse"

        def forward():
            if native:
                shared.fill_(1)
                return shared
            # Model setup or another compiled call may reuse the returned workspace.
            shared.fill_(2)
            return torch.ones_like(shared) if candidate_correct else shared

        return forward

    monkeypatch.setattr(torch, "compile", compiler)
    with torch.inference_mode():
        _, error = benchmark.compile_phase(
            object(),
            (),
            SimpleNamespace(pixel_blocks=[]),
            "pixel-state-row-layout",
            [],
            [],
        )
    assert error["relative_rms"] == (0.0 if candidate_correct else 1.0)
    assert error["reference_mutation"]["relative_rms"] == 1.0
