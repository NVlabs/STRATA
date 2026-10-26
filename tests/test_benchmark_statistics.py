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
"""Behavioral tests of timing acceptance, independent of GPU model imports."""

import numpy as np
import pytest

from screamcast.benchmark_statistics import latency_gate


def scope(control, candidate, after=None):
    return dict(
        before=dict(cuda_ms=control),
        candidate=dict(cuda_ms=candidate),
        after=dict(cuda_ms=control if after is None else after),
    )


@pytest.mark.parametrize("isolated_scope", ["decoder", "backbone"])
def test_stable_two_percent_gain_passes(isolated_scope):
    data = scope(np.full(50, 100.0), np.full(50, 98.0))
    result = latency_gate(data, data, isolated_scope=isolated_scope)
    assert result["significant"]
    for comparison in result["comparisons"].values():
        assert comparison["latency_reduction_ci"] == pytest.approx([0.02, 0.02])


def test_identical_samples_do_not_pass():
    data = scope(np.full(50, 100.0), np.full(50, 100.0))
    assert not latency_gate(data, data)["significant"]


def test_candidate_must_beat_trailing_control_and_full_network():
    faster = scope(np.full(50, 100.0), np.full(50, 98.0))
    drift = scope(np.full(50, 100.0), np.full(50, 98.0), np.full(50, 97.0))
    assert not latency_gate(faster, drift)["significant"]
    assert not latency_gate(drift, faster)["significant"]


def test_noisy_two_percent_gain_is_inconclusive():
    control = np.tile([80.0, 120.0], 25)
    candidate = control - 2
    data = scope(control, candidate)
    assert not latency_gate(data, data)["significant"]


def test_insufficient_samples_are_inconclusive():
    data = scope(np.full(20, 100.0), np.full(20, 98.0))
    result = latency_gate(data, data)
    assert not result["significant"]
    assert (
        result["comparisons"]["decoder_before"]["status"] == "need_at_least_30_samples"
    )


def test_invalid_measurements_are_rejected():
    data = scope(np.full(50, 100.0), np.full(50, float("nan")))
    with pytest.raises(ValueError, match="finite positive"):
        latency_gate(data, data)
