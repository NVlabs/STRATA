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
"""Latency acceptance with serially correlated samples; Cirrus-42 (AI agent)."""

import math

import numpy as np


def _bootstrap_medians(samples, rng, draws, block_size):
    starts = rng.integers(
        0, len(samples), (draws, math.ceil(len(samples) / block_size))
    )
    indices = (starts[:, :, None] + np.arange(block_size)) % len(samples)
    indices = indices.reshape(draws, -1)[:, : len(samples)]
    return np.median(samples[indices], axis=1)


def latency_gate(
    decoder, full, *, draws=20000, seed=20261020, block_size=5, isolated_scope="decoder"
):
    """Require positive intervals targeting joint 95% confidence for four controls.

    Circular blocks preserve dependence within five consecutive measurements.
    Bonferroni adjusts percentile intervals for all four comparisons together.
    These intervals describe within-run timing variation; neighboring controls
    additionally guard against drift. Confirm selected kernels in a fresh run.
    Each scope has before/candidate/after timing dictionaries containing cuda_ms.
    """
    rng = np.random.default_rng(seed)
    comparisons = {}
    for scope, timings in ((isolated_scope, decoder), ("full", full)):
        values = {
            key: np.asarray(timings[key]["cuda_ms"], dtype=float)
            for key in ("before", "candidate", "after")
        }
        for samples in values.values():
            if (
                samples.ndim != 1
                or not np.isfinite(samples).all()
                or (samples <= 0).any()
            ):
                raise ValueError(
                    "CUDA timings must be finite positive one-dimensional samples"
                )
        enough = min(map(len, values.values())) >= 30
        candidate = values["candidate"]
        candidate_boot = (
            _bootstrap_medians(candidate, rng, draws, block_size) if enough else None
        )
        for neighbor in ("before", "after"):
            control = values[neighbor]
            observed = (
                (1 - np.median(candidate) / np.median(control))
                if len(candidate) and len(control)
                else None
            )
            if enough:
                control_boot = _bootstrap_medians(control, rng, draws, block_size)
                gains = 1 - candidate_boot / control_boot
                # Four two-sided 98.75% intervals target simultaneous 95% confidence.
                lower, upper = np.quantile(gains, [0.05 / 8, 1 - 0.05 / 8]).tolist()
                significant = lower > 0
            else:
                lower, upper, significant = None, None, False
            comparisons[f"{scope}_{neighbor}"] = dict(
                control_samples=len(control),
                candidate_samples=len(candidate),
                median_latency_reduction=observed,
                latency_reduction_ci=[lower, upper],
                significant=significant,
                status="sufficient_samples" if enough else "need_at_least_30_samples",
            )
    return dict(
        method="circular block bootstrap of median latency ratio; Bonferroni four comparisons",
        family_confidence=0.95,
        comparison_confidence=0.9875,
        block_size=block_size,
        draws=draws,
        seed=seed,
        minimum_samples=30,
        comparisons=comparisons,
        significant=all(c["significant"] for c in comparisons.values()),
    )
