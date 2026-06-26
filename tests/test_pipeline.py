#!/usr/bin/env python3
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

from screamcast.dali_ext_src import ScreamV2
from screamcast.model_pipelines import MixedPredictionAsymmetric
from screamcast.normalization import RunningNorm2d


class _StubNet(torch.nn.Module):
    """Minimal in->out network (1x1 conv) standing in for a real architecture."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, index):
        return self.conv(x)


def create_pipeline(
    variables_prognostic,
    variables_forcing,
    variables_diagnostic,
    variables_prognostic_state,
    plevel=4,
    level_start=3,
    level_end=128,
):
    """Helper to create MixedPredictionAsymmetric pipeline"""

    in_channels = ScreamV2.num_of_input_channels(
        variables_prognostic, variables_forcing, plevel, level_start, level_end
    )
    out_channels = ScreamV2.num_of_output_channels(
        variables_prognostic, variables_diagnostic, plevel, level_start, level_end
    )

    network = _StubNet(in_channels, out_channels)

    return MixedPredictionAsymmetric(
        network=network,
        loss_fn=torch.nn.MSELoss(),
        input_norm=RunningNorm2d(in_channels, fit_batches=1),
        target_norm=RunningNorm2d(out_channels, fit_batches=1),
        plevel=plevel,
        level_start=level_start,
        level_end=level_end,
        variables_prognostic=variables_prognostic,
        variables_forcing=variables_forcing,
        variables_diagnostic=variables_diagnostic,
        variables_prognostic_state=variables_prognostic_state,
    )


def test_step_prediction_consistency():
    """Test that step() correctly applies diff vs state prediction per variable"""
    variables_prognostic = ("U", "T_2m")
    variables_forcing = ("coszr",)
    variables_diagnostic = ()
    variables_prognostic_state = ("T_2m",)  # Only T_2m uses state prediction

    pipeline = create_pipeline(
        variables_prognostic,
        variables_forcing,
        variables_diagnostic,
        variables_prognostic_state,
    )

    batch_size, height, width = 2, 64, 64
    in_channels = ScreamV2.num_of_input_channels(
        variables_prognostic, variables_forcing, 4, 3, 128
    )

    # Create mock state
    state = torch.randn(batch_size, in_channels, height, width)
    index = torch.randint(0, 1000, (batch_size, height, width))
    ranges_input = ScreamV2.ranges_input(
        variables_prognostic, variables_forcing, 4, 3, 128
    )
    ranges_output = ScreamV2.ranges_output(
        variables_prognostic, variables_diagnostic, 4, 3, 128
    )

    # compare the prediction using step() vs manual processing
    with torch.no_grad():
        # Manual processing
        pred = pipeline.network(pipeline.input_norm(state), index)
        pred_denorm = pipeline.target_norm.denormalize(pred)
        u_current = state[:, ranges_input["U"]]
        u_pred = pred_denorm[:, ranges_output["U"]]
        t2m_pred = pred_denorm[:, ranges_output["T_2m"]]

        u_next_manual = u_pred + u_current
        t2m_next_manual = t2m_pred

        # Get pipeline result from .step()
        output_full, _ = pipeline.step(state, index)
        u_next_pipeline = output_full[:, ranges_output["U"]]
        t2m_next_pipeline = output_full[:, ranges_output["T_2m"]]

        # compare the results
        torch.testing.assert_close(u_next_manual, u_next_pipeline, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            t2m_next_manual, t2m_next_pipeline, rtol=1e-5, atol=1e-6
        )


# ---------------------------------------------------------------------------
# Helpers for constraint tests (qv softplus, precip relu)
# ---------------------------------------------------------------------------

_PLEVEL, _LEVEL_START, _LEVEL_END = 4, 3, 7  # small level range for speed
_B, _H, _W = 2, 8, 8


class _ConstantNetwork(torch.nn.Module):
    """Stub network that always returns a tensor filled with a fixed scalar."""

    def __init__(self, out_channels, value):
        super().__init__()
        self.out_channels = out_channels
        self.value = value

    def forward(self, x, index):
        return torch.full(
            (x.shape[0], self.out_channels, x.shape[2], x.shape[3]),
            self.value,
            dtype=x.dtype,
        )


def _make_constraint_pipeline(
    variables_prognostic,
    variables_diagnostic,
    network_value,
    do_qv_softplus=False,
    do_precip_relu=False,
    qv_input_value=0.005,
):
    """Build a minimal MixedPredictionAsymmetric with fitted norms and a stub network."""
    variables_forcing = ()
    in_ch = ScreamV2.num_of_input_channels(
        variables_prognostic, variables_forcing, _PLEVEL, _LEVEL_START, _LEVEL_END
    )
    out_ch = ScreamV2.num_of_output_channels(
        variables_prognostic, variables_diagnostic, _PLEVEL, _LEVEL_START, _LEVEL_END
    )
    input_norm = RunningNorm2d(in_ch, fit_batches=1)
    target_norm = RunningNorm2d(out_ch, fit_batches=1)
    input_norm(
        torch.full((_B, in_ch, _H, _W), qv_input_value)
        + torch.randn(_B, in_ch, _H, _W) * 0.001
    )
    target_norm(torch.randn(_B, out_ch, _H, _W) * 0.001)
    return MixedPredictionAsymmetric(
        network=_ConstantNetwork(out_ch, network_value),
        loss_fn=torch.nn.MSELoss(),
        input_norm=input_norm,
        target_norm=target_norm,
        plevel=_PLEVEL,
        level_start=_LEVEL_START,
        level_end=_LEVEL_END,
        variables_prognostic=variables_prognostic,
        variables_forcing=variables_forcing,
        variables_diagnostic=variables_diagnostic,
        variables_prognostic_state=(),
        do_qv_softplus=do_qv_softplus,
        do_precip_relu=do_precip_relu,
    )


# ---------------------------------------------------------------------------
# qv softplus constraint tests
# ---------------------------------------------------------------------------


def test_qv_softplus_non_negative_for_positive_input():
    """Softplus constraint ensures qv + dqv >= 0 when input qv > 0."""
    pipeline = _make_constraint_pipeline(
        ("qv",), (), network_value=-50.0, do_qv_softplus=True
    )
    in_ch = ScreamV2.num_of_input_channels(
        ("qv",), (), _PLEVEL, _LEVEL_START, _LEVEL_END
    )
    state = torch.zeros(_B, in_ch, _H, _W)
    index = torch.zeros(_B, _H, _W, dtype=torch.long)
    with torch.no_grad():
        out = pipeline.forward_from_flat(state, index)
    in_sl, out_sl = pipeline.ranges_input["qv"], pipeline.ranges_output["qv"]
    qv_std = (
        pipeline.input_norm.var[in_sl].sqrt() + pipeline.input_norm.eps
    ).unsqueeze(0)
    qv_mean = pipeline.input_norm.mean[in_sl].unsqueeze(0)
    dqv_std = (
        pipeline.target_norm.var[out_sl].sqrt() + pipeline.target_norm.eps
    ).unsqueeze(0)
    dqv_mean = pipeline.target_norm.mean[out_sl].unsqueeze(0)
    qv_phys = pipeline.input_norm(state)[:, in_sl] * qv_std + qv_mean
    dqv_phys = out[:, out_sl] * dqv_std + dqv_mean
    # The pipeline guarantees qv_new >= 0 exactly, but reconstructing it here via
    # a separate normalize/denormalize roundtrip introduces float32 FP roundoff
    # (~1e-10). Allow a small absolute tolerance well below physical qv scales.
    tol = 1e-6
    assert (
        qv_phys + dqv_phys >= -tol
    ).all(), f"qv_new has negatives: min={(qv_phys + dqv_phys).min():.2e}"


def test_qv_softplus_recovers_negative_input():
    """Softplus constraint ensures qv_new >= 0 even when input qv < 0 (rollout drift)."""
    pipeline = _make_constraint_pipeline(
        ("qv",), (), network_value=0.0, do_qv_softplus=True
    )
    in_ch = ScreamV2.num_of_input_channels(
        ("qv",), (), _PLEVEL, _LEVEL_START, _LEVEL_END
    )
    ranges_input = ScreamV2.ranges_input(("qv",), (), _PLEVEL, _LEVEL_START, _LEVEL_END)
    state = torch.zeros(_B, in_ch, _H, _W)
    state[:, ranges_input["qv"], :, :] = -0.005  # negative physical qv
    index = torch.zeros(_B, _H, _W, dtype=torch.long)
    with torch.no_grad():
        out = pipeline.forward_from_flat(state, index)
    in_sl, out_sl = pipeline.ranges_input["qv"], pipeline.ranges_output["qv"]
    qv_std = (
        pipeline.input_norm.var[in_sl].sqrt() + pipeline.input_norm.eps
    ).unsqueeze(0)
    qv_mean = pipeline.input_norm.mean[in_sl].unsqueeze(0)
    dqv_std = (
        pipeline.target_norm.var[out_sl].sqrt() + pipeline.target_norm.eps
    ).unsqueeze(0)
    dqv_mean = pipeline.target_norm.mean[out_sl].unsqueeze(0)
    qv_phys = pipeline.input_norm(state)[:, in_sl] * qv_std + qv_mean
    dqv_phys = out[:, out_sl] * dqv_std + dqv_mean
    assert (qv_phys < 0).all(), "Precondition: input qv must be negative"
    # Same FP-roundoff tolerance as test_qv_softplus_non_negative_for_positive_input;
    # observed ~-4.66e-10 on some CI hardware due to normalize/denormalize roundoff.
    tol = 1e-6
    assert (
        qv_phys + dqv_phys >= -tol
    ).all(), f"qv_new still negative: min={(qv_phys + dqv_phys).min():.2e}"


# ---------------------------------------------------------------------------
# precip relu constraint test
# ---------------------------------------------------------------------------


def test_precip_relu_non_negative():
    """ReLU constraint ensures physical precip >= 0 even for large negative network output."""
    variables_prognostic = ("qv",)
    variables_diagnostic = ("precip_ice_surf_mass_flux", "precip_liq_surf_mass_flux")
    pipeline = _make_constraint_pipeline(
        variables_prognostic,
        variables_diagnostic,
        network_value=-100.0,
        do_precip_relu=True,
    )
    in_ch = ScreamV2.num_of_input_channels(
        variables_prognostic, (), _PLEVEL, _LEVEL_START, _LEVEL_END
    )
    state = torch.rand(_B, in_ch, _H, _W) * 0.01 + 0.001
    index = torch.zeros(_B, _H, _W, dtype=torch.long)
    with torch.no_grad():
        out = pipeline.forward_from_flat(state, index)
    ranges_output = ScreamV2.ranges_output(
        variables_prognostic, variables_diagnostic, _PLEVEL, _LEVEL_START, _LEVEL_END
    )
    for var in ("precip_ice_surf_mass_flux", "precip_liq_surf_mass_flux"):
        sl = ranges_output[var]
        p_std = (
            pipeline.target_norm.var[sl].sqrt() + pipeline.target_norm.eps
        ).unsqueeze(0)
        p_mean = pipeline.target_norm.mean[sl].unsqueeze(0)
        p_phys = out[:, sl] * p_std + p_mean
        assert (
            p_phys >= -1e-9
        ).all(), f"{var} physical values negative: min={p_phys.min():.2e}"
