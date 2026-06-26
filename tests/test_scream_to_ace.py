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
import pytest
import torch

from screamcast.ace._channels import ACE_VARIABLE_NAMES
from screamcast.ace._scream_to_ace import ScreamToACEState
from screamcast.thermodynamics import temperature_from_potential_temperature
from screamcast.vertical_interpolation import regrid_hybrid_vertical

REV2_LEVELS = list(range(8, 32))
REV2_VARIABLE_NAMES = [
    f"{name}_{level}"
    for name in ["PotentialTemperature", "U", "V", "z_mid", "omega", "qv"]
    for level in REV2_LEVELS
] + ["T_2m", "ps"]


def test_scream_to_ace_state_requires_surface_fields():
    variable_names = [
        "PotentialTemperature_0",
        "U_0",
        "V_0",
        "qv_0",
        "T_2m",
    ]

    with pytest.raises(ValueError, match="missing required surface fields: \\['ps'\\]"):
        ScreamToACEState(
            scream_variable_names=variable_names,
            scream_hyam_sub=torch.tensor([0.1, 0.9], dtype=torch.float64),
            scream_hybm_sub=torch.tensor([0.9, 0.1], dtype=torch.float64),
        )


def test_scream_to_ace_state_stacks_regridded_fields_and_surface_channels():
    module = ScreamToACEState(
        scream_variable_names=list(REV2_VARIABLE_NAMES),
        scream_hyam_sub=torch.linspace(
            0.20, 0.98, len(REV2_LEVELS), dtype=torch.float64
        ),
        scream_hybm_sub=torch.linspace(
            0.80, 0.02, len(REV2_LEVELS), dtype=torch.float64
        ),
    ).eval()

    scream_input_state = torch.zeros((2, len(REV2_VARIABLE_NAMES), 2, 3))
    idx = {name: i for i, name in enumerate(REV2_VARIABLE_NAMES)}
    ps = torch.tensor(
        [
            [[90000.0, 91000.0, 92000.0], [93000.0, 94000.0, 95000.0]],
            [[96000.0, 97000.0, 98000.0], [99000.0, 100000.0, 101000.0]],
        ],
        dtype=torch.float32,
    )
    t2m = torch.tensor(
        [
            [[280.0, 281.0, 282.0], [283.0, 284.0, 285.0]],
            [[286.0, 287.0, 288.0], [289.0, 290.0, 291.0]],
        ],
        dtype=torch.float32,
    )
    for level in REV2_LEVELS:
        scream_input_state[:, idx[f"PotentialTemperature_{level}"]] = 300.0
        scream_input_state[:, idx[f"U_{level}"]] = 5.0
        scream_input_state[:, idx[f"V_{level}"]] = -3.0
        scream_input_state[:, idx[f"qv_{level}"]] = 0.01
    scream_input_state[:, idx["T_2m"]] = t2m
    scream_input_state[:, idx["ps"]] = ps

    ace_state = module(scream_input_state)

    theta_ace_lfirst, p_ace_lfirst = regrid_hybrid_vertical(
        src_values=torch.full((len(REV2_LEVELS), 2, 2, 3), 300.0, dtype=torch.float32),
        src_ak_interface=module.scream_hyam_sub,
        src_bk_interface=module.scream_hybm_sub,
        surface_pressure=ps,
        src_p0=100000.0,
        target_ak_interface=module.ace_ak,
        target_bk_interface=module.ace_bk,
        target_p0=1.0,
    )
    expected_temperature = (
        temperature_from_potential_temperature(
            theta_ace_lfirst, p_ace_lfirst, p0_pa=100000.0
        )
        .movedim(0, 1)
        .to(dtype=scream_input_state.dtype)
    )

    assert ace_state.shape == (2, len(ACE_VARIABLE_NAMES), 2, 3)
    torch.testing.assert_close(ace_state[:, :8], expected_temperature)
    torch.testing.assert_close(ace_state[:, 8:16], torch.full((2, 8, 2, 3), 0.01))
    torch.testing.assert_close(ace_state[:, 16:24], torch.full((2, 8, 2, 3), 5.0))
    torch.testing.assert_close(ace_state[:, 24:32], torch.full((2, 8, 2, 3), -3.0))
    torch.testing.assert_close(ace_state[:, 32], ps)
    torch.testing.assert_close(ace_state[:, 33], t2m)
    torch.testing.assert_close(ace_state[:, 34], torch.full((2, 2, 3), 5.0))
    torch.testing.assert_close(ace_state[:, 35], torch.full((2, 2, 3), -3.0))
    torch.testing.assert_close(ace_state[:, 36], t2m)
    torch.testing.assert_close(ace_state[:, 37], torch.full((2, 2, 3), 0.01))
