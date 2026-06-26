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

from screamcast.ace._vertical_coordinate import (
    AK_ACE2_8L,
    BK_ACE2_8L,
)
from screamcast.thermodynamics import temperature_from_potential_temperature
from screamcast.vertical_interpolation import interpolate_1d, regrid_hybrid_vertical


def test_interpolate_1d_clamps_endpoints_and_interpolates_interior():
    src_coordinate = torch.tensor(
        [[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]], dtype=torch.float64
    )
    src_values = torch.tensor(
        [[2.0, 20.0], [6.0, 60.0], [10.0, 100.0]], dtype=torch.float64
    )
    target_coordinate = torch.tensor(
        [[0.0, 0.0], [2.0, 20.0], [6.0, 60.0]], dtype=torch.float64
    )

    interpolated = interpolate_1d(
        src_coordinate=src_coordinate,
        src_values=src_values,
        target_coordinate=target_coordinate,
    )

    expected = torch.tensor(
        [[2.0, 20.0], [4.0, 40.0], [10.0, 100.0]], dtype=torch.float64
    )
    torch.testing.assert_close(interpolated, expected)


def test_regrid_hybrid_vertical_log_pressure_and_temperature_conversion():
    sp = torch.tensor([100000.0, 90000.0], dtype=torch.float64)
    src_ak_interface = torch.tensor([0.0, 50000.0, 100000.0], dtype=torch.float64)
    src_bk_interface = torch.zeros_like(src_ak_interface)
    ace_ak_interface = torch.tensor(
        [0.0, 25000.0, 50000.0, 100000.0], dtype=torch.float64
    )
    ace_bk_interface = torch.zeros_like(ace_ak_interface)

    src_press = torch.tensor([25000.0, 75000.0], dtype=torch.float64)
    src_values = torch.stack(
        [
            2.0 * torch.log(src_press) + 1.0,
            2.0 * torch.log(src_press) + 11.0,
        ],
        axis=1,
    )

    remapped, target_pressure = regrid_hybrid_vertical(
        src_values=src_values,
        src_ak_interface=src_ak_interface,
        src_bk_interface=src_bk_interface,
        surface_pressure=sp,
        src_p0=1.0,
        target_ak_interface=ace_ak_interface,
        target_bk_interface=ace_bk_interface,
        target_p0=1.0,
    )

    expected_col0 = torch.tensor(
        [
            src_values[0, 0],
            2.0 * torch.log(torch.tensor(37500.0, dtype=torch.float64)) + 1.0,
            src_values[1, 0],
        ],
        dtype=torch.float64,
    )
    expected_col1 = expected_col0 + 10.0
    torch.testing.assert_close(remapped[:, 0], expected_col0, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(remapped[:, 1], expected_col1, rtol=1e-6, atol=1e-6)

    theta = torch.full_like(remapped, 300.0)
    temperature = temperature_from_potential_temperature(
        theta, target_pressure, p0_pa=100000.0
    )
    expected_temperature = 300.0 * torch.pow(
        target_pressure / 100000.0, 287.05 / 1004.0
    )
    torch.testing.assert_close(temperature, expected_temperature, rtol=1e-6, atol=1e-6)


def test_regrid_hybrid_vertical_rejects_unaligned_source_coefficients():
    sp = torch.full((2, 3), 100000.0, dtype=torch.float64)
    src_values = torch.ones((32, 2, 3), dtype=torch.float64)
    src_ak_mid_128 = torch.linspace(100.0, 90000.0, 128, dtype=torch.float64)
    src_bk_mid_128 = torch.linspace(0.0, 1.0, 128, dtype=torch.float64)
    try:
        regrid_hybrid_vertical(
            src_values=src_values,
            src_ak_interface=src_ak_mid_128,
            src_bk_interface=src_bk_mid_128,
            surface_pressure=sp,
            src_p0=1.0,
            target_ak_interface=AK_ACE2_8L,
            target_bk_interface=BK_ACE2_8L,
            target_p0=1.0,
        )
        raise AssertionError("Expected ValueError for unaligned source coefficients.")
    except ValueError as exc:
        assert "Source coefficient length mismatch" in str(exc)
