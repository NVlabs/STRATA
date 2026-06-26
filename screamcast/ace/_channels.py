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
"""Shared ACE/SCREAM channel metadata used by the rev2 workflow."""

from __future__ import annotations

P0_SCREAM = 100000.0

ACE_VARIABLE_NAMES: list[str] = [
    f"{v}{k}k" for v in ["t", "qtot", "u", "v"] for k in range(8)
] + ["sp", "skt", "u10m", "v10m", "t2m", "q2m"]

ACE_FORCING_NAMES: list[str] = [
    "land_fraction",
    "ocean_fraction",
    "sea_ice_fraction",
    "DSWRFtoa",
    "HGTsfc",
    "global_mean_co2",
]

SCREAM_3D_VARIABLE_NAMES: list[str] = [
    "PotentialTemperature",
    "U",
    "V",
    "qv",
    "z_mid",
    "omega",
]

SCREAM_SURFACE_VARIABLE_NAMES: list[str] = [
    "T_2m",
    "ps",
    "U_at_10m_above_surface",
    "V_at_10m_above_surface",
    "qv_2m",
]

SCREAM_VARIABLE_NAMES: list[str] = [
    f"{v}_{k}" for v in SCREAM_3D_VARIABLE_NAMES for k in range(32)
] + list(SCREAM_SURFACE_VARIABLE_NAMES)

if len(ACE_VARIABLE_NAMES) != 38:
    raise RuntimeError(
        f"ACE variable list must have 38 entries, got {len(ACE_VARIABLE_NAMES)}."
    )

if len(SCREAM_VARIABLE_NAMES) != 197:
    raise RuntimeError(
        "SCREAM variable list must have 197 entries, "
        f"got {len(SCREAM_VARIABLE_NAMES)}."
    )
