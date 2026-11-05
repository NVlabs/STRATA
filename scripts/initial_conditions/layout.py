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
"""Layout of the one-frame ne1024pg2 L128 IC stores."""

NCOL = 25165824
NLEV = 128
NLEV_SUBSET = 32  # len(range(128)[3::4])
GRID = "ne1024pg2"

VARS_3D = (
    "PotentialTemperature",
    "U",
    "V",
    "qv",
    "omega",
    "z_mid",
)
VARS_2D = ("T_2m", "ps", "coszr")
VARS_1D = ("phis", "lat", "lon", "sgh30", "landfrac")


def zarr_name(tag: str) -> str:
    return f"scream_hiccup_ne1024pg2_{tag}.zarr"
