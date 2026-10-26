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
"""Shared layout constants for ne1024pg2 L128 HICCUP ICs and Zarr stores."""

NCOL = 25165824
NLEV = 128
LEVEL_SUBSET = slice(3, None, 4)
NLEV_SUBSET = 32  # len(range(128)[3::4])
GRID = "ne1024pg2"

CASES = [
    {
        "tag": "2017091400_maria_pre",
        "storm": "Maria 2017",
        "timestamp": "2017-09-14T00:00:00",
    },
    {
        "tag": "2017091812_maria_post",
        "storm": "Maria 2017",
        "timestamp": "2017-09-18T12:00:00",
    },
    {
        "tag": "2018100600_michael_pre",
        "storm": "Michael 2018",
        "timestamp": "2018-10-06T00:00:00",
    },
    {
        "tag": "2018100912_michael_post",
        "storm": "Michael 2018",
        "timestamp": "2018-10-09T12:00:00",
    },
    {
        "tag": "2019082600_dorian_pre",
        "storm": "Dorian 2019",
        "timestamp": "2019-08-26T00:00:00",
    },
    {
        "tag": "2019090100_dorian_post",
        "storm": "Dorian 2019",
        "timestamp": "2019-09-01T00:00:00",
    },
    {
        "tag": "2022092200_ian_pre",
        "storm": "Ian 2022",
        "timestamp": "2022-09-22T00:00:00",
    },
    {
        "tag": "2022092612_ian_post",
        "storm": "Ian 2022",
        "timestamp": "2022-09-26T12:00:00",
    },
    {
        "tag": "2023102100_otis_pre",
        "storm": "Otis 2023",
        "timestamp": "2023-10-21T00:00:00",
    },
    {
        "tag": "2023102406_otis_post",
        "storm": "Otis 2023",
        "timestamp": "2023-10-24T06:00:00",
    },
]

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


def nc_name(timestamp: str) -> str:
    date, hour = timestamp.split("T")
    hh = hour.split(":")[0]
    return f"HICCUP.atm_era5.{date}.{hh}.ne1024pg2.L128.nc"


def hiccup_timestamp(timestamp: str) -> str:
    date, hour = timestamp.split("T")
    ymd = date.replace("-", "")
    hhmmss = hour.replace(":", "")
    return f"{ymd}.{hhmmss}"


def zarr_name(tag: str) -> str:
    return f"scream_hiccup_ne1024pg2_{tag}.zarr"
