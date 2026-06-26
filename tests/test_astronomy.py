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
import numpy as np
import pytest

from screamcast.astronomy import calculate_dswrftoa


@pytest.mark.parametrize(
    "time",
    [
        "2020-01-15T12:00:00",
        "2020-04-01T00:00:00",
        "2020-07-15T06:00:00",
        "2020-10-15T18:00:00",
    ],
)
def test_calculate_dswrftoa_regression(regtest, time):
    lat = np.linspace(-90, 90, 73)
    lon = np.linspace(0, 360, 144, endpoint=False)
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")

    dswrf = calculate_dswrftoa(
        np.datetime64(time), lat_grid, lon_grid, correct_eot=True
    )

    idx = np.unravel_index(np.argmax(dswrf), dswrf.shape)
    print(f"max:  {dswrf.max():.6f}", file=regtest)
    print(f"argmax lat: {lat[idx[0]]:.6f}", file=regtest)
    print(f"argmax lon: {lon[idx[1]]:.6f}", file=regtest)
    print(f"mean: {dswrf.mean():.6f}", file=regtest)
    print(f"var:  {dswrf.var():.6f}", file=regtest)
