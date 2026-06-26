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
import pandas as pd

from screamcast.inference import _get_nearest


def test__get_nearest():
    nan = np.float64("NaN")

    index = pd.MultiIndex.from_tuples(
        [
            ("PotentialTemperature", 3.0),
            ("PotentialTemperature", 41.0),
            ("PotentialTemperature", 79.0),
            ("PotentialTemperature", 117.0),
            ("T_2m", nan),
            ("U", 3.0),
            ("U", 41.0),
            ("U", 79.0),
            ("U", 117.0),
            ("V", 3.0),
            ("V", 41.0),
            ("V", 79.0),
            ("V", 117.0),
            ("geopotential_mid", 3.0),
            ("geopotential_mid", 41.0),
            ("geopotential_mid", 79.0),
            ("geopotential_mid", 117.0),
            ("omega", 3.0),
            ("omega", 41.0),
            ("omega", 79.0),
            ("omega", 117.0),
            ("qv", 3.0),
            ("qv", 41.0),
            ("qv", 79.0),
            ("qv", 117.0),
            ("precip_ice_surf_mass_flux", nan),
            ("precip_liq_surf_mass_flux", nan),
        ],
        names=["name", "level"],
    )

    assert ("qv", 79.0) == _get_nearest(index, ("qv", 78))
    assert (
        "precip_liq_surf_mass_flux"
        == _get_nearest(index, ("precip_liq_surf_mass_flux", nan))[0]
    )
