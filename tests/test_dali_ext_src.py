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

from screamcast.config import DataConfig, TrainConfig
from screamcast.dali_ext_src import ScreamV2, output_channels


def _make_channel_index_dataset():
    ds = object.__new__(ScreamV2)
    ds.variables_output = ("U", "ps", "phis", "qv", "T_2m")
    ds.level_start = 3
    ds.level_end = 11
    ds.plevel = 2
    return ds


def _flatten_output_channel_index(index):
    return [
        name if level is None or np.isnan(level) else f"{name}_{int(level)}"
        for name, level in index.tolist()
    ]


def test_channel_index_output_regression(regtest):
    standard = _make_channel_index_dataset()

    print("standard", file=regtest)
    for item in standard.channel_index_output().tolist():
        print(item, file=regtest)


def test_screamv2_configuration():
    """Test ScreamV2 with a typical variable configuration"""
    variables_prognostic = ("PotentialTemperature", "U", "V", "omega", "qv")
    variables_forcing = ("coszr", "sst", "phis")
    variables_diagnostic = ()

    # Test channel calculations
    in_channels = ScreamV2.num_of_input_channels(
        variables_prognostic, variables_forcing, 4, 3, 128
    )
    out_channels = ScreamV2.num_of_output_channels(
        variables_prognostic, variables_diagnostic, 4, 3, 128
    )

    assert in_channels == 163
    assert out_channels == 160

    # Test ranges
    ranges_input = ScreamV2.ranges_input(
        variables_prognostic, variables_forcing, 4, 3, 128
    )
    ranges_output = ScreamV2.ranges_output(
        variables_prognostic, variables_diagnostic, 4, 3, 128
    )
    assert len(ranges_input) == 8
    assert len(ranges_output) == 5
    assert ranges_input["PotentialTemperature"] == slice(0, 32)
    assert ranges_output["U"] == slice(32, 64)

    """Test different plevel configurations"""
    variables_prognostic = ("PotentialTemperature",)  # 3D variable
    variables_forcing = ("coszr",)  # 2D variable

    # Test different plevel, level_start, level_end configurations
    for plevel in [2, 4, 8]:
        for level_start, level_end in [(3, 128), (10, 100)]:
            channels = ScreamV2.num_of_input_channels(
                variables_prognostic, variables_forcing, plevel, level_start, level_end
            )

            expected_3d_levels = len(np.r_[level_start:level_end:plevel])
            expected_channels = expected_3d_levels + 1  # 3D var + 2D var

            assert channels == expected_channels


def test_output_channel_names_from_train_config():
    cfg = TrainConfig(
        data=DataConfig(
            variables_prognostic=(
                "PotentialTemperature",
                "U",
                "V",
                "z_mid",
                "omega",
                "qv",
                "T_2m",
            ),
            variables_diagnostic=(
                "precip_ice_surf_mass_flux",
                "precip_liq_surf_mass_flux",
            ),
            level_start=3,
            level_end=128,
            plevel=4,
        )
    )

    names = output_channels(cfg)

    assert len(names) == 195
    assert names[:4] == [
        "PotentialTemperature_3",
        "PotentialTemperature_7",
        "PotentialTemperature_11",
        "PotentialTemperature_15",
    ]
    assert names[-3:] == [
        "T_2m",
        "precip_ice_surf_mass_flux",
        "precip_liq_surf_mass_flux",
    ]


def test_output_channels_matches_channel_index_output():
    cfg = TrainConfig(
        data=DataConfig(
            variables_prognostic=("U", "qv", "T_2m"),
            variables_diagnostic=("precip_liq_surf_mass_flux",),
            level_start=3,
            level_end=11,
            plevel=2,
        )
    )
    ds = object.__new__(ScreamV2)
    ds.variables_prognostic = cfg.data.variables_prognostic
    ds.variables_diagnostic = cfg.data.variables_diagnostic
    ds.variables_output = cfg.data.variables_prognostic + cfg.data.variables_diagnostic
    ds.level_start = cfg.data.level_start
    ds.level_end = cfg.data.level_end
    ds.plevel = cfg.data.plevel

    index = ds.channel_index_output()
    expected = _flatten_output_channel_index(index)

    assert output_channels(cfg) == expected
