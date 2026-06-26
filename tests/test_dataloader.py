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
import os
from types import SimpleNamespace

import pytest

from screamcast.dali_ext_src import ScreamV2
from screamcast.pipelines import get_dataloader

# Use the HEALPix dataset for tests in this module.
HEALPIX_MAIN_ZARR_PATH = "s3://SCREAM_zarrv3/sdecadal.ne1024pg2_ne1024pg2.F20TR-SCREAMv1.c10-sep11.out10min.zarr"
HEALPIX_AUX_ZARR_PATH = "s3://SCREAM_zarrv3/sdecadal.ne1024pg2_ne1024pg2.F20TR-SCREAMv1.c10-sep11.out10min.zarr"

# Skip tests that require S3/rclone access when running in CI or when explicitly requested
skip_s3 = (
    os.getenv("GITLAB_CI") == "true"
    or os.getenv("CI") == "true"
    or os.getenv("SCREAM_SKIP_S3_TESTS") in {"1", "true", "True"}
)
pytestmark = pytest.mark.skipif(
    skip_s3, reason="S3/rclone not available in CI environment"
)


@pytest.fixture(autouse=True)
def _set_healpix_zarr_paths(monkeypatch):
    # Make the data source explicit so these tests don't depend on whatever happens
    # to be in the user's environment.
    monkeypatch.setenv("SCREAM_MAIN_ZARR_PATH", HEALPIX_MAIN_ZARR_PATH)
    monkeypatch.setenv("SCREAM_AUX_ZARR_PATH", HEALPIX_AUX_ZARR_PATH)


def test_dataloader_split_by_time():
    # test train/test split
    ds = ScreamV2(
        batch_size=1,
        plevel=1,
        level_start=3,
        level_end=5,
        split_by_time=True,
        train_start_index=3,
        train_end_index=8,
        test_start_index=10,
        test_end_index=14,
        test_stride=2,
        split="train",
        grid_type="healpix",
    )

    # Compute expected values from actual dataset properties
    time_len = max(1, ds.nt - ds.num_steps - 1)
    npatches = ds.npatches
    train_times = [3, 4, 5, 6, 7]  # range(3, 8)
    test_times = [10, 12]  # range(10, 14, 2)

    # Filter to valid time range [1, time_len]
    valid_train_times = [t for t in train_times if 1 <= t <= time_len]
    valid_test_times = [t for t in test_times if 1 <= t <= time_len]

    # Verify total count: num_times × num_patches
    assert len(ds.index) == len(valid_train_times) * npatches

    # Verify the index formula: j = patch * time_len + (t - 1)
    expected_train_indices = sorted(
        [
            patch * time_len + (t - 1)
            for t in valid_train_times
            for patch in range(npatches)
        ]
    )
    assert (
        ds.index == expected_train_indices
    ), f"Train index mismatch:\n  got:      {ds.index[:10]}...\n  expected: {expected_train_indices[:10]}..."

    # Verify first and last indices explicitly
    assert ds.index[0] == 0 * time_len + (valid_train_times[0] - 1)
    assert ds.index[-1] == (npatches - 1) * time_len + (valid_train_times[-1] - 1)

    # test test split with stride
    ds_test = ScreamV2(
        batch_size=1,
        plevel=1,
        level_start=3,
        level_end=5,
        split_by_time=True,
        train_start_index=3,
        train_end_index=8,
        test_start_index=10,
        test_end_index=14,
        test_stride=2,
        split="test",
        grid_type="healpix",
    )

    # Verify test split count and formula
    assert len(ds_test.index) == len(valid_test_times) * npatches

    expected_test_indices = [
        patch * time_len + (t - 1)
        for t in valid_test_times
        for patch in range(npatches)
    ]
    assert (
        ds_test.index == expected_test_indices
    ), f"Test index mismatch:\n  got:      {ds_test.index[:10]}...\n  expected: {expected_test_indices[:10]}..."


def test_dataloader():
    variables_prognostic = (
        "PotentialTemperature",
        "U",
        "V",
        "omega",
        "qv",
        "T_2m",
        "precip_ice_surf_mass_flux",
        "precip_liq_surf_mass_flux",
    )
    variables_forcing = ("coszr", "phis")
    variables_diagnostic = ()
    dataloader = get_dataloader(
        global_rank=0,
        world_size=1,
        device_id=0,
        batch_size=1,
        num_workers=1,
        split="train",
        mock=True,
        plevel=4,
        level_start=3,
        level_end=128,
        variables_prognostic=variables_prognostic,
        variables_forcing=variables_forcing,
        variables_diagnostic=variables_diagnostic,
    )
    inp, tar, index, j = next(iter(dataloader))
    num_of_channels_output = ScreamV2.num_of_output_channels(
        variables_prognostic=variables_prognostic,
        variables_diagnostic=variables_diagnostic,
        plevel=4,
        level_start=3,
        level_end=128,
    )
    num_of_channels_input = ScreamV2.num_of_input_channels(
        variables_prognostic=variables_prognostic,
        variables_forcing=variables_forcing,
        plevel=4,
        level_start=3,
        level_end=128,
    )
    assert num_of_channels_output == 163
    assert num_of_channels_input == 165
    assert inp.shape == (1, num_of_channels_input, 256, 256)
    assert tar.shape == (1, num_of_channels_output, 256, 256)


def test_dataloader_multistep():
    variables_prognostic = (
        "PotentialTemperature",
        "U",
        "V",
        "omega",
        "qv",
        "T_2m",
        "precip_ice_surf_mass_flux",
        "precip_liq_surf_mass_flux",
    )
    variables_forcing = ("coszr", "phis")
    variables_diagnostic = ()
    num_steps = 3
    dataloader = get_dataloader(
        global_rank=0,
        world_size=1,
        device_id=0,
        batch_size=1,
        num_workers=1,
        split="train",
        mock=True,
        plevel=4,
        level_start=3,
        level_end=128,
        variables_prognostic=variables_prognostic,
        variables_forcing=variables_forcing,
        variables_diagnostic=variables_diagnostic,
        num_steps=num_steps,
        load_final_target_only=True,
    )
    inp, tar, _, _, s_forcings = next(iter(dataloader))
    num_of_channels_output = ScreamV2.num_of_output_channels(
        variables_prognostic=variables_prognostic,
        variables_diagnostic=variables_diagnostic,
        plevel=4,
        level_start=3,
        level_end=128,
    )
    num_of_channels_input = ScreamV2.num_of_input_channels(
        variables_prognostic=variables_prognostic,
        variables_forcing=variables_forcing,
        plevel=4,
        level_start=3,
        level_end=128,
    )
    assert num_of_channels_output == 163
    assert num_of_channels_input == 165
    assert inp.shape == (1, num_of_channels_input, 256, 256)
    assert tar.shape == (1, 1, num_of_channels_output, 256, 256)
    assert s_forcings.shape == (1, num_steps - 1, 2, 256, 256)

    # test multistep loading with load_final_target_only=False
    dataloader = get_dataloader(
        global_rank=0,
        world_size=1,
        device_id=0,
        batch_size=1,
        num_workers=1,
        split="train",
        mock=True,
        plevel=4,
        level_start=3,
        level_end=128,
        variables_prognostic=variables_prognostic,
        variables_forcing=variables_forcing,
        variables_diagnostic=variables_diagnostic,
        num_steps=num_steps,
        load_final_target_only=False,
    )
    inp, tar, _, _, s_forcings = next(iter(dataloader))
    num_of_channels_output = ScreamV2.num_of_output_channels(
        variables_prognostic=variables_prognostic,
        variables_diagnostic=variables_diagnostic,
        plevel=4,
        level_start=3,
        level_end=128,
    )
    num_of_channels_input = ScreamV2.num_of_input_channels(
        variables_prognostic=variables_prognostic,
        variables_forcing=variables_forcing,
        plevel=4,
        level_start=3,
        level_end=128,
    )
    assert num_of_channels_output == 163
    assert num_of_channels_input == 165
    assert inp.shape == (1, num_of_channels_input, 256, 256)
    assert tar.shape == (1, num_steps, num_of_channels_output, 256, 256)
    assert s_forcings.shape == (1, num_steps - 1, 2, 256, 256)


def test_dataloader_mock():
    variables_prognostic = (
        "PotentialTemperature",
        "U",
        "V",
        "omega",
        "qv",
        "T_2m",
        "precip_ice_surf_mass_flux",
        "precip_liq_surf_mass_flux",
    )
    variables_forcing = ("coszr", "phis")
    variables_diagnostic = ()
    dataloader = get_dataloader(
        global_rank=0,
        world_size=1,
        device_id=0,
        batch_size=1,
        num_workers=1,
        split="train",
        mock=True,
        plevel=4,
        level_start=3,
        level_end=128,
        variables_prognostic=variables_prognostic,
        variables_forcing=variables_forcing,
        variables_diagnostic=variables_diagnostic,
        split_by_time=False,
    )
    inp, tar, index, j = next(iter(dataloader))
    num_of_channels_output = ScreamV2.num_of_output_channels(
        variables_prognostic=variables_prognostic,
        variables_diagnostic=variables_diagnostic,
        plevel=4,
        level_start=3,
        level_end=128,
    )
    num_of_channels_input = ScreamV2.num_of_input_channels(
        variables_prognostic=variables_prognostic,
        variables_forcing=variables_forcing,
        plevel=4,
        level_start=3,
        level_end=128,
    )
    assert num_of_channels_output == 163
    assert num_of_channels_input == 165
    assert inp.shape == (1, num_of_channels_input, 256, 256)
    assert tar.shape == (1, num_of_channels_output, 256, 256)


def test_datasrc():
    batch_info = SimpleNamespace(iteration=10, epoch_idx=2)

    ext_src = ScreamV2(batch_size=1, mock=True, num_steps=1)

    assert len(ext_src(batch_info)) == 4

    ext_src = ScreamV2(batch_size=1, mock=True, num_steps=2)
    assert len(ext_src(batch_info)) == 5
