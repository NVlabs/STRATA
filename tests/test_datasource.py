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
"""Smoke test for ScreamDataSource earth2studio wrapper.

Run via:
    pytest tests/test_datasource.py

"""
import os

import dotenv
import numpy as np
import pytest

if "PROJECT_ROOT" not in os.environ:
    pytest.skip("PROJECT_ROOT is not configured", allow_module_level=True)

from screamcast.dali_ext_src import open_consolidated_zarr_group
from screamcast.earth2studio_wrappers import ScreamDataSource

dotenv.load_dotenv(".")


def _open_dataset():

    storage_env = os.getenv("SCREAM_ZARR_PROFILE", "")
    try:
        path = os.environ["SCREAM_MAIN_ZARR_PATH"]
    except KeyError:
        pytest.skip("Dataset not configured. Set the SCREAM_MAIN_ZARR_PATH env var")

    main_group = open_consolidated_zarr_group(path, storage_env=storage_env)
    var_group = {key: main_group for key in main_group.keys()}
    return ScreamDataSource(var_group)


def test_scream_datasource():

    ds = _open_dataset()

    # --------------------------------------------------------------------------
    # Build coords: small tile on face 0
    # --------------------------------------------------------------------------
    tile_size = 256
    t0_np = np.datetime64("2020-10-13T00:00:00")

    # Request a 3D variable at two levels and a 2D variable
    variables = [
        "U_31",
        "PotentialTemperature_8",
        "PotentialTemperature_9",
        "T_2m",
        "coszr",
    ]

    coords = {
        "time": np.array([t0_np]),
        "variable": variables,
        "face": np.array([0]),
        "x": np.arange(tile_size),
        "y": np.arange(tile_size),
    }

    print(
        f"Requesting: time={t0_np}, variables={variables}, face=0, tile={tile_size}x{tile_size}"
    )

    out = ds(coords)

    # --------------------------------------------------------------------------
    # Assertions
    # --------------------------------------------------------------------------
    assert out.dims == ("time", "variable", "face", "x", "y"), f"Wrong dims: {out.dims}"
    assert out.shape == (
        1,
        len(variables),
        1,
        tile_size,
        tile_size,
    ), f"Wrong shape: {out.shape}"
    assert out.coords["face"].values.tolist() == [0]
    assert out.coords["variable"].values.tolist() == variables

    arr = out.values
    assert np.isfinite(arr).all(), f"Output contains non-finite values: {arr}"

    print(f"Output shape: {out.shape}")
    for i, v in enumerate(variables):
        d = arr[0, i, 0]
        print(f"  {v:30s}  min={d.min():.4f}  max={d.max():.4f}  mean={d.mean():.4f}")

    print("datasource: PASS")
