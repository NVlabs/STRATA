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
import xarray
import zarr

from screamcast.catalog_core import ScreamData, open_zarr_with_dimension_names


def test_open_zarr_with_dimension_names_reads_back_data(tmp_path):
    path = tmp_path / "catalog.zarr"
    ds = xarray.Dataset(
        data_vars={
            "T_2m": (
                ("time", "cell"),
                np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            ),
            "U": (
                ("time", "level", "cell"),
                np.array([[[10.0, 11.0]], [[12.0, 13.0]]], dtype=np.float32),
            ),
        }
    )
    ds.to_zarr(path, zarr_format=3)
    zarr.consolidate_metadata(path)

    group = zarr.open_consolidated(path)
    opened = open_zarr_with_dimension_names(group, chunks=None)

    np.testing.assert_array_equal(opened["T_2m"].values, ds["T_2m"].values)
    np.testing.assert_array_equal(opened["U"].values, ds["U"].values)
    assert opened["T_2m"].dims == ("time", "cell")
    assert opened["U"].dims == ("time", "level", "cell")


def test_open_zarr_with_dimension_names_rejects_unknown_rank(tmp_path):
    path = tmp_path / "catalog_rank4.zarr"
    ds = xarray.Dataset(
        data_vars={
            "bad": (
                ("sample", "time", "level", "cell"),
                np.zeros((1, 2, 3, 4), dtype=np.float32),
            )
        }
    )
    ds.to_zarr(path, zarr_format=3)
    zarr.consolidate_metadata(path)

    group = zarr.open_consolidated(path)
    with pytest.raises(ValueError, match="Unsupported array rank"):
        open_zarr_with_dimension_names(group, chunks=None)


def test_scream_data_time_property(tmp_path):
    path = tmp_path / "catalog_time.zarr"
    ds = xarray.Dataset(
        data_vars={
            "T_2m": (
                ("time", "cell"),
                np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
            ),
        }
    )
    ds.to_zarr(path, zarr_format=3)
    zarr.consolidate_metadata(path)
    (tmp_path / "grid.nc").touch()
    (tmp_path / "vertical.nc").touch()

    data = ScreamData(
        path=str(path),
        profile="mock",
        vertical_subset=slice(0, None),
        grid_file=str(tmp_path / "grid.nc"),
        vertical_coordinate_file=str(tmp_path / "vertical.nc"),
        reference_time=np.datetime64("2020-10-01T00:00:00"),
        native_timestep=np.timedelta64(10, "m"),
    )

    expected = np.array(
        ["2020-10-01T00:00:00", "2020-10-01T00:10:00", "2020-10-01T00:20:00"],
        dtype="datetime64[s]",
    )
    np.testing.assert_array_equal(data.time, expected)
