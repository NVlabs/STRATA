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
"""Reading a one-frame IC store (``screamcast/initial_conditions/``) as the rollout does."""
import dataclasses

import numpy as np
import pytest
import zarr

import data_catalog

NCOL = 6 * 1024 * 1024 * 4
CHUNK = 1 << 16
LEVELS = 32


def _ic_store(path, initial_time):
    """An IC-store skeleton: full ne1024pg2 shapes, only the first chunk written.

    3D variables hold ``100 * k + level`` at every column, 2D ones ``k``, where
    ``k`` numbers the variable, so a wrong variable, level or frame shows.
    """
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    root.attrs.update(grid="ne1024pg2", initial_time=initial_time)
    for k, name in enumerate(("U", "PotentialTemperature"), start=1):
        a = root.create_array(
            name,
            shape=(1, LEVELS, NCOL),
            chunks=(1, 1, CHUNK),
            dtype="float32",
            fill_value=np.nan,
        )
        for lev in range(LEVELS):
            a[0, lev, :CHUNK] = 100 * k + lev
    for k, name in enumerate(("T_2m", "ps"), start=1):
        a = root.create_array(
            name, shape=(1, NCOL), chunks=(1, CHUNK), dtype="float32", fill_value=np.nan
        )
        a[0, :CHUNK] = k
    zarr.consolidate_metadata(str(path))
    return str(path)


def _request(time):
    # an 8x8 tile in the corner of face 0 lies in the first CHUNK columns
    return {
        "time": np.array([time]),
        "variable": ["U_0", "U_31", "PotentialTemperature_7", "T_2m", "ps"],
        "face": np.array([0]),
        "x": np.arange(8),
        "y": np.arange(8),
    }


def test_ic_store_frame_is_at_its_initial_time(tmp_path):
    path = _ic_store(tmp_path / "ic.zarr", "2017-09-14T00:00:00")
    entry = data_catalog.ic_store(path)
    assert entry.reference_time == np.datetime64("2017-09-14T00:00:00")

    out = entry.to_data_source()(_request(np.datetime64("2017-09-14T00:00")))
    assert out.shape == (1, 5, 1, 8, 8)
    expected = [100, 131, 207, 1, 2]
    for v, want in enumerate(expected):
        np.testing.assert_array_equal(out.values[0, v], want)


def test_ic_store_has_no_later_frame(tmp_path):
    path = _ic_store(tmp_path / "ic.zarr", "2017-09-14T00:00:00")
    source = data_catalog.ic_store(path).to_data_source()
    with pytest.raises(IndexError):
        source(_request(np.datetime64("2017-09-14T00:10")))


def test_catalog_reference_time_reaches_the_reader(tmp_path):
    """Frame ``n`` of a catalog entry is ``reference_time + n * 10 min``."""
    root = zarr.open_group(str(tmp_path / "two.zarr"), mode="w", zarr_format=3)
    a = root.create_array(
        "T_2m", shape=(3, NCOL), chunks=(1, CHUNK), dtype="float32", fill_value=np.nan
    )
    for t in range(3):
        a[t, :CHUNK] = t
    zarr.consolidate_metadata(str(tmp_path / "two.zarr"))
    entry = dataclasses.replace(
        data_catalog.scream_sdecadal,
        path=str(tmp_path / "two.zarr"),
        profile="",
        reference_time=np.datetime64("2001-02-03T04:00:00"),
    )
    request = _request(np.datetime64("2001-02-03T04:20"))
    request["variable"] = ["T_2m"]
    np.testing.assert_array_equal(entry.to_data_source()(request).values, 2)
