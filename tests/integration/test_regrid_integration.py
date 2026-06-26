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
import xarray

from screamcast.catalog_core import Grids

regrid = __import__("scripts.ace.regrid_zarr", fromlist=["regrid"]).regrid


class MockInput:
    def to_xarray(self, **kwargs):
        ds = xarray.Dataset(
            data_vars={
                "T_2m": (
                    ("time", "cell"),
                    np.linspace(280.0, 291.0, 12, dtype=np.float32).reshape(1, 12),
                ),
                "U": (
                    ("time", "level", "cell"),
                    np.linspace(0.0, 11.0, 12, dtype=np.float32).reshape(1, 1, 12),
                ),
                "landfrac": (
                    ("cell",),
                    np.linspace(0.0, 1.0, 12, dtype=np.float32),
                ),
            },
            coords={
                "time": np.array(["2000-01-01T00:00:00"], dtype="datetime64[ns]"),
                "level": np.array([1000.0], dtype=np.float32),
                "lat": ("cell", np.linspace(-80.0, 80.0, 12, dtype=np.float32)),
                "lon": ("cell", np.linspace(0.0, 330.0, 12, dtype=np.float32)),
                "hyam": ("level", np.array([0.1], dtype=np.float32)),
                "hybm": ("level", np.array([0.9], dtype=np.float32)),
            },
            attrs={"grid": Grids.ne1024pg2, "history": "existing history"},
        )
        ds["T_2m"].attrs["units"] = "K"
        ds["U"].attrs["units"] = "m s-1"
        ds["hyam"].attrs["source_location"] = "mock_vertical.nc"
        return ds


def test_regrid_writes_xarray_readable_zarr(tmp_path):
    output = tmp_path / "out.zarr"

    regrid(
        MockInput(),
        output=str(output),
        selection={"time": slice(0, 1), "level": slice(0, 1)},
        device="cpu",
    )

    ds = xarray.open_zarr(output)

    assert ds["T_2m"].dims == ("time", "lat", "lon")
    assert ds["U"].dims == ("time", "level", "lat", "lon")
    assert ds["landfrac"].dims == ("lat", "lon")
    assert ds["hyam"].attrs["source_location"] == "mock_vertical.nc"
    assert ds["T_2m"].attrs["units"] == "K"
    assert "existing history" in ds.attrs["history"]
    assert "lat" in ds.coords
    assert "lon" in ds.coords
