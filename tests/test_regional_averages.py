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
"""Unit tests for screamcast.regional_averages."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pytest
import torch
import xarray as xr

from screamcast.distributed_halo import TileTopology
from screamcast.regional_averages import (
    REGIONS,
    RegionalAverager,
    _build_region_weights,
)


def test_region_weights_partition_globe():
    """land + ocean == global and tropics + extratropics == global; masks clean."""
    torch.manual_seed(0)
    nside = 4
    area = torch.rand(6, nside, nside) + 0.1  # strictly positive
    # Lat spanning the full range including ±30
    lat = torch.linspace(-90.0, 90.0, 6 * nside * nside).reshape(6, nside, nside)
    landfrac = torch.rand(6, nside, nside)

    weights = _build_region_weights(
        area=area,
        lat_deg=lat,
        landfrac=landfrac,
        tropics_lat_cutoff_deg=30.0,
    )

    torch.testing.assert_close(
        weights["land"] + weights["ocean"], weights["global"], rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        weights["tropics"] + weights["extratropics"],
        weights["global"],
        rtol=1e-5,
        atol=1e-6,
    )
    abs_lat = lat.abs()
    assert torch.all(weights["tropics"][abs_lat >= 30.0] == 0.0)
    assert torch.all(weights["extratropics"][abs_lat < 30.0] == 0.0)


def _make_averager(tmp_path, variable_names, grouped, n_steps=2, output_path=None):
    """Build a single-rank averager over a 6-face, 4x4 synthetic grid."""
    topology = TileTopology(
        world_size=1,
        rank=0,
        face_size=4,
        tile_size=4,
        halo_width=0,
        n_faces=6,
    )
    # Set half the cells to tropics by construction: lat alternates below/above 30
    lat_face = torch.zeros(6, 4, 4)
    lat_face[:, :2, :] = 10.0  # tropics
    lat_face[:, 2:, :] = 60.0  # extratropics
    area_face = torch.ones(6, 4, 4)
    landfrac_face = torch.full((6, 4, 4), 0.25)

    hyam = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    hybm = np.array([1.0, 0.5, 0.0], dtype=np.float32)

    if output_path is None:
        output_path = str(tmp_path / "averages.nc")

    return RegionalAverager.from_topology(
        topology=topology,
        lat_face_deg=lat_face,
        landfrac_face=landfrac_face,
        area_face=area_face,
        variable_names=variable_names,
        grouped_variables=grouped,
        output_path=output_path,
        t0=np.datetime64("2020-10-13T00:00:00"),
        dt=np.timedelta64(600, "s"),
        n_steps=n_steps,
        device=torch.device("cpu"),
        tropics_lat_cutoff_deg=30.0,
        hyam=hyam,
        hybm=hybm,
        attrs={"source_test": True, "run_name": "unit_test"},
    )


def test_constant_field_gives_constant_mean(tmp_path):
    variable_names = ["ps", "T_0", "T_1", "T_2"]
    grouped = OrderedDict(
        [
            ("ps", [(None, 0)]),
            ("T", [(0, 1), (1, 2), (2, 3)]),
        ]
    )
    averager = _make_averager(tmp_path, variable_names, grouped, n_steps=1)
    try:
        out_tensor = torch.full((1, 4, 6, 4, 4), 3.14)
        means = averager(0, out_tensor)
        assert list(means.keys()) == list(REGIONS)
        for per_channel in means.values():
            for value in per_channel.values():
                assert value == pytest.approx(3.14, abs=1e-5)
    finally:
        averager.close()


def test_netcdf_schema_and_values(tmp_path):
    variable_names = ["ps", "T_0", "T_1", "T_2"]
    grouped = OrderedDict(
        [
            ("ps", [(None, 0)]),
            ("T", [(0, 1), (1, 2), (2, 3)]),
        ]
    )
    n_steps = 3
    nc_path = tmp_path / "averages.nc"
    averager = _make_averager(
        tmp_path, variable_names, grouped, n_steps=n_steps, output_path=str(nc_path)
    )
    try:
        torch.manual_seed(1)
        for s in range(n_steps):
            out_tensor = torch.randn(1, 4, 6, 4, 4)
            averager(s, out_tensor)
    finally:
        averager.close()

    with xr.open_dataset(nc_path, engine="h5netcdf") as ds:
        assert ds.sizes["time"] == 1
        assert ds.sizes["step"] == n_steps
        assert ds.sizes["region"] == len(REGIONS)
        assert ds.sizes["level"] == 3
        assert ds["ps"].dims == ("time", "step", "region")
        assert ds["T"].dims == ("time", "step", "region", "level")
        assert list(ds["region"].values) == list(REGIONS)
        assert "tropics_lat_cutoff_deg" in ds.attrs
        assert ds.attrs["tropics_lat_cutoff_deg"] == 30.0
        # No NaNs left from preallocation
        assert np.all(np.isfinite(ds["ps"].values))
        assert np.all(np.isfinite(ds["T"].values))


def test_write_step_out_of_range_raises(tmp_path):
    variable_names = ["ps"]
    grouped = OrderedDict([("ps", [(None, 0)])])
    averager = _make_averager(tmp_path, variable_names, grouped, n_steps=1)
    try:
        out_tensor = torch.zeros(1, 1, 6, 4, 4)
        with pytest.raises(IndexError):
            averager(1, out_tensor)
    finally:
        averager.close()
