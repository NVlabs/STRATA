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
from __future__ import annotations

import numpy as np
import xarray as xr
import zarr

from initial_conditions.hiccup_l128_to_zarr import validate_zarr, write_zarr_from_netcdf
from initial_conditions.layout import NLEV, NLEV_SUBSET, VARS_3D
from initial_conditions.run_hiccup_eamxx import GRAV, RD, theta_and_zmid

P0 = 1.0e5


def _sigma_coords(nlev: int):
    """Pure-sigma interfaces from 0.05*ps (top) to ps (bottom)."""
    hyai = np.zeros(nlev + 1)
    hybi = np.linspace(0.05, 1.0, nlev + 1)
    return 0.5 * (hyai[:-1] + hyai[1:]), 0.5 * (hybi[:-1] + hybi[1:]), hyai, hybi


def test_isothermal_dry_zmid_matches_hypsometric_closed_form():
    nlev, ncol = 6, 3
    hyam, hybm, hyai, hybi = _sigma_coords(nlev)
    temp = 250.0
    t = np.full((nlev, ncol), temp)
    qv = np.zeros((nlev, ncol))
    ps = np.array([1.0e5, 9.0e4, 7.0e4])
    phis = np.array([0.0, 5.0e3, 2.0e4])

    _, z_mid = theta_and_zmid(t, qv, ps, phis, hyam, hybm, hyai, hybi, P0)

    p_mid = hybm[:, None] * ps[None, :]
    expected = phis / GRAV + (RD * temp / GRAV) * np.log(ps / p_mid)
    np.testing.assert_allclose(z_mid, expected, rtol=1e-12, atol=1e-9)


def test_theta_equals_t_at_reference_pressure_and_exceeds_it_above():
    hyam, hybm = np.array([0.0]), np.array([1.0])
    hyai, hybi = np.array([0.0, 0.0]), np.array([0.5, 1.0])
    t = np.array([[250.0, 300.0]])
    zeros = np.zeros(2)

    theta, _ = theta_and_zmid(
        t, np.zeros_like(t), np.full(2, P0), zeros, hyam, hybm, hyai, hybi, P0
    )
    np.testing.assert_array_equal(theta, t)

    theta, _ = theta_and_zmid(
        t, np.zeros_like(t), np.full(2, 0.5 * P0), zeros, hyam, hybm, hyai, hybi, P0
    )
    assert np.all(theta > t)


def test_moisture_raises_every_mid_level_and_heights_increase_upward():
    nlev, ncol = 8, 4
    hyam, hybm, hyai, hybi = _sigma_coords(nlev)
    rng = np.random.default_rng(0)
    t = rng.uniform(200.0, 300.0, size=(nlev, ncol))
    ps = rng.uniform(6.0e4, 1.05e5, size=ncol)
    phis = rng.uniform(0.0, 3.0e4, size=ncol)
    dry = np.zeros((nlev, ncol))
    moist = np.full((nlev, ncol), 0.01)

    _, z_dry = theta_and_zmid(t, dry, ps, phis, hyam, hybm, hyai, hybi, P0)
    _, z_moist = theta_and_zmid(t, moist, ps, phis, hyam, hybm, hyai, hybi, P0)

    assert np.all(z_moist > z_dry)
    assert np.all(np.diff(z_dry, axis=0) < 0)
    assert np.all(z_dry[-1] > phis / GRAV)


def _write_synthetic_inputs(tmp_path, ncol: int):
    lev = np.arange(NLEV)
    col = np.arange(ncol)
    data_vars = {}
    for i, name in enumerate(VARS_3D):
        values = 1.0e6 * (i + 1) + 1000.0 * lev[:, None] + col[None, :]
        data_vars[name] = (("time", "lev", "ncol"), values[None].astype(np.float32))
    data_vars["T_2m"] = (("time", "ncol"), (280.0 + col)[None].astype(np.float32))
    data_vars["ps"] = (("time", "ncol"), (9.0e4 + col)[None].astype(np.float32))
    data_vars["phis"] = (("time", "ncol"), (10.0 * col)[None].astype(np.float32))
    ic_path = tmp_path / "ic.nc"
    xr.Dataset(data_vars, attrs={"history": "upstream hiccup step"}).to_netcdf(ic_path)

    lat_deg = np.array([0.0, 0.0, 45.0, -45.0, 89.0, -89.0])[:ncol]
    lon_deg = np.array([0.0, 180.0, 90.0, 270.0, 10.0, 200.0])[:ncol]
    latlon_path = tmp_path / "latlon.nc"
    xr.Dataset(
        {
            "lat": ("ncol", np.radians(lat_deg), {"units": "radians"}),
            "lon": ("ncol", np.radians(lon_deg), {"units": "radians"}),
        }
    ).to_netcdf(latlon_path)

    ref = tmp_path / "scream_static"
    ref.mkdir()
    statics = {"sgh30": 3.0 * col + 1.0, "landfrac": col / ncol}
    for name, values in statics.items():
        zarr.create_array(
            store=str(ref / f"{name}.zarr"), data=values.astype(np.float32)
        )
    return ic_path, latlon_path, ref, statics


def test_netcdf_to_zarr_subsets_levels_without_transposing(tmp_path):
    ncol = 6
    timestamp = "2016-03-20T12:00:00"
    ic_path, latlon_path, ref, statics = _write_synthetic_inputs(tmp_path, ncol)
    out = tmp_path / "ic.zarr"

    write_zarr_from_netcdf(
        str(ic_path),
        str(out),
        timestamp,
        str(latlon_path),
        str(ref),
        workers=1,
        argv=["convert", "--flag"],
    )
    validate_zarr(str(out), timestamp)

    root = zarr.open_consolidated(str(out))
    col = np.arange(ncol)
    for i, name in enumerate(VARS_3D):
        assert root[name].shape == (1, NLEV_SUBSET, ncol)
        for j in range(NLEV_SUBSET):
            expected = 1.0e6 * (i + 1) + 1000.0 * (3 + 4 * j) + col
            np.testing.assert_array_equal(root[name][0, j, :], expected)
    np.testing.assert_array_equal(root["T_2m"][0, :], 280.0 + col)
    np.testing.assert_array_equal(root["ps"][0, :], 9.0e4 + col)
    np.testing.assert_array_equal(root["phis"][:], 10.0 * col)
    for name, values in statics.items():
        np.testing.assert_array_equal(root[name][:], values.astype(np.float32))

    # lon 0 is local noon and lon 180 local midnight at 12Z on the equinox;
    # SCREAM's coszr is clipped to 0 on the night side.
    coszr = root["coszr"][0, :]
    assert coszr[0] > 0.99
    assert coszr[1] == 0.0
    assert np.all((coszr >= 0.0) & (coszr <= 1.0))

    assert root.attrs["level_subset"] == "128[3::4]"
    history = root.attrs["history"].splitlines()
    assert history[0] == "upstream hiccup step"
    assert history[-1].endswith("convert --flag")
