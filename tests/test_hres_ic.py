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
import pytest
import zarr

from initial_conditions.hres_ic import fetch_hres_t0, surface_geopotential
from initial_conditions.run_hiccup_eamxx import RD

# isothermal hydrostatic atmosphere: z(p) = RD * T * ln(P_REF / p)
T_ISO = 250.0
P_REF = 100000.0
PLEV_HPA = np.array([500.0, 850.0, 1000.0])
LAT_SN = np.array([-60.0, -30.0, 0.0, 30.0, 60.0])
LON = np.arange(8) * 45.0
TIMES = ["2022-09-26T06", "2022-09-26T12"]


def _isothermal_geopotential(p_pa: np.ndarray, temp: float) -> np.ndarray:
    return RD * temp * np.log(P_REF / p_pa)


def test_surface_geopotential_is_exact_for_an_isothermal_atmosphere():
    plev_pa = PLEV_HPA * 100.0
    # inside the profile, on a level, and below the lowest level
    ps = np.array([[60000.0, 85000.0, 92000.0, 100000.0, 104000.0]], np.float32)
    z = _isothermal_geopotential(plev_pa, T_ISO)[:, None, None] * np.ones(
        (1, *ps.shape)
    )
    t_bottom = np.full(ps.shape, T_ISO, np.float32)
    phis = surface_geopotential(z.astype(np.float32), t_bottom, plev_pa, ps)
    np.testing.assert_allclose(
        phis, _isothermal_geopotential(ps.astype(np.float64), T_ISO), rtol=0, atol=0.1
    )


def _hours_since_2016(timestamp: str) -> int:
    return int(
        (np.datetime64(timestamp, "h") - np.datetime64("2016-01-01T00", "h"))
        / np.timedelta64(1, "h")
    )


def _write_wb2_like_store(path, omega_units: str = "Pa s**-1") -> None:
    """A tiny store in the WeatherBench2 hres_t0 layout (south-to-north, Zarr v2)."""
    nt, nlev, nlat, nlon = len(TIMES), PLEV_HPA.size, LAT_SN.size, LON.size
    root = zarr.open_group(str(path), mode="w", zarr_format=2)

    def put(name, data, units=""):
        arr = root.create_array(name, shape=data.shape, dtype=data.dtype)
        arr[...] = data
        arr.attrs["units"] = units

    put("time", np.array([_hours_since_2016(t) for t in TIMES], np.int64))
    put("latitude", LAT_SN)
    put("longitude", LON)
    put("level", PLEV_HPA)
    shape3 = (nt, nlev, nlat, nlon)
    shape2 = (nt, nlat, nlon)
    # temperature differs between the two times so the time lookup is visible
    temp = np.stack([np.full(shape3[1:], T_ISO + 10.0 * i) for i in range(nt)])
    put("temperature", temp.astype(np.float32), "K")
    # u carries the latitude so the north-to-south flip is visible
    put(
        "u_component_of_wind",
        np.broadcast_to(LAT_SN[None, None, :, None], shape3).astype(np.float32),
    )
    put("v_component_of_wind", np.zeros(shape3, np.float32))
    put("specific_humidity", np.zeros(shape3, np.float32))
    put("vertical_velocity", np.zeros(shape3, np.float32), omega_units)
    z = np.stack(
        [
            np.broadcast_to(
                _isothermal_geopotential(PLEV_HPA * 100.0, T_ISO + 10.0 * i)[
                    :, None, None
                ],
                shape3[1:],
            )
            for i in range(nt)
        ]
    )
    put("geopotential", z.astype(np.float32))
    ps = np.broadcast_to(np.linspace(90000.0, 103000.0, nlon)[None, None, :], shape2)
    put("surface_pressure", ps.astype(np.float32), "Pa")
    put("2m_temperature", np.full(shape2, 290.0, np.float32), "K")


def test_fetch_hres_t0_selects_the_time_and_flips_to_north_to_south(tmp_path):
    store = tmp_path / "hres_t0.zarr"
    _write_wb2_like_store(store)
    era5 = fetch_hres_t0("2022-09-26T12:00:00", str(store), workers=2)

    np.testing.assert_array_equal(era5.plev_pa, PLEV_HPA * 100.0)
    assert set(era5.atm) == {"T_mid", "qv", "U", "V", "omega"}
    # second time in the store
    np.testing.assert_array_equal(era5.atm["T_mid"], T_ISO + 10.0)
    # north-to-south: first row is 60N, last is 60S
    u = era5.atm["U"]
    assert u.shape == (PLEV_HPA.size, LAT_SN.size, LON.size)
    np.testing.assert_array_equal(u[:, 0, :], 60.0)
    np.testing.assert_array_equal(u[:, -1, :], -60.0)
    # phis is the isothermal geopotential at ps, inside and below the lowest level
    ps = era5.sfc["ps"].astype(np.float64)
    np.testing.assert_allclose(
        era5.sfc["phis"], _isothermal_geopotential(ps, T_ISO + 10.0), rtol=0, atol=0.1
    )


@pytest.mark.parametrize("timestamp", ["2022-09-26T12:30:00", "2022-09-26T12:00:05"])
def test_fetch_hres_t0_rejects_off_hour_timestamps(tmp_path, timestamp):
    store = tmp_path / "hres_t0.zarr"
    _write_wb2_like_store(store)
    with pytest.raises(SystemExit, match="not on the hour"):
        fetch_hres_t0(timestamp, str(store), workers=1)


def test_fetch_hres_t0_rejects_missing_time_and_hpa_omega(tmp_path):
    store = tmp_path / "hres_t0.zarr"
    _write_wb2_like_store(store)
    with pytest.raises(SystemExit, match="has no"):
        fetch_hres_t0("2022-09-27T00:00:00", str(store), workers=1)

    bad = tmp_path / "hres_t0_hpa.zarr"
    _write_wb2_like_store(bad, omega_units="hPa s-1")
    with pytest.raises(SystemExit, match="not Pa s-1"):
        fetch_hres_t0("2022-09-26T12:00:00", str(bad), workers=1)
