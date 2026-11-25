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

from screamcast.initial_conditions.fast_ic import RD
from screamcast.initial_conditions.hres_ic import (
    check_wb2_init,
    fetch_hres_forecast,
    fetch_hres_t0,
    surface_geopotential,
)

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


def _temperature(i: int, lead: int) -> float:
    """Temperature of time ``i`` at ``lead``: every (time, lead) differs."""
    return T_ISO + 10.0 * i + 0.5 * lead


def _write_wb2_like_store(
    path,
    omega_units: str = "Pa s**-1",
    time_units: str = "hours since 2016-01-01",
    dims_3d: tuple[str, ...] = ("time", "level", "latitude", "longitude"),
    leads: list[int] | None = None,
    lead_units: str = "hours",
) -> None:
    """A tiny store in the WeatherBench2 hres_t0 layout (south-to-north, Zarr v2), or,
    with ``leads``, in the ``hres`` forecast layout (a ``prediction_timedelta`` axis
    after ``time``).

    Attributes as in the real store: ``_ARRAY_DIMENSIONS`` and time in hours
    since 2016-01-01.
    """
    nt, nlev, nlat, nlon = len(TIMES), PLEV_HPA.size, LAT_SN.size, LON.size
    lead_axis = [] if leads is None else ["prediction_timedelta"]
    lead_values = [0] if leads is None else leads
    root = zarr.open_group(str(path), mode="w", zarr_format=2)

    def put(name, data, units="", dims=None):
        arr = root.create_array(name, shape=data.shape, dtype=data.dtype)
        arr[...] = data
        arr.attrs["units"] = units
        arr.attrs["_ARRAY_DIMENSIONS"] = dims or [name]

    dims3 = [dims_3d[0], *lead_axis, *dims_3d[1:]]
    dims2 = ["time", *lead_axis, "latitude", "longitude"]
    put("time", np.array([_hours_since_2016(t) for t in TIMES], np.int64), time_units)
    if leads is not None:
        put("prediction_timedelta", np.array(leads, np.int64), lead_units)
    put("latitude", LAT_SN)
    put("longitude", LON)
    put("level", PLEV_HPA)
    lead_shape = [] if leads is None else [len(leads)]
    shape3 = (nt, *lead_shape, nlev, nlat, nlon)
    shape2 = (nt, *lead_shape, nlat, nlon)

    def per_time(make):
        """Stack ``make(temperature)`` over the times (and leads)."""
        out = np.stack(
            [
                np.stack([make(_temperature(i, lead)) for lead in lead_values])
                for i in range(nt)
            ]
        )
        return out if leads is not None else out[:, 0]

    level_shape = (nlev, nlat, nlon)
    # temperature differs between the times (and leads) so the lookup is visible
    temp = per_time(lambda t: np.full(level_shape, t))
    put("temperature", temp.astype(np.float32), "K", dims3)
    # u carries the latitude so the north-to-south flip is visible
    put(
        "u_component_of_wind",
        np.broadcast_to(LAT_SN[:, None], shape3).astype(np.float32),
        dims=dims3,
    )
    put("v_component_of_wind", np.zeros(shape3, np.float32), dims=dims3)
    put("specific_humidity", np.zeros(shape3, np.float32), dims=dims3)
    put("vertical_velocity", np.zeros(shape3, np.float32), omega_units, dims3)
    z = per_time(
        lambda t: np.broadcast_to(
            _isothermal_geopotential(PLEV_HPA * 100.0, t)[:, None, None], level_shape
        )
    )
    put("geopotential", z.astype(np.float32), dims=dims3)
    ps = np.broadcast_to(np.linspace(90000.0, 103000.0, nlon), shape2)
    put("surface_pressure", ps.astype(np.float32), "Pa", dims2)
    put("2m_temperature", np.full(shape2, 290.0, np.float32), "K", dims2)


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


def test_fetch_hres_t0_rejects_another_epoch_or_dimension_order(tmp_path):
    other_epoch = tmp_path / "epoch.zarr"
    _write_wb2_like_store(other_epoch, time_units="hours since 1900-01-01")
    with pytest.raises(SystemExit, match="time units"):
        fetch_hres_t0("2022-09-26T12:00:00", str(other_epoch), workers=1)

    lon_lat = tmp_path / "lon_lat.zarr"
    _write_wb2_like_store(lon_lat, dims_3d=("time", "level", "longitude", "latitude"))
    with pytest.raises(SystemExit, match="dimensions"):
        fetch_hres_t0("2022-09-26T12:00:00", str(lon_lat), workers=1)


LEADS = [0, 6, 12]


@pytest.mark.parametrize("init, i", [("2022-09-26T06", 0), ("2022-09-26T12", 1)])
@pytest.mark.parametrize("lead", LEADS)
def test_fetch_hres_forecast_selects_init_and_lead(tmp_path, init, i, lead):
    store = tmp_path / "hres.zarr"
    _write_wb2_like_store(store, leads=LEADS)
    era5 = fetch_hres_forecast(init, lead, str(store), workers=2)
    np.testing.assert_array_equal(era5.atm["T_mid"], np.float32(_temperature(i, lead)))
    np.testing.assert_array_equal(era5.atm["U"][:, 0, :], 60.0)  # north first
    np.testing.assert_array_equal(era5.atm["U"][:, -1, :], -60.0)
    ps = era5.sfc["ps"].astype(np.float64)
    np.testing.assert_allclose(
        era5.sfc["phis"],
        _isothermal_geopotential(ps, _temperature(i, lead)),
        rtol=0,
        atol=0.1,
    )


def test_fetch_hres_forecast_refuses_what_the_store_lacks(tmp_path):
    store = tmp_path / "hres.zarr"
    _write_wb2_like_store(store, leads=LEADS)
    with pytest.raises(SystemExit, match="no init"):
        fetch_hres_forecast("2022-09-27T00", 6, str(store), workers=1)
    with pytest.raises(SystemExit, match=r"no \+18 h"):
        fetch_hres_forecast("2022-09-26T12", 18, str(store), workers=1)
    with pytest.raises(SystemExit, match="not on the hour"):
        fetch_hres_forecast("2022-09-26T12:30", 6, str(store), workers=1)
    minutes = tmp_path / "minutes.zarr"
    _write_wb2_like_store(minutes, leads=LEADS, lead_units="minutes")
    with pytest.raises(SystemExit, match="not hours"):
        fetch_hres_forecast("2022-09-26T12", 6, str(minutes), workers=1)
    # an analysis store (no lead axis) is not a forecast store
    t0_store = tmp_path / "hres_t0.zarr"
    _write_wb2_like_store(t0_store)
    with pytest.raises((SystemExit, KeyError)):
        fetch_hres_forecast("2022-09-26T12", 0, str(t0_store), workers=1)


def test_fetch_hres_forecast_checks_the_dimension_order(tmp_path):
    lon_lat = tmp_path / "lon_lat.zarr"
    _write_wb2_like_store(
        lon_lat, dims_3d=("time", "level", "longitude", "latitude"), leads=LEADS
    )
    with pytest.raises(SystemExit, match="dimensions"):
        fetch_hres_forecast("2022-09-26T12", 6, str(lon_lat), workers=1)


@pytest.mark.parametrize("init", ["2016-01-01T00", "2020-06-15T12", "2023-01-10T12"])
def test_check_wb2_init_accepts_the_archived_runs(init):
    check_wb2_init(np.datetime64(init))


@pytest.mark.parametrize(
    "init", ["2015-12-31T12", "2023-01-11T00", "2020-06-15T06", "2020-06-15T18"]
)
def test_check_wb2_init_refuses_other_runs(init):
    with pytest.raises(ValueError, match="WeatherBench2 holds"):
        check_wb2_init(np.datetime64(init))
