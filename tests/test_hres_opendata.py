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
"""The ECMWF HRES open-data reader on fake GRIB messages: which messages it keeps,
the checks it makes, the level order, the roll to Greenwich and the surface
geopotential it derives."""

import datetime

import numpy as np
import pytest

pytest.importorskip("pygrib")

from screamcast.initial_conditions import hres_opendata as od  # noqa: E402

INIT = np.datetime64("2025-08-15T12", "h")
LAT = np.linspace(90.0, -90.0, od.NLAT)
LON = np.arange(od.NLON) * 0.25 - 180.0  # open data starts at -180 deg


class _Msg:
    def __init__(self, short, level, value, *, step=6, units=None, anal=None):
        self.shortName = short
        self.level = level
        self.typeOfLevel = "isobaricInhPa" if level else "surface"
        self.step = step
        self.units = units or od.UNITS[short]
        self.analDate = anal or datetime.datetime(2025, 8, 15, 12)
        self.distinctLatitudes = LAT
        self.distinctLongitudes = LON
        # each field is its longitude (deg) plus a per-message offset
        self.values = np.broadcast_to(LON + value, (od.NLAT, od.NLON))


def _messages(override=None):
    # levels stored top-down in the file: the reader must not rely on the order
    msgs = [
        _Msg(p, int(lev), 1000.0 * k + float(lev))
        for k, p in enumerate(od.LEVEL_PARAMS)
        for lev in od.PLEV_HPA[::-1]
    ]
    msgs += [_Msg(p, 0, 100000.0 + k) for k, p in enumerate(od.SURFACE_PARAMS)]
    msgs.append(_Msg("tp", 0, 0.0, units="m"))  # not wanted: skipped
    for i, m in enumerate(msgs):
        for key, val in (override or {}).get(i, {}).items():
            setattr(m, key, val)
    return msgs


class _Open:
    def __init__(self, msgs):
        self.msgs = msgs

    def __call__(self, path):
        return self

    def __enter__(self):
        return iter(self.msgs)

    def __exit__(self, *exc):
        return False


def test_bucket_key_of_a_forecast_step():
    assert od.opendata_name(INIT, 6) == (
        "20250815/12z/ifs/0p25/oper/20250815120000-6h-oper-fc.grib2"
    )


def test_to_greenwich_puts_zero_longitude_first():
    rolled = od.to_greenwich(LON.copy(), LON)
    np.testing.assert_array_equal(np.mod(rolled, 360.0), np.arange(od.NLON) * 0.25)


def test_read_keeps_the_wanted_fields_in_pressure_order(monkeypatch):
    monkeypatch.setattr(od.pygrib, "open", _Open(_messages()))
    out = od.read_opendata("x.grib2", INIT, 6)
    assert set(out) == set(od.LEVEL_PARAMS) | set(od.SURFACE_PARAMS)
    t = out["t"]
    assert t.shape == (od.PLEV_HPA.size, od.NLAT, od.NLON)
    # level k holds the message of PLEV_HPA[k] (increasing pressure); column 0 is 0 deg
    np.testing.assert_array_equal(t[:, 0, 0], od.PLEV_HPA.astype(np.float32))
    np.testing.assert_array_equal(out["sp"][0, :3], [100000.0, 100000.25, 100000.5])


@pytest.mark.parametrize(
    "override, match",
    [
        ({0: {"step": 12}}, "expected"),
        ({0: {"units": "degC"}}, "in 'degC'"),
    ],
)
def test_read_refuses_a_wrong_message(monkeypatch, override, match):
    monkeypatch.setattr(od.pygrib, "open", _Open(_messages(override)))
    with pytest.raises(SystemExit, match=match):
        od.read_opendata("x.grib2", INIT, 6)


def test_read_refuses_a_file_missing_a_field(monkeypatch):
    msgs = [m for m in _messages() if not (m.shortName == "w" and m.level == 500)]
    monkeypatch.setattr(od.pygrib, "open", _Open(msgs))
    with pytest.raises(SystemExit, match="lacks"):
        od.read_opendata("x.grib2", INIT, 6)


def test_surface_geopotential_at_a_pressure_level_is_that_level_height():
    """With sp exactly at 850 hPa, phis is the 850 hPa geopotential (gh * G0)."""
    shape = (od.NLAT, od.NLON)
    gh = np.stack(
        [np.full(shape, 16000.0 - 15.0 * lev, np.float32) for lev in od.PLEV_HPA]
    )
    fields = {p: np.zeros((od.PLEV_HPA.size, *shape), np.float32) for p in "tquvw"}
    fields["t"][:] = 280.0
    fields["gh"] = gh
    fields["sp"] = np.full(shape, 85000.0, np.float32)
    fields["2t"] = np.full(shape, 290.0, np.float32)
    state = od.hres_state(fields)
    k850 = int(np.flatnonzero(od.PLEV_HPA == 850)[0])
    np.testing.assert_allclose(state.sfc["phis"], gh[k850] * od.G0, rtol=1e-6)
    np.testing.assert_array_equal(state.plev_pa, od.PLEV_HPA * 100.0)
    assert set(state.atm) == {"T_mid", "qv", "U", "V", "omega"}


@pytest.mark.parametrize(
    "init", ["2024-09-25T06", "2024-03-05T00", "2024-02-29T12", "2023-06-01T00"]
)
def test_check_init_refuses_what_open_data_lacks(init):
    with pytest.raises(ValueError, match="HRES init"):
        od.check_init(np.datetime64(init))


def test_check_init_accepts_complete_cycles():
    for init in ("2024-03-05T12", "2025-08-15T00"):
        od.check_init(np.datetime64(init))


def test_hres_forecast_refuses_an_early_init_before_downloading(monkeypatch, tmp_path):
    from screamcast.initial_conditions import hres_forecast

    def no_download(*args, **kwargs):
        raise AssertionError("downloaded")

    monkeypatch.setattr(hres_forecast, "fetch_opendata", no_download)
    argv = ["hres_forecast", "--init", "2024-02-29T12", "--leads", "0"]
    argv += ["--tag-prefix", "x", "--out-dir", str(tmp_path), "--cache", str(tmp_path)]
    for flag in ("--map", "--phis", "--latlon", "--scream-ref", "--vert"):
        argv += [flag, str(tmp_path / "missing")]
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit, match="complete from 2024-03-05"):
        hres_forecast.main()
