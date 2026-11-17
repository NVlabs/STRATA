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
"""ECMWF HRES forecasts from the open-data bucket, in fast_ic's ERA5 layout.

ECMWF open data (``ecmwf-forecasts`` on AWS) holds every HRES ``oper`` forecast since
2024-02 as one GRIB2 file per step, on a 0.25 deg 721x1440 grid north-to-south and
from -180 deg longitude, with t, q, u, v, w and gh on the 13 pressure levels 50-1000 hPa
and sp, 2t, msl, 10u/10v, 100u/100v and tcwv at the surface. The files are fetched over
HTTP into a cache (earth2studio's ``IFS`` source needs ``ecmwf-opendata`` and refuses
times before 2024-03) and read with pygrib.

``hres_state`` builds what ``hres_ic.fetch_hres_t0`` builds from WeatherBench2: the
geopotential is ``gh * G0`` and the surface geopotential comes from that profile at sp
(``hres_ic.surface_geopotential``), so ``fast_ic.build_ic`` runs unchanged. Lead 0 of a
forecast is the HRES analysis the case IC is built from.
"""
from __future__ import annotations

import os
import time
import urllib.error
import urllib.request

import numpy as np
import pygrib

from screamcast.initial_conditions.fast_ic import Era5, log
from screamcast.initial_conditions.hres_ic import surface_geopotential

OPENDATA_ROOT = "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"
G0 = 9.80665  # WMO standard gravity, ECMWF's gh = z / G0
PLEV_HPA = np.array([50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000])
NLAT, NLON = 721, 1440
LEVEL_PARAMS = ("t", "q", "u", "v", "w", "gh")
SURFACE_PARAMS = ("sp", "2t", "msl", "10u", "10v", "100u", "100v", "tcwv")
UNITS = {
    "t": "K",
    "q": "kg kg**-1",
    "u": "m s**-1",
    "v": "m s**-1",
    "w": "Pa s**-1",
    "gh": "gpm",
    "sp": "Pa",
    "2t": "K",
    "msl": "Pa",
    "10u": "m s**-1",
    "10v": "m s**-1",
    "100u": "m s**-1",
    "100v": "m s**-1",
    "tcwv": "kg m**-2",
}
HRES_ATM = {"T_mid": "t", "qv": "q", "U": "u", "V": "v", "omega": "w"}


def opendata_name(init: np.datetime64, lead: int) -> str:
    """Bucket key of the ``lead``-hour HRES file of forecast ``init``."""
    stamp = np.datetime_as_string(np.datetime64(init, "h"), unit="h")
    day, hour = stamp.replace("-", "")[:8], stamp[11:13]
    return f"{day}/{hour}z/ifs/0p25/oper/{day}{hour}0000-{lead}h-oper-fc.grib2"


def fetch_opendata(
    init: np.datetime64, lead: int, cache: str, root: str = OPENDATA_ROOT
) -> str:
    """Path of the cached GRIB file, downloaded first if absent.

    S3 answers bursts with 503 "Slow Down"; those and other transient errors are
    retried with backoff. A 404 (no such forecast) is raised at once.
    """
    name = opendata_name(init, lead)
    path = os.path.join(cache, name)
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    url = f"{root}/{name}"
    if not url.startswith("https://"):
        raise SystemExit(f"open-data root must be https: {root}")
    tmp = f"{path}.part{os.getpid()}"
    for attempt in range(8):
        try:
            with urllib.request.urlopen(  # noqa: S310 (https checked above)
                url, timeout=120
            ) as r, open(tmp, "wb") as f:
                size = int(r.headers["Content-Length"])
                while chunk := r.read(1 << 22):
                    f.write(chunk)
            if os.path.getsize(tmp) != size:
                raise OSError(f"short read {os.path.getsize(tmp)} of {size} bytes")
            os.replace(tmp, path)
            log(f"fetched {url} ({size / 2**20:.0f} MiB)")
            return path
        except urllib.error.HTTPError as err:
            if err.code == 404:
                raise SystemExit(f"no HRES open data at {url}") from err
            reason = f"HTTP {err.code}"
        except OSError as err:  # URLError, timeouts, resets, short reads
            reason = str(err)
        wait = min(2**attempt, 60)
        log(f"{url}: {reason}, retry in {wait} s")
        time.sleep(wait)
    if os.path.exists(tmp):
        os.remove(tmp)  # partial download of this process, never a cached file
    raise SystemExit(f"could not fetch {url}")


def to_greenwich(values: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Roll the last axis so longitude starts at 0 deg, as in ARCO ERA5."""
    shift = int(np.argmin(np.abs(np.mod(lon + 180.0, 360.0) - 180.0)))
    rolled = np.mod(np.roll(lon, -shift), 360.0)
    if not np.allclose(rolled, np.arange(lon.size) * (360.0 / lon.size)):
        raise SystemExit(f"longitudes are not a regular grid from 0: {rolled[:3]}")
    return np.roll(values, -shift, axis=-1)


def read_opendata(path: str, init: np.datetime64, lead: int) -> dict[str, np.ndarray]:
    """Fields of one open-data file, north to south from Greenwich, float32.

    Level parameters are (13, 721, 1440) on increasing pressure ``PLEV_HPA``; surface
    ones (721, 1440). Checks that every message is forecast ``init`` at step ``lead``
    and carries the expected units.
    """
    want = {(p, int(lev)) for p in LEVEL_PARAMS for lev in PLEV_HPA}
    want |= {(p, 0) for p in SURFACE_PARAMS}
    got: dict[tuple[str, int], np.ndarray] = {}
    lat = lon = None
    analysis = np.datetime64(init, "s").astype(object)
    with pygrib.open(path) as grbs:
        for msg in grbs:
            level = int(msg.level) if msg.typeOfLevel == "isobaricInhPa" else 0
            key = (msg.shortName, level)
            if key not in want:
                continue
            if msg.analDate != analysis or int(msg.step) != lead:
                raise SystemExit(
                    f"{path}: {key} is {msg.analDate} +{msg.step} h, "
                    f"expected {init} +{lead} h"
                )
            if msg.units != UNITS[msg.shortName]:
                raise SystemExit(f"{path}: {key} in {msg.units!r}")
            if lat is None:
                lat = np.asarray(msg.distinctLatitudes)
                lon = np.asarray(msg.distinctLongitudes)
            got[key] = np.asarray(msg.values, dtype=np.float32)
    missing = sorted(want - set(got))
    if missing:
        raise SystemExit(f"{path} lacks {missing}")
    if (lat.size, lon.size) != (NLAT, NLON):
        raise SystemExit(f"{path} grid is {lat.size}x{lon.size}")
    if not lat[0] > lat[-1]:
        raise SystemExit(f"{path} latitude is not north-to-south")
    out = {
        p: to_greenwich(np.stack([got[(p, int(lev))] for lev in PLEV_HPA]), lon)
        for p in LEVEL_PARAMS
    }
    out |= {p: to_greenwich(got[(p, 0)], lon) for p in SURFACE_PARAMS}
    return out


def hres_state(fields: dict[str, np.ndarray]) -> Era5:
    """``read_opendata`` fields as the ``Era5`` that ``fast_ic.build_ic`` takes."""
    plev_pa = PLEV_HPA.astype(np.float64) * 100.0
    phis = surface_geopotential(
        fields["gh"] * G0, fields["t"][-1], plev_pa, fields["sp"]
    )
    return Era5(
        atm={key: fields[p] for key, p in HRES_ATM.items()},
        sfc={"ps": fields["sp"], "phis": phis, "T_2m": fields["2t"]},
        plev_pa=plev_pa,
    )
