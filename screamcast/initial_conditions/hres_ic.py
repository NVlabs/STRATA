#!/usr/bin/env python
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
"""SCREAM ne1024pg2 IC from the ECMWF HRES analysis with fast_ic's HICCUP steps.

WeatherBench2's ``hres_t0`` store holds the HRES 0-h forecast every 6 h from 2016 to 2022 on the
same 0.25 deg 721x1440 grid as ARCO ERA5, but south-to-north and on 13 pressure levels
(50-1000 hPa). After flipping latitude the ERA5 ``fv2fv_flx`` map applies unchanged and
``fast_ic.build_ic`` runs the HICCUP steps exactly as for ERA5.

The store has no surface geopotential, which HICCUP's ps adjustment needs. It is taken from the
HRES geopotential profile at the HRES surface pressure: log-pressure interpolation between the
bracketing levels (ECMWF fills levels below the ground by extrapolation), and below 1000 hPa the
hypsometric equation with the 1000 hPa temperature. It is therefore consistent with the ps it
adjusts.

WeatherBench2's ``hres`` store holds the HRES forecasts themselves: the 00 and 12 UTC runs
from 2016-01-01 to 2023-01-10 12 UTC, every 6 h to +240 h, on the same grid and levels
(``fetch_hres_forecast``; ``hres_forecast --archive wb2`` and the ``wb2_hres`` source of
``screamcast.replay_targets`` read it). Lead 0 of a forecast is its ``hres_t0`` time.
WeatherBench2's HRES data are ECMWF's, under CC BY 4.0: credit ECMWF when you use them, as
the README's License section says.

Every output field is HRES. Above HRES's 50 hPa top HICCUP holds the 50 hPa values up to the
model top, so the five output levels above 50 hPa (L128 3-19) are not physical (PotentialTemperature
far too cold at the top). STRATA reads L128 ``[3::4][8:]`` (levels 35-127, from 100 hPa down) and
never sees them; anything that reads all 32 output levels must not use these ICs.

Run from the repository root on one GPU, or with ``--device cpu`` (about 5 min on 64 cores)::

    python -m screamcast.initial_conditions.hres_ic --timestamp 2022-09-26T12:00:00 \\
        --tag 2022092612_ian_post_hres --out-dir DIR --map map_721x1440_to_ne1024pg2.nc \\
        --phis PHIS --latlon latlon_ne1024pg2.nc --scream-ref DIR --vert L128.nc
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import zarr

from screamcast.history import history_entry
from screamcast.initial_conditions.fast_ic import (
    RD,
    ChunkWriter,
    Era5,
    _check_grid,
    _lower_priority,
    build_ic,
    check_store_layout,
    load_map,
    load_vert,
    log,
)
from screamcast.initial_conditions.ic_store import (
    compute_coszr,
    load_latlon,
    load_static_array,
)
from screamcast.initial_conditions.layout import NCOL, zarr_name

WB2_HRES_T0 = (
    "https://storage.googleapis.com/weatherbench2/datasets/hres_t0/"
    "2016-2022-6h-1440x721.zarr"
)
WB2_HRES_FORECAST = (
    "https://storage.googleapis.com/weatherbench2/datasets/hres/"
    "2016-2022-0012-1440x721.zarr"
)
# the inits and leads of WB2_HRES_FORECAST (read from the store on 2026-11-24)
WB2_FIRST_INIT = np.datetime64("2016-01-01T00", "h")
WB2_LAST_INIT = np.datetime64("2023-01-10T12", "h")
WB2_LEADS = frozenset(range(0, 241, 6))
WB2_EPOCH = "hours since 2016-01-01"
HRES_ATM = {
    "T_mid": "temperature",
    "qv": "specific_humidity",
    "U": "u_component_of_wind",
    "V": "v_component_of_wind",
    "omega": "vertical_velocity",
}


def surface_geopotential(
    z: np.ndarray, t_bottom: np.ndarray, plev_pa: np.ndarray, ps: np.ndarray
) -> np.ndarray:
    """Geopotential (m2 s-2) at ``ps`` from the profile ``z`` (nlev, ...) on increasing ``plev_pa``."""
    log_p = np.log(plev_pa)
    log_ps = np.log(ps.astype(np.float64))
    i1 = np.clip(np.searchsorted(log_p, log_ps), 1, log_p.size - 1)
    i0 = i1 - 1
    z0 = np.take_along_axis(z, i0[None], 0)[0].astype(np.float64)
    z1 = np.take_along_axis(z, i1[None], 0)[0].astype(np.float64)
    w = (log_ps - log_p[i0]) / (log_p[i1] - log_p[i0])
    inside = z0 + w * (z1 - z0)
    below = z[-1].astype(np.float64) - RD * t_bottom.astype(np.float64) * (
        log_ps - log_p[-1]
    )
    return np.where(log_ps > log_p[-1], below, inside).astype(np.float32)


def check_wb2_init(init: np.datetime64) -> None:
    """Raise ``ValueError`` unless ``WB2_HRES_FORECAST`` holds forecast ``init``."""
    init = np.datetime64(init, "h")
    hour = int(np.datetime_as_string(init, unit="h")[11:13])
    if hour not in (0, 12) or not WB2_FIRST_INIT <= init <= WB2_LAST_INIT:
        raise ValueError(
            f"HRES init {init}: WeatherBench2 holds the 00 and 12 UTC forecasts from "
            f"{WB2_FIRST_INIT} to {WB2_LAST_INIT}"
        )


def fetch_hres_t0(timestamp: str, url: str, workers: int) -> Era5:
    """One HRES analysis time in fast_ic's ERA5 layout (north to south, increasing pressure)."""
    root = zarr.open_group(url, mode="r", zarr_format=2)
    if np.datetime64(timestamp, "s") != np.datetime64(timestamp, "h"):
        raise SystemExit(f"--timestamp {timestamp} is not on the hour")
    idx = _time_index(root, timestamp)
    if idx is None:
        raise SystemExit(f"HRES t0 has no {timestamp} in {url}")
    return _wb2_state(root, (idx,), ("time",), url, workers)


def fetch_hres_forecast(init: str, lead: int, url: str, workers: int) -> Era5:
    """The ``lead``-hour state of the HRES forecast from ``init`` in WeatherBench2's
    ``hres`` store, as ``fetch_hres_t0`` gives an analysis."""
    root = zarr.open_group(url, mode="r", zarr_format=2)
    if np.datetime64(init, "s") != np.datetime64(init, "h"):
        raise SystemExit(f"HRES init {init} is not on the hour")
    idx = _time_index(root, init)
    if idx is None:
        raise SystemExit(f"HRES forecasts have no init {init} in {url}")
    units = str(root["prediction_timedelta"].attrs.get("units", ""))
    if units != "hours":
        raise SystemExit(f"{url}: prediction_timedelta is in {units!r}, not hours")
    k = np.flatnonzero(np.asarray(root["prediction_timedelta"][:]) == lead)
    if k.size != 1:
        raise SystemExit(f"HRES forecast {init} has no +{lead} h in {url}")
    return _wb2_state(
        root, (idx, int(k[0])), ("time", "prediction_timedelta"), url, workers
    )


def _time_index(root: zarr.Group, timestamp: str) -> int | None:
    hours = int(
        (np.datetime64(timestamp, "h") - np.datetime64("2016-01-01T00", "h"))
        / np.timedelta64(1, "h")
    )
    idx = np.flatnonzero(np.asarray(root["time"][:]) == hours)
    return int(idx[0]) if idx.size == 1 else None


def _wb2_state(
    root: zarr.Group, index: tuple, leading: tuple[str, ...], url: str, workers: int
) -> Era5:
    """Era5 of the WeatherBench2 fields at ``index`` along the ``leading`` dimensions."""
    lat = np.asarray(root["latitude"][:])
    lon = np.asarray(root["longitude"][:])
    plev_hpa = np.asarray(root["level"][:])
    if not lat[0] < lat[-1]:
        raise SystemExit("expected HRES latitude south-to-north")
    _check_grid(lat[::-1], lon, plev_hpa)
    check_store_layout(
        root,
        [*HRES_ATM.values(), "geopotential"],
        ["surface_pressure", "2m_temperature"],
        WB2_EPOCH,
        url,
        leading,
    )
    omega_units = str(root["vertical_velocity"].attrs.get("units", ""))
    if "Pa" not in omega_units or "hPa" in omega_units:
        raise SystemExit(f"HRES vertical velocity is in {omega_units!r}, not Pa s-1")
    names = [*HRES_ATM.values(), "geopotential", "surface_pressure", "2m_temperature"]
    with ThreadPoolExecutor(workers) as pool:
        data = dict(
            zip(
                names,
                pool.map(lambda n: np.asarray(root[n][index], dtype=np.float32), names),
            )
        )
    # south-to-north -> north-to-south, like ARCO ERA5
    flip = {n: np.ascontiguousarray(a[..., ::-1, :]) for n, a in data.items()}
    plev_pa = plev_hpa.astype(np.float64) * 100.0
    ps = flip["surface_pressure"]
    phis = surface_geopotential(
        flip["geopotential"], flip["temperature"][-1], plev_pa, ps
    )
    return Era5(
        atm={key: flip[name] for key, name in HRES_ATM.items()},
        sfc={"ps": ps, "phis": phis, "T_2m": flip["2m_temperature"]},
        plev_pa=plev_pa,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--timestamp", required=True, help="YYYY-MM-DDTHH:MM:SS, a 6-hourly time"
    )
    p.add_argument("--tag", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--map", required=True, help="ERA5 721x1440 -> ne1024pg2 map")
    p.add_argument("--phis", required=True, help="SCREAM phis Zarr array")
    p.add_argument("--latlon", required=True, help="NetCDF with lat/lon on ncol")
    p.add_argument(
        "--scream-ref", required=True, help="dir with sgh30.zarr and landfrac.zarr"
    )
    p.add_argument(
        "--vert",
        required=True,
        help="L128 hybrid coefficients (E3SM vert_coord_E3SM_L128.nc)",
    )
    p.add_argument("--hres-zarr", default=WB2_HRES_T0)
    p.add_argument("--device", default="cuda", help="cuda, or cpu (slower)")
    p.add_argument("--fetch-workers", type=int, default=8)
    p.add_argument("--write-workers", type=int, default=os.cpu_count())
    p.add_argument("--rows-per-chunk", type=int, default=8 * 1024 * 1024)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    argv = [
        sys.executable,
        "-m",
        "screamcast.initial_conditions.hres_ic",
        *sys.argv[1:],
    ]
    timestamp = str(np.datetime64(args.timestamp, "s"))
    device = torch.device(args.device)
    start = time.perf_counter()
    hres = fetch_hres_t0(timestamp, args.hres_zarr, args.fetch_workers)
    log(
        f"HRES {timestamp}: {hres.plev_pa.size} levels, {time.perf_counter() - start:.0f} s"
    )
    lat, lon, lat_deg, lon_deg = load_latlon(args.latlon)
    statics = {
        "phis": np.asarray(zarr.open_array(args.phis, mode="r")[:], np.float32),
        "lat": lat,
        "lon": lon,
        "sgh30": load_static_array(args.scream_ref, "sgh30"),
        "landfrac": load_static_array(args.scream_ref, "landfrac"),
    }
    remap = load_map(args.map, device)
    if remap.shape != (NCOL, hres.sfc["ps"].size):
        raise SystemExit(f"map shape {tuple(remap.shape)} does not match the HRES grid")
    vert, p0 = load_vert(args.vert)
    out = os.path.join(args.out_dir, zarr_name(args.tag))
    os.makedirs(args.out_dir, exist_ok=True)
    with ThreadPoolExecutor(args.write_workers, initializer=_lower_priority) as writers:
        writer = ChunkWriter(
            out,
            NCOL,
            timestamp,
            f"ECMWF HRES t0 {args.hres_zarr} via screamcast.initial_conditions.hres_ic",
            history_entry(argv),
            writers,
        )
        for name, arr in (
            *statics.items(),
            ("coszr", compute_coszr(timestamp, lat_deg, lon_deg)),
        ):
            writer.submit(name, 0, arr)
        build_ic(
            hres,
            remap,
            torch.from_numpy(statics["phis"]).to(device),
            vert,
            p0,
            args.rows_per_chunk,
            writer.submit,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        writer.close()
    log(f"wrote {out} in {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
