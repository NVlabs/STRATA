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
"""Convert a HICCUP L128 NetCDF IC to a one-frame SCREAM-style Zarr v3 store.

Reads 3D fields stored as (time, lev, ncol), subsets 128[3::4] -> 32 levels,
and writes flat ncol arrays (no face reshape). The store's metadata is
consolidated so loaders can use ``zarr.open_consolidated``.

Run from the repository root::

    python -m initial_conditions.hiccup_l128_to_zarr --ic IC.nc --out IC.zarr ...
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import xarray as xr
import zarr

from initial_conditions.layout import (
    GRID,
    LEVEL_SUBSET,
    NLEV,
    NLEV_SUBSET,
    VARS_1D,
    VARS_2D,
    VARS_3D,
)
from screamcast.astronomy import calculate_cosine_zenith_direct
from screamcast.history import history_entry

NC_VARS_2D = ("T_2m", "ps")
DIMS_3D = ("time", "lev", "ncol")


def log(msg: str) -> None:
    print(msg, flush=True)


def load_latlon(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (lat, lon) as stored plus (lat, lon) in degrees for coszr."""
    with xr.open_dataset(path) as ds:
        lat = np.asarray(ds["lat"].values, dtype=np.float32)
        lon = np.asarray(ds["lon"].values, dtype=np.float32)
        units = str(ds["lat"].attrs.get("units", ""))
    lat_deg = lat.astype(np.float64)
    lon_deg = lon.astype(np.float64)
    if units.lower().startswith("rad"):
        lat_deg = np.degrees(lat_deg)
        lon_deg = np.degrees(lon_deg)
    return lat, lon, lat_deg, lon_deg


def load_static_array(ref_dir: str, name: str) -> np.ndarray:
    path = os.path.join(ref_dir, name + ".zarr")
    return np.asarray(zarr.open_array(path, mode="r")[:], dtype=np.float32).reshape(-1)


def compute_coszr(
    timestamp: str, lat_deg: np.ndarray, lon_deg: np.ndarray
) -> np.ndarray:
    text = np.datetime_as_string(np.datetime64(timestamp, "s")).replace("T", " ")
    return np.asarray(
        calculate_cosine_zenith_direct(text, lat_deg, lon_deg), dtype=np.float32
    )


def create_ic_store(
    out_path: str, ncol: int, timestamp: str, ic_source: str, history: str
) -> zarr.Group:
    """Create an empty one-frame IC store with the SCREAM layout and root attrs."""
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    root = zarr.open_group(out_path, mode="w", zarr_format=3)
    root.attrs["grid"] = GRID
    root.attrs["initial_time"] = timestamp
    root.attrs["level_subset"] = "128[3::4]"
    root.attrs["ic_source"] = ic_source
    root.attrs["history"] = history
    shapes = (
        [(name, (1, NLEV_SUBSET, ncol), (1, 1, ncol)) for name in VARS_3D]
        + [(name, (1, ncol), (1, ncol)) for name in VARS_2D]
        + [(name, (ncol,), (ncol,)) for name in VARS_1D]
    )
    for name, shape, chunks in shapes:
        root.create_array(
            name, shape=shape, chunks=chunks, dtype="float32", fill_value=0.0
        )
    return root


def write_zarr_from_netcdf(
    ic_path: str,
    out_path: str,
    timestamp: str,
    latlon_path: str,
    scream_ref: str,
    workers: int,
    argv: list[str],
) -> None:
    with xr.open_dataset(ic_path) as ds:
        for name in (*VARS_3D, *NC_VARS_2D, "phis"):
            if name not in ds.variables:
                raise SystemExit(f"IC missing {name}")
        for name in VARS_3D:
            if tuple(ds[name].dims) != DIMS_3D:
                raise SystemExit(f"{name} dims {ds[name].dims}, expected {DIMS_3D}")
        if int(ds.sizes["lev"]) != NLEV:
            raise SystemExit(f"IC lev is {ds.sizes['lev']}, expected {NLEV}")
        ncol = int(ds.sizes["ncol"])
        phis = np.asarray(ds["phis"].isel(time=0).values, dtype=np.float32)
        fields_2d = {
            name: np.asarray(ds[name].isel(time=0).values, dtype=np.float32)
            for name in NC_VARS_2D
        }
        nc_history = ds.attrs.get("history")

    lat, lon, lat_deg, lon_deg = load_latlon(latlon_path)
    statics = {
        name: load_static_array(scream_ref, name) for name in ("sgh30", "landfrac")
    }
    for name, arr in (("lat", lat), ("lon", lon), *statics.items()):
        if arr.shape != (ncol,):
            raise SystemExit(f"{name} shape {arr.shape} does not match IC ncol={ncol}")
    coszr = compute_coszr(timestamp, lat_deg, lon_deg)

    root = create_ic_store(
        out_path,
        ncol,
        timestamp,
        os.path.abspath(ic_path),
        history_entry(argv, previous=nc_history),
    )
    lev_index = np.arange(NLEV)[LEVEL_SUBSET]

    root["phis"][:] = phis
    root["lat"][:] = lat
    root["lon"][:] = lon
    for name, arr in statics.items():
        root[name][:] = arr
    root["coszr"][0, :] = coszr
    for name, arr in fields_2d.items():
        root[name][0, :] = arr

    tasks = [
        (ic_path, name, j, int(lev), out_path)
        for name in VARS_3D
        for j, lev in enumerate(lev_index.tolist())
    ]
    log(f"writing {len(tasks)} 3D level slabs with {workers} workers")
    if workers <= 1:
        for task in tasks:
            _write_one_level(*task)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_write_one_level, *task) for task in tasks]
            for i, fut in enumerate(as_completed(futs), 1):
                fut.result()
                if i % 16 == 0 or i == len(futs):
                    log(f"  3D slabs {i}/{len(futs)}")
    zarr.consolidate_metadata(out_path)
    log(f"consolidated metadata {out_path}")
    log(f"wrote {out_path}")


def _write_one_level(ic_path: str, name: str, j: int, lev: int, out_path: str) -> None:
    with xr.open_dataset(ic_path) as ds:
        slab = np.asarray(ds[name].isel(time=0, lev=lev).values, dtype=np.float32)
    root = zarr.open_group(out_path, mode="r+")
    root[name][0, j, :] = slab


def validate_zarr(path: str, timestamp: str) -> None:
    root = zarr.open_consolidated(path)
    for name in (*VARS_3D, *VARS_2D, *VARS_1D):
        if name not in root:
            raise SystemExit(f"zarr missing {name}")
    ncol = root["phis"].shape[0]
    if tuple(root["U"].shape) != (1, NLEV_SUBSET, ncol):
        raise SystemExit(f"U.shape {root['U'].shape}")
    if tuple(root["T_2m"].shape) != (1, ncol):
        raise SystemExit(f"T_2m.shape {root['T_2m'].shape}")
    if root.attrs.get("grid") != GRID:
        raise SystemExit(f"grid attr {root.attrs.get('grid')}")
    if str(root.attrs.get("initial_time")) != timestamp:
        raise SystemExit(f"initial_time {root.attrs.get('initial_time')}")
    if not np.isfinite(np.asarray(root["U"][0, 0, :1024])).all():
        raise SystemExit("U has non-finite values")
    log("Zarr checks passed")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ic", required=True, help="HICCUP L128 NetCDF IC")
    p.add_argument("--out", required=True, help="output Zarr v3 store")
    p.add_argument("--timestamp", required=True, help="YYYY-MM-DDTHH:MM:SS")
    p.add_argument("--latlon", required=True, help="NetCDF with lat/lon on ncol")
    p.add_argument(
        "--scream-ref",
        required=True,
        help="directory holding SCREAM sgh30.zarr and landfrac.zarr arrays",
    )
    p.add_argument("--workers", type=int, default=16)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    argv = [
        sys.executable,
        "-m",
        "initial_conditions.hiccup_l128_to_zarr",
        *sys.argv[1:],
    ]
    write_zarr_from_netcdf(
        args.ic,
        args.out,
        args.timestamp,
        args.latlon,
        args.scream_ref,
        args.workers,
        argv,
    )
    validate_zarr(args.out, args.timestamp)


if __name__ == "__main__":
    main()
