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
"""One-frame SCREAM-style Zarr v3 IC stores: static inputs, creation and checks.

3D fields are (time, lev, ncol) with the 32 levels 128[3::4] and flat ncol
arrays (no face reshape). The store's metadata is consolidated so loaders can
use ``zarr.open_consolidated``.
"""
from __future__ import annotations

import os
import shutil

import numpy as np
import xarray as xr
import zarr

from screamcast.astronomy import calculate_cosine_zenith_direct
from scripts.initial_conditions.layout import (
    GRID,
    NLEV_SUBSET,
    VARS_1D,
    VARS_2D,
    VARS_3D,
)


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
    elif not units.lower().startswith("deg"):
        raise SystemExit(f"{path}: lat units {units!r} are neither degrees nor radians")
    if np.abs(lat_deg).max() > 90.0:
        raise SystemExit(
            f"{path}: lat is {lat_deg.min():.4g}..{lat_deg.max():.4g} deg after "
            f"reading units {units!r}; the units attribute does not match the values"
        )
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
