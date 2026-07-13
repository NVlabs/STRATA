#!/usr/bin/env python3
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
"""Derive the per-column lat/lon auxiliary file from a SCRIP grid file.

Training and cubed-sphere inference read ``latlon_ne1024pg2.nc``: a NetCDF
file with float32 ``lat``/``lon`` coordinate variables (degrees) over the
``ncol`` dimension, in the model's global column ordering. That ordering is
exactly the SCRIP cell ordering produced by this directory's mesh workflow,
so the file is just the SCRIP ``grid_center_lat``/``grid_center_lon``
columns cast to float32:

    python derive_latlon.py --in-scrip data/ne1024pg2_scrip.nc \\
        --out data/latlon_ne1024pg2.nc
"""

import argparse
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


def derive(in_scrip: Path, out_path: Path) -> None:
    with Dataset(in_scrip, "r") as src:
        lat = np.asarray(src.variables["grid_center_lat"][:], dtype=np.float32)
        lon = np.asarray(src.variables["grid_center_lon"][:], dtype=np.float32)
        units = getattr(src.variables["grid_center_lat"], "units", "degrees")
    if units.startswith("radian"):
        lat = np.degrees(lat).astype(np.float32)
        lon = np.degrees(lon).astype(np.float32)

    ncol = lat.shape[0]
    with Dataset(out_path, "w") as out:
        out.createDimension("ncol", ncol)
        v_lat = out.createVariable("lat", "f4", ("ncol",))
        v_lon = out.createVariable("lon", "f4", ("ncol",))
        v_lat.units = "degrees_north"
        v_lon.units = "degrees_east"
        v_lat[:] = lat
        v_lon[:] = lon
        out.title = f"per-column cell-center lat/lon derived from {in_scrip.name}"
    print(f"wrote {out_path} (ncol={ncol})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive the per-column lat/lon file from a SCRIP grid file."
    )
    parser.add_argument("--in-scrip", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    derive(args.in_scrip, args.out)


if __name__ == "__main__":
    main()
