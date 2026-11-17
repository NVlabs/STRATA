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
"""SCREAM ne1024pg2 states from an ECMWF HRES forecast (open data), one store per lead.

Each lead of the HRES forecast started at ``--init`` goes through the same steps as
``fast_ic`` and ``hres_ic`` (``fast_ic.build_ic``) into
``<out-dir>/scream_hiccup_ne1024pg2_<tag-prefix>_f<LLL>.zarr``, whose ``initial_time``
is the valid time ``init + lead``. Lead 0 is the HRES analysis, so its store is also an
initial condition (``--ic-store``). The stores of all leads are the drivers of an
anchored regional window (``--boundary-stores``, see ``docs/regional_window.md``)::

    python -m screamcast.initial_conditions.hres_forecast \\
        --init 2025-08-15T12 --leads 0 6 12 18 24 --tag-prefix erin \\
        --out-dir DIR --cache GRIB_CACHE --map MAP --phis PHIS --latlon LATLON \\
        --scream-ref DIR --vert L128
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
    ChunkWriter,
    _lower_priority,
    build_ic,
    load_map,
    load_vert,
    log,
)
from screamcast.initial_conditions.hres_opendata import (
    OPENDATA_ROOT,
    fetch_opendata,
    hres_state,
    opendata_name,
    read_opendata,
)
from screamcast.initial_conditions.ic_store import (
    compute_coszr,
    load_latlon,
    load_static_array,
)
from screamcast.initial_conditions.layout import NCOL, zarr_name


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--init", required=True, help="forecast start, YYYY-MM-DDTHH (UTC)")
    p.add_argument("--leads", required=True, nargs="+", type=int, help="hours")
    p.add_argument("--tag-prefix", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cache", required=True, help="directory for the GRIB downloads")
    p.add_argument("--opendata-root", default=OPENDATA_ROOT)
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
    p.add_argument("--device", default="cuda", help="cuda, or cpu (slower)")
    p.add_argument("--write-workers", type=int, default=os.cpu_count())
    p.add_argument("--rows-per-chunk", type=int, default=8 * 1024 * 1024)
    args = p.parse_args()
    if any(lead < 0 for lead in args.leads) or len(set(args.leads)) != len(args.leads):
        p.error("--leads must be distinct and non-negative")
    return args


def main() -> None:
    args = parse_args()
    argv = [
        sys.executable,
        "-m",
        "screamcast.initial_conditions.hres_forecast",
        *sys.argv[1:],
    ]
    init = np.datetime64(args.init, "h")
    device = torch.device(args.device)
    start = time.perf_counter()

    lat, lon, lat_deg, lon_deg = load_latlon(args.latlon)
    statics = {
        "phis": np.asarray(zarr.open_array(args.phis, mode="r")[:], np.float32),
        "lat": lat,
        "lon": lon,
        "sgh30": load_static_array(args.scream_ref, "sgh30"),
        "landfrac": load_static_array(args.scream_ref, "landfrac"),
    }
    for name, arr in statics.items():
        if arr.shape != (NCOL,):
            raise SystemExit(f"{name} shape {arr.shape}, expected {(NCOL,)}")
    phis_dst = torch.from_numpy(statics["phis"]).to(device)
    remap = load_map(args.map, device)
    vert, p0 = load_vert(args.vert)
    os.makedirs(args.out_dir, exist_ok=True)
    with ThreadPoolExecutor(args.write_workers, initializer=_lower_priority) as writers:
        for lead in sorted(args.leads):
            path = fetch_opendata(init, lead, args.cache, args.opendata_root)
            state = hres_state(read_opendata(path, init, lead))
            if remap.shape != (NCOL, state.sfc["ps"].size):
                raise SystemExit(f"map shape {tuple(remap.shape)} != the HRES grid")
            valid = str(np.datetime64(init + np.timedelta64(lead, "h"), "s"))
            out = os.path.join(
                args.out_dir, zarr_name(f"{args.tag_prefix}_f{lead:03d}")
            )
            writer = ChunkWriter(
                out,
                NCOL,
                valid,
                f"ECMWF HRES open data {args.opendata_root}/"
                f"{opendata_name(init, lead)} via "
                "screamcast.initial_conditions.hres_forecast",
                history_entry(argv),
                writers,
            )
            for name, arr in (
                *statics.items(),
                ("coszr", compute_coszr(valid, lat_deg, lon_deg)),
            ):
                writer.submit(name, 0, arr)
            build_ic(
                state, remap, phis_dst, vert, p0, args.rows_per_chunk, writer.submit
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            writer.close()
            log(f"wrote {out} (+{lead} h, valid {valid})")
    log(f"done in {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
