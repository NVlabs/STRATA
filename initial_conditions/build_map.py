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
# ruff: noqa: S603 S607
"""Build the ERA5 721x1440 -> ne1024pg2 ``fv2fv_flx`` map in parallel.

``ncremap -a fv2fv_flx --lrg2sml`` runs two serial TempestRemap commands:
``GenerateOverlapMesh --a <ne1024pg2> --b <ERA5>`` and ``GenerateOfflineMap``
(fv, np 1 -> fv, np 1). At this size they take 13-18 h on one core. The map is
first order, so the weights of a destination column depend only on that
column's overlaps with the ERA5 cells. This script splits the ne1024pg2 SCRIP
file into contiguous column ranges, runs the same two commands on each range
in a process pool and stacks the pieces in column order. On 72 Grace cores
the whole map took 32 min; the two ranges that hold the poles finish last.

An existing map is reused when its ``.command.txt`` sidecar is next to it, as
run_hiccup_eamxx.py does. One without a sidecar is adopted after its n_a/n_b
sizes are checked. The ERA5 SCRIP file is made with the ``ncremap -G`` command
HICCUP uses when it does not exist.

Run from the repository root in an environment with conda-forge ``nco`` and
``tempest-remap``::

    python -m initial_conditions.build_map --dst-scrip ne1024pg2_scrip.nc \\
        --src-scrip grid/scrip_ERA5_721x1440.nc --map map/map_721x1440_to_ne1024pg2.nc
"""
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import netCDF4
import numpy as np

from initial_conditions.layout import NCOL
from screamcast.history import history_entry

ERA5_NLAT, ERA5_NLON = 721, 1440
N_SRC = ERA5_NLAT * ERA5_NLON
# HICCUP's hiccup_data_subclass_ERA5.create_src_grid_file
ERA5_GRID = (
    f"ttl=Equi-Angular grid {ERA5_NLAT}x{ERA5_NLON}"
    f"#latlon={ERA5_NLAT},{ERA5_NLON}#lat_typ=uni#lat_drc=n2s#lon_typ=grn_ctr"
)
OFFLINE_OPTS = ["--in_type", "fv", "--in_np", "1", "--out_type", "fv", "--out_np", "1"]
SCRIP_VARS = (
    "grid_center_lat",
    "grid_center_lon",
    "grid_corner_lat",
    "grid_corner_lon",
    "grid_imask",
    "grid_area",
)
DST_VARS = ("yc_b", "xc_b", "yv_b", "xv_b", "area_b", "mask_b", "frac_b")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def grid_size(path: str) -> int:
    with netCDF4.Dataset(path) as ds:
        return int(ds.dimensions["grid_size"].size)


def map_sizes(path: str) -> tuple[int, int]:
    with netCDF4.Dataset(path) as ds:
        return int(ds.dimensions["n_a"].size), int(ds.dimensions["n_b"].size)


def ensure_src_scrip(path: str, tmp_dir: str) -> None:
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        log(f"writing the ERA5 SCRIP file {path}")
        subprocess.check_call(
            ["ncremap", f"--tmp_dir={tmp_dir}", "-G", ERA5_GRID, "-g", path]
        )
    if grid_size(path) != N_SRC:
        raise SystemExit(f"{path} has {grid_size(path)} cells, expected {N_SRC}")


def write_scrip_slice(src: str, out: str, start: int, stop: int) -> None:
    """Columns [start, stop) of a 1-D (unstructured) SCRIP file."""
    with netCDF4.Dataset(src) as ds, netCDF4.Dataset(out, "w") as part:
        part.createDimension("grid_size", stop - start)
        part.createDimension("grid_corners", ds.dimensions["grid_corners"].size)
        part.createDimension("grid_rank", 1)
        part.createVariable("grid_dims", "i4", ("grid_rank",))[:] = [stop - start]
        for name in SCRIP_VARS:
            if name not in ds.variables:
                continue
            var = ds[name]
            new = part.createVariable(name, var.dtype, var.dimensions)
            new.setncatts({key: var.getncattr(key) for key in var.ncattrs()})
            new[:] = var[start:stop]


def build_part(
    src_scrip: str, dst_scrip: str, part_dir: str, k: int, start: int, stop: int
) -> str:
    """Run GenerateOverlapMesh and GenerateOfflineMap on columns [start, stop)."""
    out = os.path.join(part_dir, f"map_{k:04d}.nc")
    if os.path.isfile(out):
        return out
    scrip = os.path.join(part_dir, f"dst_{k:04d}.nc")
    overlap = os.path.join(part_dir, f"overlap_{k:04d}.g")
    tmp_map = os.path.join(part_dir, f"map_{k:04d}.tmp.nc")
    write_scrip_slice(dst_scrip, scrip, start, stop)
    commands = [
        ["GenerateOverlapMesh", "--allow_no_overlap"]
        + ["--b", src_scrip, "--a", scrip, "--out", overlap],
        ["GenerateOfflineMap", "--in_mesh", src_scrip, "--out_mesh", scrip]
        + ["--ov_mesh", overlap, "--out_map", tmp_map, *OFFLINE_OPTS],
    ]
    with open(os.path.join(part_dir, f"log_{k:04d}.txt"), "w") as handle:
        for cmd in commands:
            handle.write(shlex.join(cmd) + "\n")
            handle.flush()
            if subprocess.call(cmd, stdout=handle, stderr=subprocess.STDOUT) != 0:
                raise RuntimeError(f"part {k} failed; see {handle.name}")
    os.replace(tmp_map, out)
    os.remove(overlap)
    os.remove(scrip)
    return out


def merge_parts(parts: list[str], bounds: np.ndarray, out: str, dst_scrip: str) -> None:
    """Stack the part maps in column order into one map file."""
    n_s = []
    for path in parts:
        with netCDF4.Dataset(path) as ds:
            n_s.append(ds.dimensions["n_s"].size)
    offsets = np.concatenate([[0], np.cumsum(n_s)])
    with netCDF4.Dataset(parts[0]) as first, netCDF4.Dataset(out, "w") as dst:
        for name, dim in first.dimensions.items():
            size = {"n_b": int(bounds[-1]), "n_s": int(offsets[-1])}.get(name)
            dst.createDimension(name, len(dim) if size is None else size)
        for name, var in first.variables.items():
            new = dst.createVariable(name, var.dtype, var.dimensions)
            new.setncatts({key: var.getncattr(key) for key in var.ncattrs()})
            if "n_b" not in var.dimensions and "n_s" not in var.dimensions:
                new[:] = var[:]
        dst["dst_grid_dims"][:] = [int(bounds[-1])]
        attrs = {key: first.getncattr(key) for key in first.ncattrs()}
        attrs.update(
            grid_file_dst=dst_scrip,
            domain_b=dst_scrip,
            grid_file_ovr="per-part overlap meshes (initial_conditions.build_map)",
            history=history_entry(),
        )
        dst.setncatts(attrs)
        frac_a = np.zeros(dst.dimensions["n_a"].size)
        for k, path in enumerate(parts):
            with netCDF4.Dataset(path) as part:
                s0, s1 = offsets[k], offsets[k + 1]
                dst["row"][s0:s1] = part["row"][:] + bounds[k]
                dst["col"][s0:s1] = part["col"][:]
                dst["S"][s0:s1] = part["S"][:]
                for name in DST_VARS:
                    dst[name][bounds[k] : bounds[k + 1]] = part[name][:]
                frac_a += part["frac_a"][:]
        dst["frac_a"][:] = frac_a


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dst-scrip", required=True, help="ne1024pg2 SCRIP grid file")
    p.add_argument(
        "--src-scrip",
        required=True,
        help="ERA5 721x1440 SCRIP grid file (written with ncremap -G if missing)",
    )
    p.add_argument("--map", required=True, help="output map file")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument(
        "--parts", type=int, help="number of column ranges (default 2 x workers)"
    )
    p.add_argument(
        "--tmp-dir",
        help="part files; default <map>.parts, removed after a successful merge",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sidecar = args.map + ".command.txt"
    command = shlex.join(
        [sys.executable, "-m", "initial_conditions.build_map"] + sys.argv[1:]
    )
    if os.path.isfile(args.map) and os.path.getsize(args.map) > 0:
        if os.path.isfile(sidecar):
            log(f"reusing {args.map}")
            return
        if map_sizes(args.map) != (N_SRC, NCOL):
            raise SystemExit(
                f"{args.map} is not a {N_SRC} -> {NCOL} map (n_a/n_b); not adopting it"
            )
        log(f"adopting existing {args.map} (n_a={N_SRC}, n_b={NCOL})")
        with open(sidecar, "w") as handle:
            handle.write(f"adopted by: {command}\n")
        return

    if grid_size(args.dst_scrip) != NCOL:
        raise SystemExit(
            f"{args.dst_scrip} has {grid_size(args.dst_scrip)} cells, expected {NCOL}"
        )
    part_dir = args.tmp_dir or args.map + ".parts"
    os.makedirs(part_dir, exist_ok=True)
    ensure_src_scrip(args.src_scrip, part_dir)
    n_parts = args.parts or 2 * args.workers
    bounds = np.linspace(0, NCOL, n_parts + 1).astype(np.int64)
    log(f"{n_parts} parts of ~{NCOL // n_parts} columns on {args.workers} workers")
    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [
            pool.submit(
                build_part,
                args.src_scrip,
                args.dst_scrip,
                part_dir,
                k,
                int(bounds[k]),
                int(bounds[k + 1]),
            )
            for k in range(n_parts)
        ]
        parts = []
        for k, future in enumerate(futures):
            parts.append(future.result())
            log(f"part {k + 1}/{n_parts} done ({time.perf_counter() - t0:.0f} s)")
    os.makedirs(os.path.dirname(os.path.abspath(args.map)), exist_ok=True)
    tmp_map = args.map + ".tmp"
    merge_parts(parts, bounds, tmp_map, args.dst_scrip)
    os.replace(tmp_map, args.map)
    with open(sidecar, "w") as handle:
        handle.write(command + "\n")
    shutil.rmtree(part_dir)
    log(f"wrote {args.map} in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
