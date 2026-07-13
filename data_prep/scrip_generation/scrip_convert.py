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
import argparse
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

CHUNK_ELEMS = 200_000
FILL = 9.96920996838687e36


def xyz_to_lonlat_deg(x, y, z):
    lon = np.degrees(np.arctan2(y, x))
    lat = np.degrees(np.arcsin(z))
    lon = np.mod(lon, 360.0)
    return lon, lat


def wrap_lon_0_360(lon):
    return np.mod(lon, 360.0)


def spherical_tri_area(a, b, c):
    cross = np.cross(b, c)
    numer = np.abs(np.einsum("ij,ij->i", a, cross))
    denom = (
        1.0
        + np.einsum("ij,ij->i", a, b)
        + np.einsum("ij,ij->i", b, c)
        + np.einsum("ij,ij->i", c, a)
    )
    return 2.0 * np.arctan2(numer, denom)


def get_total_elements(ds):
    if "num_elem" in ds.dimensions:
        return ds.dimensions["num_elem"].size
    total = 0
    nblk = ds.dimensions["num_el_blk"].size
    for b in range(1, nblk + 1):
        dim = ds.dimensions.get(f"num_el_in_blk{b}")
        if dim is not None:
            total += dim.size
    return total


def get_max_corners(ds):
    nblk = ds.dimensions["num_el_blk"].size
    max_c = 0
    for b in range(1, nblk + 1):
        dim = ds.dimensions.get(f"num_nod_per_el{b}")
        if dim is not None:
            max_c = max(max_c, dim.size)
    return max_c


def default_out_path(in_exodus: str) -> str:
    in_path = Path(in_exodus)
    return str(in_path.with_name(f"{in_path.stem}_scrip.nc"))


def convert_exodus_to_scrip(in_exodus: str, out_scrip: str, chunk_elems: int) -> None:
    with Dataset(in_exodus) as ds:
        if "coord" in ds.variables:
            coord = ds.variables["coord"][:]
            x = coord[0].astype(np.float64)
            y = coord[1].astype(np.float64)
            z = coord[2].astype(np.float64)
        else:
            x = ds.variables["coordx"][:].astype(np.float64)
            y = ds.variables["coordy"][:].astype(np.float64)
            z = ds.variables["coordz"][:].astype(np.float64)

        nblk = ds.dimensions["num_el_blk"].size
        grid_size = get_total_elements(ds)
        grid_corners = get_max_corners(ds)

        with Dataset(out_scrip, "w") as out:
            out.createDimension("grid_size", grid_size)
            out.createDimension("grid_corners", grid_corners)
            out.createDimension("grid_rank", 1)

            out.api_version = np.float32(5.00)
            out.version = np.float32(5.00)
            out.floating_point_word_size = 8
            out.file_size = 0

            v_area = out.createVariable("grid_area", "f8", ("grid_size",))
            v_area.units = "radians^2"

            v_clat = out.createVariable(
                "grid_center_lat", "f8", ("grid_size",), fill_value=FILL
            )
            v_clon = out.createVariable(
                "grid_center_lon", "f8", ("grid_size",), fill_value=FILL
            )
            v_clat.units = "degrees"
            v_clon.units = "degrees"

            v_lat = out.createVariable(
                "grid_corner_lat",
                "f8",
                ("grid_size", "grid_corners"),
                fill_value=FILL,
            )
            v_lon = out.createVariable(
                "grid_corner_lon",
                "f8",
                ("grid_size", "grid_corners"),
                fill_value=FILL,
            )
            v_lat.units = "degrees"
            v_lon.units = "degrees"

            v_mask = out.createVariable("grid_imask", "i4", ("grid_size",))
            v_dims = out.createVariable("grid_dims", "i4", ("grid_rank",))
            v_dims[:] = np.array([grid_size], dtype=np.int32)

            elem_offset = 0

            for b in range(1, nblk + 1):
                conn_name = f"connect{b}"
                if conn_name not in ds.variables:
                    continue

                conn_var = ds.variables[conn_name]
                n_elem_blk = conn_var.shape[0]
                n_corners = conn_var.shape[1]

                gid_name = f"global_id{b}"
                gid_var = ds.variables.get(gid_name, None)

                for i0 in range(0, n_elem_blk, chunk_elems):
                    i1 = min(i0 + chunk_elems, n_elem_blk)
                    conn = conn_var[i0:i1, :].astype(np.int64) - 1

                    if gid_var is not None:
                        idx = gid_var[i0:i1].astype(np.int64) - 1
                    else:
                        idx = np.arange(
                            elem_offset + i0, elem_offset + i1, dtype=np.int64
                        )

                    xv = x[conn]
                    yv = y[conn]
                    zv = z[conn]

                    lon, lat = xyz_to_lonlat_deg(xv, yv, zv)

                    cx = xv.mean(axis=1)
                    cy = yv.mean(axis=1)
                    cz = zv.mean(axis=1)
                    r = np.sqrt(cx * cx + cy * cy + cz * cz)
                    cx /= r
                    cy /= r
                    cz /= r

                    center_lon, center_lat = xyz_to_lonlat_deg(cx, cy, cz)

                    lon_adj = lon.copy()
                    center_lon_col = center_lon[:, None]

                    pole_mask = np.isclose(lat, 90.0) | np.isclose(lat, -90.0)
                    lon_adj = np.where(pole_mask, center_lon_col, lon_adj)

                    lon_diff = center_lon_col - lon_adj
                    lon_adj = np.where(lon_diff > 180.0, lon_adj + 360.0, lon_adj)
                    lon_adj = np.where(lon_diff < -180.0, lon_adj - 360.0, lon_adj)
                    center_lon = wrap_lon_0_360(center_lon)
                    lon_adj = wrap_lon_0_360(lon_adj)

                    if n_corners == 4:
                        p0 = np.stack([xv[:, 0], yv[:, 0], zv[:, 0]], axis=1)
                        p1 = np.stack([xv[:, 1], yv[:, 1], zv[:, 1]], axis=1)
                        p2 = np.stack([xv[:, 2], yv[:, 2], zv[:, 2]], axis=1)
                        p3 = np.stack([xv[:, 3], yv[:, 3], zv[:, 3]], axis=1)
                        area = spherical_tri_area(p0, p1, p2) + spherical_tri_area(
                            p0, p2, p3
                        )
                    else:
                        p0 = np.stack([xv[:, 0], yv[:, 0], zv[:, 0]], axis=1)
                        area = np.zeros((xv.shape[0],), dtype=np.float64)
                        for k in range(1, n_corners - 1):
                            pk = np.stack([xv[:, k], yv[:, k], zv[:, k]], axis=1)
                            pk1 = np.stack(
                                [xv[:, k + 1], yv[:, k + 1], zv[:, k + 1]], axis=1
                            )
                            area += spherical_tri_area(p0, pk, pk1)

                    v_area[idx] = area
                    v_clat[idx] = center_lat
                    v_clon[idx] = center_lon
                    v_mask[idx] = 1

                    if n_corners < grid_corners:
                        lat_pad = np.repeat(
                            lat[:, -1][:, None], grid_corners - n_corners, axis=1
                        )
                        lon_pad = np.repeat(
                            lon_adj[:, -1][:, None], grid_corners - n_corners, axis=1
                        )
                        lat = np.concatenate([lat, lat_pad], axis=1)
                        lon_adj = np.concatenate([lon_adj, lon_pad], axis=1)

                    v_lat[idx, :] = lat
                    v_lon[idx, :] = lon_adj

                elem_offset += n_elem_blk


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a TempestRemap-style Exodus mesh into a SCRIP file."
    )
    parser.add_argument(
        "--in-exodus",
        required=True,
        help="Input Exodus mesh (.g) produced by generate_exodus_meshes.py",
    )
    parser.add_argument(
        "--out-scrip",
        help="Output SCRIP NetCDF path. Defaults to <input-stem>_scrip.nc",
    )
    parser.add_argument(
        "--chunk-elems",
        type=int,
        default=CHUNK_ELEMS,
        help="Elements processed per chunk; lower this if memory is tight",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_scrip = args.out_scrip or default_out_path(args.in_exodus)
    convert_exodus_to_scrip(args.in_exodus, out_scrip, args.chunk_elems)
    print("Wrote:", out_scrip)


if __name__ == "__main__":
    main()
