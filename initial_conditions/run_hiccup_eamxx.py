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
# ruff: noqa: S603 S607
"""Build one EAMxx HICCUP atmospheric IC on ne1024pg2 / L128 from ERA5.

Steps, in HICCUP EAMxx order: horizontal ncremap (fv2fv_flx), rename, time
variables, surface pressure adjustment against SCREAM PHIS, vertical remap
onto L128, then remove_supersaturation (only rewrites qv). PotentialTemperature
and hydrostatic z_mid are derived after the HICCUP steps. 3D fields are written
as (time, lev, ncol).

Run from the repository root inside an environment with HICCUP's dependencies
and conda-forge NCO / TempestRemap::

    python -m initial_conditions.run_hiccup_eamxx --timestamp 2016-08-01T00:00:00 ...
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

import numpy as np
import xarray as xr

from initial_conditions.layout import NCOL, NLEV, hiccup_timestamp
from screamcast.history import history_entry

ARCO_ERA5 = (
    "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/"
    "full_37-1h-0p25deg-chunk-1.zarr-v3"
)
RD = 287.05
CP = 1004.0
KAPPA = RD / CP
GRAV = 9.80616
RV_OVER_RD_M1 = 461.5 / RD - 1.0
DIMS_3D = ("time", "lev", "ncol")
RECORDED: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)


def as_timestamp(value) -> np.datetime64:
    if isinstance(value, np.datetime64):
        return np.datetime64(value, "s")
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1]
    return np.datetime64(text.replace(" ", "T")[:19], "s")


def extract_era5(
    atm_path: str, sfc_path: str, timestamp: str, era5_zarr: str
) -> tuple[int, int]:
    os.makedirs(os.path.dirname(atm_path), exist_ok=True)
    if os.path.isfile(atm_path) and os.path.isfile(sfc_path):
        log(f"reusing ERA5 inputs {atm_path}")
    else:
        import zarr

        def zopen(name):
            return zarr.open_array(era5_zarr + "/" + name, mode="r")

        time_arr = zopen("time")
        origin = np.datetime64("1900-01-01T00:00:00", "s")
        target = np.datetime64(timestamp, "s")
        hours = int((target - origin) / np.timedelta64(1, "h"))
        t0 = int(time_arr[0])
        idx = hours - t0
        if idx < 0 or idx >= time_arr.shape[0] or int(time_arr[idx]) != hours:
            raise SystemExit(
                f"ERA5 time {timestamp} is not an exact hourly sample "
                f"(t0={t0}, hours={hours}, idx={idx})"
            )
        log(f"ERA5 time index {idx} hours-since-1900={hours}")
        lat = np.asarray(zopen("latitude")[:])
        lon = np.asarray(zopen("longitude")[:])
        level = np.asarray(zopen("level")[:])
        when = np.array([target.astype("datetime64[ns]")])
        coords3 = {
            "time": when,
            "level": ("level", level, {"units": "hPa"}),
            "latitude": ("latitude", lat, {"units": "degrees_north"}),
            "longitude": ("longitude", lon, {"units": "degrees_east"}),
        }
        coords2 = {
            "time": when,
            "latitude": coords3["latitude"],
            "longitude": coords3["longitude"],
        }

        def slab(name, rank):
            array = zopen(name)
            units = str(array.attrs.get("units", ""))
            data = np.asarray(array[idx], dtype=np.float32)[None, ...]
            if rank == 3:
                dims = ("time", "level", "latitude", "longitude")
                coords = coords3
            else:
                dims = ("time", "latitude", "longitude")
                coords = coords2
            return (
                xr.DataArray(data, dims=dims, coords=coords, attrs={"units": units}),
                units,
            )

        t, _ = slab("temperature", 3)
        q, _ = slab("specific_humidity", 3)
        u, _ = slab("u_component_of_wind", 3)
        v, _ = slab("v_component_of_wind", 3)
        w, w_units = slab("vertical_velocity", 3)
        sp, _ = slab("surface_pressure", 2)
        z, z_units = slab("geopotential_at_surface", 2)
        t2m, t2_units = slab("2m_temperature", 2)
        if "Pa" not in w_units:
            raise SystemExit(f"vertical velocity units are {w_units}, expected Pa s-1")
        if "m**2" not in z_units and "m2" not in z_units:
            raise SystemExit(f"surface geopotential units are {z_units}")
        if t2_units not in ("K", "kelvin"):
            raise SystemExit(f"2m temperature units are {t2_units}")
        xr.Dataset({"t": t, "q": q, "u": u, "v": v, "w": w}).to_netcdf(atm_path)
        xr.Dataset({"sp": sp, "z": z, "t2m": t2m}).to_netcdf(sfc_path)
        log(f"wrote ERA5 inputs {atm_path}")

    # both files may be reused from an earlier run: check that each holds this one time
    for path in (atm_path, sfc_path):
        with xr.open_dataset(path) as ds:
            decoded = xr.decode_cf(ds)["time"].values
            stamp = as_timestamp(decoded[0] if np.ndim(decoded) else decoded)
            if stamp != np.datetime64(timestamp, "s") or ds.sizes.get("time") != 1:
                raise SystemExit(f"{path} time is {stamp}, expected {timestamp}")
    atm = xr.open_dataset(atm_path)
    nlat = int(atm.sizes["latitude"])
    nlon = int(atm.sizes["longitude"])
    lat = atm["latitude"].values
    lon = atm["longitude"].values
    if not (lat[0] > lat[-1]):
        raise SystemExit("ERA5 latitude is not north-to-south; HICCUP src grid is n2s")
    if abs(float(lon[0])) > 1e-4:
        raise SystemExit(f"ERA5 longitude does not start at Greenwich, lon0={lon[0]}")
    log(f"ERA5 nlat={nlat} nlon={nlon} levels={atm.sizes.get('level')}")
    atm.close()
    return nlat, nlon


def load_scream_phis(phis_zarr: str) -> np.ndarray:
    import zarr

    if not os.path.isfile(os.path.join(phis_zarr, "zarr.json")):
        raise SystemExit(
            f"no SCREAM phis array at {phis_zarr}; download the phis array of the "
            "nvidia/STRATA-SCREAM-sdy1 Hugging Face dataset there "
            "(see docs/initial_conditions.md)"
        )
    phis = np.asarray(zarr.open_array(phis_zarr, mode="r")[:], dtype=np.float32)
    if phis.shape != (NCOL,):
        raise SystemExit(f"SCREAM phis shape {phis.shape} is not {(NCOL,)}")
    if not np.isfinite(phis).all():
        raise SystemExit("SCREAM phis contains non-finite values")
    log(f"SCREAM phis min={phis.min():.3f} max={phis.max():.3f}")
    return phis


def write_topo(path: str, phis: np.ndarray) -> None:
    if os.path.isfile(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    xr.Dataset(
        {
            "PHIS": xr.DataArray(
                phis,
                dims=("ncol",),
                attrs={"units": "m2/s2", "long_name": "surface geopotential"},
            )
        }
    ).to_netcdf(path)
    log(f"wrote topo {path}")


def install_command_log() -> None:
    import hiccup.hiccup_data_class as hdc
    import hiccup.hiccup_data_subclass_ERA5 as era

    def wrap(fn):
        def inner(cmd, *args, **kwargs):
            RECORDED.append(cmd)
            log("HICCUP$ " + cmd)
            return fn(cmd, *args, **kwargs)

        return inner

    hdc.run_cmd = wrap(hdc.run_cmd)
    era.run_cmd = wrap(era.run_cmd)


def configure_hiccup(args, nlat: int, nlon: int):
    import hiccup.hiccup_data_class as hdc
    import hiccup.hiccup_vertical_remap as vremap
    from hiccup import hiccup

    install_command_log()
    vremap.max_workers = args.vremap_workers
    vremap.chunk_mem_gb = args.vremap_chunk_mem_gb
    hdc.state_adj_workers = args.state_adj_workers
    obj = hiccup.create_hiccup_data(
        src_data_name="ERA5",
        target_model="EAMXX",
        dst_horz_grid="ne1024pg2",
        dst_vert_grid="L128",
        input_file_list=[args.era5_atm, args.era5_sfc],
        topo_file=args.topo,
        grid_dir=args.grid_dir,
        map_dir=os.path.dirname(args.map),
        tmp_dir=args.tmp_dir,
        verbose=True,
        check_input_files=False,
    )
    for key in ("qc", "qi", "o3_volume_mix_ratio"):
        obj.atm_var_name_dict.pop(key, None)
    obj.atm_var_name_dict["omega"] = "w"
    obj.sfc_var_name_dict["T_2m"] = "t2m"
    obj.check_file_vars()
    obj.map_file = args.map
    obj.dst_grid_file = args.dst_scrip
    if obj.src_nlat != nlat or obj.src_nlon != nlon:
        raise SystemExit(
            f"HICCUP read nlat={obj.src_nlat} nlon={obj.src_nlon}, "
            f"file has {nlat} x {nlon}"
        )
    return obj


def ensure_map(obj, dst_scrip: str) -> None:
    import netCDF4

    obj.create_src_grid_file()
    obj.dst_grid_file = dst_scrip
    with netCDF4.Dataset(dst_scrip) as ds:
        ncol = int(ds.dimensions["grid_size"].size)
    if ncol != NCOL:
        raise SystemExit(f"destination SCRIP ncol is {ncol}, expected {NCOL}")
    if os.path.isfile(obj.map_file) and os.path.isfile(obj.map_file + ".command.txt"):
        log(f"reusing map {obj.map_file}")
        return
    obj.create_map_file(src_type="FV", dst_type="FV")


def map_command(obj) -> str:
    cmds = [c for c in RECORDED if "fv2fv_flx" in c and "--map_file=" in c]
    if cmds:
        return cmds[-1]
    with open(obj.map_file + ".command.txt") as handle:
        return handle.read().strip()


def column_values(path: str, name: str) -> np.ndarray:
    with xr.open_dataset(path) as ds:
        da = ds[name]
        if "time" in da.dims:
            da = da.isel(time=0)
        return np.asarray(da.values)


def replace_phis(path: str, phis: np.ndarray) -> None:
    ds = xr.open_dataset(path)
    da = ds["phis"]
    expand = tuple(slice(None) if dim == "ncol" else None for dim in da.dims)
    data = np.broadcast_to(phis.astype(np.float32)[expand], da.shape).copy()
    attrs = dict(da.attrs)
    ds["phis"] = da.copy(data=data)
    ds["phis"].attrs = attrs
    tmp = path + ".phis_scream.nc"
    ds.to_netcdf(tmp, encoding={"phis": {"dtype": "float32", "_FillValue": None}})
    ds.close()
    os.replace(tmp, path)


def theta_and_zmid(
    t: np.ndarray,
    qv: np.ndarray,
    ps: np.ndarray,
    phis: np.ndarray,
    hyam: np.ndarray,
    hybm: np.ndarray,
    hyai: np.ndarray,
    hybi: np.ndarray,
    p0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Potential temperature and hydrostatic mid-level height.

    ``t`` and ``qv`` are (lev, ncol) with lev ordered top to bottom; ``ps`` and
    ``phis`` are (ncol,). Heights integrate the virtual temperature upward from
    ``phis / g`` across the hybrid interfaces.
    """
    nlev = hyam.size
    p_mid = hyam[:, None] * p0 + hybm[:, None] * ps[None, :]
    p_int = hyai[:, None] * p0 + hybi[:, None] * ps[None, :]
    theta = t * np.power(p0 / np.clip(p_mid, 1.0, None), KAPPA)
    tv = t * (1.0 + RV_OVER_RD_M1 * np.clip(qv, 0.0, None))
    z_int = np.empty((nlev + 1, t.shape[1]), dtype=np.float64)
    z_mid = np.empty((nlev, t.shape[1]), dtype=np.float64)
    z_int[-1] = phis / GRAV
    for k in range(nlev - 1, -1, -1):
        p_top = np.maximum(p_int[k], 1.0e-6)
        p_bot = np.maximum(p_int[k + 1], 1.0e-6)
        p_m = np.maximum(p_mid[k], 1.0e-6)
        z_int[k] = z_int[k + 1] + (RD * tv[k] / GRAV) * np.log(p_bot / p_top)
        z_mid[k] = z_int[k + 1] + (RD * tv[k] / GRAV) * np.log(p_bot / p_m)
    return theta, z_mid


def write_theta_and_zmid(file_dict: dict[str, str], out_dir: str) -> None:
    """Write PotentialTemperature and z_mid files next to the HICCUP T_mid file."""
    from netCDF4 import Dataset

    t_path = file_dict["T_mid"]
    base = os.path.basename(t_path)
    theta_path = os.path.join(
        out_dir, base.replace(".T_mid.", ".PotentialTemperature.")
    )
    z_path = os.path.join(out_dir, base.replace(".T_mid.", ".z_mid."))
    shutil.copy(t_path, theta_path)
    shutil.copy(t_path, z_path)
    with Dataset(t_path) as tds, Dataset(file_dict["qv"]) as qds, Dataset(
        file_dict["ps"]
    ) as psds, Dataset(file_dict["phis"]) as phds, Dataset(
        theta_path, "r+"
    ) as thds, Dataset(
        z_path, "r+"
    ) as zds:
        t_var = tds.variables["T_mid"]
        q_var = qds.variables["qv"]
        for var in (t_var, q_var):
            if tuple(var.dimensions) != DIMS_3D:
                raise SystemExit(
                    f"{var.name} dims {var.dimensions}, expected {DIMS_3D}"
                )
        hyam = np.asarray(tds.variables["hyam"][:], dtype=np.float64)
        hybm = np.asarray(tds.variables["hybm"][:], dtype=np.float64)
        hyai = np.asarray(tds.variables["hyai"][:], dtype=np.float64)
        hybi = np.asarray(tds.variables["hybi"][:], dtype=np.float64)
        p0 = float(np.asarray(tds.variables["P0"][:]).reshape(-1)[0])
        if hyam.size != NLEV:
            raise SystemExit(f"T_mid lev is {hyam.size}, expected {NLEV}")
        ncol = t_var.shape[-1]
        thds.renameVariable("T_mid", "PotentialTemperature")
        zds.renameVariable("T_mid", "z_mid")
        th = thds.variables["PotentialTemperature"]
        zm = zds.variables["z_mid"]
        th.units = "K"
        zm.units = "m"
        chunk = 1024 * 1024
        for i0 in range(0, ncol, chunk):
            i1 = min(i0 + chunk, ncol)
            theta, z_mid = theta_and_zmid(
                np.asarray(t_var[0, :, i0:i1], dtype=np.float64),
                np.asarray(q_var[0, :, i0:i1], dtype=np.float64),
                np.asarray(psds.variables["ps"][0, i0:i1], dtype=np.float64),
                np.asarray(phds.variables["phis"][0, i0:i1], dtype=np.float64),
                hyam,
                hybm,
                hyai,
                hybi,
                p0,
            )
            th[0, :, i0:i1] = theta.astype(np.float32)
            zm[0, :, i0:i1] = z_mid.astype(np.float32)
            log(f"derived PT/z_mid columns {i0}:{i1}")
    file_dict["PotentialTemperature"] = theta_path
    file_dict["z_mid"] = z_path


def read_history(path: str) -> str | None:
    with xr.open_dataset(path) as ds:
        return ds.attrs.get("history")


def write_history(path: str, history: str) -> None:
    tmp = path + ".history_tmp"
    subprocess.check_call(
        [
            "ncatted",
            "-h",
            "-a",
            f"history,global,o,c,{history}",
            path,
            tmp,
        ]
    )
    os.replace(tmp, path)


def save_qv_before_sat(qv_path: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if os.path.isfile(out_path):
        os.remove(out_path)
    # HICCUP rewrites qv into qv.hiccup_tmp and then mv's it over qv_path, so
    # the hard link keeps the pre-adjustment inode.
    os.link(qv_path, out_path)
    log(f"saved pre-saturation qv to {out_path}")


def record_qv_clip(before_path: str, after_path: str, argv: list[str]) -> float:
    chunks = {"ncol": 1024 * 1024}
    with xr.open_dataset(before_path, chunks=chunks) as before, xr.open_dataset(
        after_path, chunks=chunks
    ) as after:
        q0 = before["qv"]
        changed = int((q0 != after["qv"]).sum().compute())
        clip_frac = changed / q0.size
    log(f"qv clipped fraction {clip_frac:.8f}")
    entry = history_entry(
        [
            "saved qv after L128 vertical remap and before remove_supersaturation"
            " (only qv is modified) by",
            *argv,
        ],
        previous=read_history(before_path),
    )
    write_history(before_path, entry)
    return clip_frac


def run_pipeline(
    obj,
    phis: np.ndarray,
    timestamp: str,
    vert_file: str,
    qv_before_sat: str | None,
    argv: list[str],
) -> dict:
    file_dict = obj.get_multifile_dict(timestamp=hiccup_timestamp(timestamp))
    renamed = True
    for var, path in file_dict.items():
        if not os.path.isfile(path) or os.path.getsize(path) < 1000:
            renamed = False
            break
        with xr.open_dataset(path) as ds:
            if var not in ds.variables:
                renamed = False
                break
    if renamed:
        log("reusing horizontally remapped and renamed temporary files")
    else:
        obj.remap_horizontal_multifile(file_dict)
        obj.rename_vars_multifile(file_dict=file_dict)
    obj.add_time_date_variables_multifile(file_dict=file_dict)
    era_phis = column_values(file_dict["phis"], "phis").astype(np.float64)
    scream = phis.astype(np.float64)
    corr = float(np.corrcoef(era_phis, scream)[0, 1])
    log(
        f"orography correlation remapped ERA5 z vs SCREAM phis = {corr:.6f}; "
        f"ERA5 max {era_phis.max():.1f} SCREAM max {scream.max():.1f}"
    )
    if not np.isfinite(corr) or corr < 0.8:
        raise SystemExit("remapped ERA5 orography does not align with SCREAM phis")
    obj.surface_adjustment_multifile(
        file_dict=file_dict, adj_PS=True, adj_TS=False, adj_T_eam=False
    )
    replace_phis(file_dict["phis"], phis)
    obj.remap_vertical_multifile(file_dict=file_dict, vert_file_name=vert_file)
    if qv_before_sat is not None:
        save_qv_before_sat(file_dict["qv"], qv_before_sat)
    obj.atmos_state_adjustment_multifile(
        file_dict=file_dict,
        adjust_sat=True,
        adjust_wtr=True,
        convert_ozone=False,
    )
    if qv_before_sat is not None:
        record_qv_clip(qv_before_sat, file_dict["qv"], argv)
    write_theta_and_zmid(file_dict, os.path.dirname(file_dict["T_mid"]))
    return file_dict


def validate_nc(path: str, timestamp: str, phis: np.ndarray) -> None:
    ds = xr.open_dataset(path)
    if int(ds.sizes.get("ncol", -1)) != NCOL:
        raise SystemExit(f"output ncol is {ds.sizes.get('ncol')}")
    if int(ds.sizes.get("lev", -1)) != NLEV:
        raise SystemExit(f"output lev is {ds.sizes.get('lev')}")
    required = [
        "T_mid",
        "qv",
        "U",
        "V",
        "ps",
        "phis",
        "omega",
        "T_2m",
        "PotentialTemperature",
        "z_mid",
        "hyam",
        "hybm",
        "P0",
    ]
    missing = [name for name in required if name not in ds.variables]
    if missing:
        raise SystemExit(f"output missing {missing}")
    for name in ("T_mid", "U", "V", "qv", "omega", "z_mid", "PotentialTemperature"):
        if tuple(ds[name].dims) != DIMS_3D:
            raise SystemExit(f"{name} dims {ds[name].dims}, expected {DIMS_3D}")
        sample = ds[name].isel(time=0, lev=0, ncol=slice(0, 1024)).values
        if not np.isfinite(sample).all():
            raise SystemExit(f"{name} has non-finite values")
    omega_units = str(ds["omega"].attrs.get("units", ""))
    if "Pa" not in omega_units:
        raise SystemExit(f"omega units are {omega_units}, expected Pa s-1")
    if (
        not np.isfinite(ds["ps"].values).all()
        or not np.isfinite(ds["T_2m"].values).all()
    ):
        raise SystemExit("ps or T_2m has non-finite values")
    out_phis = np.asarray(ds["phis"].isel(time=0).values, dtype=np.float32)
    if float(np.max(np.abs(out_phis - phis))) != 0.0:
        raise SystemExit("phis does not match SCREAM")
    stamp = as_timestamp(xr.decode_cf(ds)["time"].values[0])
    if stamp != np.datetime64(timestamp, "s"):
        raise SystemExit(f"output valid time is {stamp}")
    ds.close()
    log("NC checks passed")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--timestamp", required=True, help="YYYY-MM-DDTHH:MM:SS")
    p.add_argument("--out", required=True, help="output HICCUP NetCDF IC")
    p.add_argument("--map", required=True, help="ERA5 721x1440 -> ne1024pg2 map")
    p.add_argument("--dst-scrip", required=True, help="ne1024pg2 SCRIP grid file")
    p.add_argument("--data-root", required=True, help="work dir for inputs/grids")
    p.add_argument("--hiccup-root", required=True, help="HICCUP clone")
    p.add_argument("--era5-atm")
    p.add_argument("--era5-sfc")
    p.add_argument("--era5-zarr", default=ARCO_ERA5)
    p.add_argument("--topo")
    p.add_argument("--vert")
    p.add_argument("--grid-dir")
    p.add_argument("--tmp-dir")
    p.add_argument("--phis-zarr", help="local copy of the SCREAM phis array")
    p.add_argument(
        "--save-qv-before-sat",
        default=None,
        help="also write qv after the L128 remap and before remove_supersaturation",
    )
    p.add_argument("--vremap-workers", type=int, default=8)
    p.add_argument("--vremap-chunk-mem-gb", type=float, default=12.0)
    p.add_argument("--state-adj-workers", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    argv = [sys.executable, "-m", "initial_conditions.run_hiccup_eamxx", *sys.argv[1:]]
    sys.path.insert(0, args.hiccup_root)
    os.makedirs(args.data_root, exist_ok=True)
    args.grid_dir = args.grid_dir or os.path.join(args.data_root, "grid")
    args.tmp_dir = args.tmp_dir or os.path.join(args.data_root, "tmp")
    args.phis_zarr = args.phis_zarr or os.path.join(args.data_root, "scream_phis.zarr")
    args.topo = args.topo or os.path.join(
        args.data_root, "data", "scream_ne1024pg2_PHIS.nc"
    )
    args.vert = args.vert or os.path.join(
        args.hiccup_root, "files_vert", "vert_coord_E3SM_L128.nc"
    )
    date, hour = args.timestamp.split("T")
    hh = hour.split(":")[0]
    args.era5_atm = args.era5_atm or os.path.join(
        args.data_root, "data", f"ERA5.atm.{date}.{hh}.nc"
    )
    args.era5_sfc = args.era5_sfc or os.path.join(
        args.data_root, "data", f"ERA5.sfc.{date}.{hh}.nc"
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    os.makedirs(args.tmp_dir, exist_ok=True)
    os.makedirs(args.grid_dir, exist_ok=True)
    nlat, nlon = extract_era5(
        args.era5_atm, args.era5_sfc, args.timestamp, args.era5_zarr
    )
    phis = load_scream_phis(args.phis_zarr)
    write_topo(args.topo, phis)
    obj = configure_hiccup(args, nlat, nlon)
    ensure_map(obj, args.dst_scrip)
    file_dict = run_pipeline(
        obj, phis, args.timestamp, args.vert, args.save_qv_before_sat, argv
    )
    if os.path.isfile(args.out):
        os.remove(args.out)
    obj.combine_files(
        file_dict=file_dict,
        delete_files=False,
        output_file_name=args.out,
        permute_dimensions=True,
        permute_dim_list=list(DIMS_3D),
        expected_dim_list=list(DIMS_3D),
        combine_uv=False,
        use_single_precision=True,
    )
    obj.clean_global_attributes(file_name=args.out)
    # args.out was created above, so there is no earlier history to keep; the
    # map line has no timestamp because the map was built earlier
    write_history(
        args.out, history_entry(argv, previous=f"map built by: {map_command(obj)}")
    )
    validate_nc(args.out, args.timestamp, phis)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
