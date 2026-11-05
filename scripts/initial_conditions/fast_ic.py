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
#
# Portions of this file are derived from HICCUP
# (https://github.com/E3SM-Project/HICCUP), released under the following license:
#
# HICCUP is released under the following 3-Clause BSD Open Source license.
#
# *******************************************************************************
# Copyright 2026, Lawrence Livermore National Security, LLC All Rights Reserved
#
# Software Name: Hindcast Initial Condition Creation Utility/Processor (HICCUP)
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# 3. Neither the names of Lawrence Livermore National Security, LLC or the
# Department of Energy nor the names of its contributors may be used to endorse
# or promote products derived from this software without specific prior written
# permission.
#
# THE SOFTWARE IS SUPPLIED "AS IS" WITHOUT WARRANTY OF ANY KIND.
#
# NEITHER THE UNITED STATES GOVERNMENT, NOR THE UNITED STATES DEPARTMENT OF
# ENERGY, NOR LAWRENCE LIVERMORE NATIONAL SECURITY, LLC, NOR ANY OF THEIR
# EMPLOYEES, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LEGAL
# LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS OF
# ANY INFORMATION, DATA, APPARATUS, PRODUCT, OR PROCESS DISCLOSED, OR REPRESENTS
# THAT ITS USE WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS.
#
# *******************************************************************************
"""Build SCREAM ne1024pg2 Zarr ICs from ERA5 on one GPU (about 80 s for one IC).

This is a torch port of the HICCUP EAMxx steps in HICCUP's NetCDF multifile
workflow (HICCUP commit 3dd04f3; these routines are the same at 0dec8bb, which
added the license above):

1. ``remap_horizontal_multifile``: apply the ncremap ``fv2fv_flx`` map file.
2. ``adjust_surface_pressure``: adjust ps from ERA5 to SCREAM PHIS.
3. ``remap_vertical_py``: log-pressure interpolation onto L128.
4. ``remove_supersaturation``: clip qv to saturation over liquid.

PotentialTemperature and hydrostatic z_mid (with virtual temperature) are then
computed on L128. Each kernel keeps the float32/float64 promotion and operation
order that HICCUP gets from numpy 2: on ten tropical-cyclone cases the output
matched the HICCUP pipeline bit for bit apart from rare last-bit differences in
transcendental functions. The output is a one-frame Zarr v3 store (see
``ic_store``); no L128 NetCDF is produced.

The output is on the ne1024pg2 physics grid that STRATA is trained on. It is not
an initial condition EAMxx can start from: EAMxx reads its ICs on the np4 GLL
grid.

The vertical steps run one target level at a time over every column, bottom
up, so z_mid accumulates in HICCUP's order and each output level is handed to
the Zarr writer threads as soon as it is final.

Run from the repository root in the STRATA Docker image::

    python -m scripts.initial_conditions.fast_ic --timestamp 2017-09-14T00:00:00 \\
        --out-dir ics --map map_721x1440_to_ne1024pg2.nc ...
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import netCDF4
import numpy as np
import torch
import zarr
from numcodecs import Zstd
from zarr.codecs import BytesCodec, Endian, ZstdCodec

from screamcast.history import history_entry
from scripts.initial_conditions.ic_store import (
    compute_coszr,
    create_ic_store,
    load_latlon,
    load_static_array,
    validate_zarr,
)
from scripts.initial_conditions.layout import (
    NCOL,
    NLEV,
    NLEV_SUBSET,
    VARS_3D,
    zarr_name,
)

ARCO_ERA5 = (
    "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/"
    "full_37-1h-0p25deg-chunk-1.zarr-v3"
)
# PotentialTemperature and z_mid
RD = 287.05
CP = 1004.0
KAPPA = RD / CP
GRAV = 9.80616
RV_OVER_RD_M1 = 461.5 / RD - 1.0
# hiccup/hiccup_constants.py
STD_LAPSE = 0.0065
GRAVIT = 9.80616
R_GAS = 6.02214e26 * 1.38065e-23
RD_AIR = R_GAS / 28.966
R_VAPOR = R_GAS / 18.0
# hiccup/hiccup_state_adjustment.py and hiccup/hiccup_vertical_remap.py
PHIS_THRESHOLD = 1e-3
QV_MIN = 1.0e-9
T_LAPSE_PER_PA = 6.5e-4

# Era5 key -> ARCO ERA5 variable
ATM_FIELDS = {
    "T_mid": "temperature",
    "qv": "specific_humidity",
    "U": "u_component_of_wind",
    "V": "v_component_of_wind",
    "omega": "vertical_velocity",
}
SFC_FIELDS = {
    "ps": "surface_pressure",
    "phis": "geopotential_at_surface",
    "T_2m": "2m_temperature",
}
# output level j is model level OUTPUT_LEVEL0 + OUTPUT_STRIDE * j (128[3::4])
OUTPUT_LEVEL0 = 3
OUTPUT_STRIDE = 4


def log(msg: str) -> None:
    print(msg, flush=True)


@dataclass
class Era5:
    """One ERA5 time on the 0.25 deg grid, north to south from Greenwich.

    ``atm`` fields are (nlev, nlat, nlon) with pressure increasing along nlev;
    ``sfc`` fields are (nlat, nlon); ``plev_pa`` is (nlev,) in Pa.
    """

    atm: dict[str, np.ndarray]
    sfc: dict[str, np.ndarray]
    plev_pa: np.ndarray


def _check_grid(lat: np.ndarray, lon: np.ndarray, plev_hpa: np.ndarray) -> None:
    if not lat[0] > lat[-1]:
        raise SystemExit("ERA5 latitude is not north-to-south")
    if abs(float(lon[0])) > 1e-4:
        raise SystemExit(f"ERA5 longitude does not start at Greenwich, lon0={lon[0]}")
    if not np.all(np.diff(plev_hpa) > 0):
        raise SystemExit("ERA5 pressure levels must increase from top to bottom")


def check_store_layout(
    root: zarr.Group,
    names_3d: list[str],
    names_2d: list[str],
    time_units: str,
    url: str,
) -> None:
    """Stop on a store whose time epoch or dimension order the readers do not assume."""
    units = str(root["time"].attrs.get("units", ""))
    if not units.startswith(time_units):
        raise SystemExit(f"{url}: time units are {units!r}, expected {time_units!r}")
    expected = {
        **{name: ["time", "level", "latitude", "longitude"] for name in names_3d},
        **{name: ["time", "latitude", "longitude"] for name in names_2d},
    }
    for name, dims in expected.items():
        got = list(root[name].attrs.get("_ARRAY_DIMENSIONS", []))
        if got != dims:
            raise SystemExit(f"{url}: {name} dimensions are {got}, expected {dims}")


def fetch_era5_arco(timestamp: str, era5_zarr: str, workers: int) -> Era5:
    """Read one hourly ERA5 time from the ARCO Zarr store with parallel requests."""
    root = zarr.open_group(era5_zarr, mode="r")
    hours = int(
        (np.datetime64(timestamp, "h") - np.datetime64("1900-01-01T00", "h"))
        / np.timedelta64(1, "h")
    )
    if np.datetime64(timestamp, "s") != np.datetime64(timestamp, "h"):
        raise SystemExit(f"ERA5 is hourly; {timestamp} is not on the hour")
    times = root["time"]
    idx = hours - int(times[0])
    if idx < 0 or idx >= times.shape[0] or int(times[idx]) != hours:
        raise SystemExit(f"ERA5 time {timestamp} is not in {era5_zarr}")
    lat = np.asarray(root["latitude"][:])
    lon = np.asarray(root["longitude"][:])
    plev_hpa = np.asarray(root["level"][:])
    _check_grid(lat, lon, plev_hpa)

    names = [*ATM_FIELDS.values(), *SFC_FIELDS.values()]
    check_store_layout(
        root,
        list(ATM_FIELDS.values()),
        list(SFC_FIELDS.values()),
        "hours since 1900-01-01",
        era5_zarr,
    )
    units = {name: str(root[name].attrs.get("units", "")) for name in names}
    if "Pa" not in units["vertical_velocity"] or "hPa" in units["vertical_velocity"]:
        raise SystemExit(f"vertical velocity units are {units['vertical_velocity']}")
    if units["temperature"] != "K" or units["2m_temperature"] != "K":
        raise SystemExit("ERA5 temperatures are not in K")
    with ThreadPoolExecutor(workers) as pool:
        data = dict(
            zip(
                names,
                pool.map(lambda n: np.asarray(root[n][idx], dtype=np.float32), names),
            )
        )
    return Era5(
        atm={key: data[arco] for key, arco in ATM_FIELDS.items()},
        sfc={key: data[arco] for key, arco in SFC_FIELDS.items()},
        plev_pa=plev_hpa.astype(np.float64) * 100.0,
    )


def load_map(path: str, device: torch.device) -> torch.Tensor:
    """Read an ncremap map file as a (n_b, n_a) float64 CSR matrix on device."""
    with netCDF4.Dataset(path) as ds:
        weights = np.asarray(ds.variables["S"][:], dtype=np.float64)
        row = np.asarray(ds.variables["row"][:], dtype=np.int64) - 1
        col = np.asarray(ds.variables["col"][:], dtype=np.int64) - 1
        shape = (ds.dimensions["n_b"].size, ds.dimensions["n_a"].size)
    # ncremap writes links sorted by (row, col); sort only if this one is not
    if not np.all((np.diff(row) > 0) | ((np.diff(row) == 0) & (np.diff(col) > 0))):
        order = np.lexsort((col, row))
        row, col, weights = row[order], col[order], weights[order]
    crow = np.zeros(shape[0] + 1, dtype=np.int64)
    np.cumsum(np.bincount(row, minlength=shape[0]), out=crow[1:])
    return torch.sparse_csr_tensor(
        torch.from_numpy(crow).to(device),
        torch.from_numpy(col).to(device),
        torch.from_numpy(weights).to(device),
        size=shape,
    )


def load_vert(path: str) -> tuple[dict[str, np.ndarray], float]:
    """Read float64 hybrid coefficients (hyam, hybm, hyai, hybi) and P0."""
    with netCDF4.Dataset(path) as ds:
        vert = {
            name: np.asarray(ds.variables[name][:], dtype=np.float64)
            for name in ("hyam", "hybm", "hyai", "hybi")
        }
        p0 = float(ds.variables["P0"][:]) if "P0" in ds.variables else 1.0e5
    if vert["hyam"].size != NLEV:
        raise SystemExit(f"{path} has {vert['hyam'].size} levels, expected {NLEV}")
    return vert, p0


def remap_horizontal(
    remap: torch.Tensor, src: torch.Tensor, rows_per_chunk: int
) -> torch.Tensor:
    """Apply a (n_b, n_a) map to (n_a, F) float32 source fields.

    Returns (F, n_b) float32. Like ncremap, the sum is accumulated in float64
    and stored as float32.
    """
    n_b, n_a = remap.shape
    src64 = src.double()
    dst = torch.empty((src.shape[1], n_b), dtype=torch.float32, device=src.device)
    crow = remap.crow_indices()
    col = remap.col_indices()
    val = remap.values()
    for r0 in range(0, n_b, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, n_b)
        a, b = int(crow[r0]), int(crow[r1])
        block = torch.sparse_csr_tensor(
            crow[r0 : r1 + 1] - a, col[a:b], val[a:b], size=(r1 - r0, n_a)
        )
        dst[:, r0:r1] = (block @ src64).float().T
    return dst


def adjust_surface_pressure(
    ps: torch.Tensor,
    phis_src: torch.Tensor,
    t_bot: torch.Tensor,
    phis_dst: torch.Tensor,
    p_bot: float,
) -> torch.Tensor:
    """HICCUP ``adjust_surface_pressure`` for a pressure-level source.

    ``ps``, ``phis_src`` and ``t_bot`` (T on the lowest source level ``p_bot``)
    are the float32 remapped ERA5 fields; ``phis_dst`` is the float32 target
    PHIS. Returns float64 ps, unchanged where ``|phis_src - phis_dst|`` is
    within ``PHIS_THRESHOLD``.
    """
    alpha = STD_LAPSE * RD_AIR / GRAVIT
    ps64 = ps.double()
    t_star = t_bot.double() + (alpha * t_bot).double() * (ps64 / p_bot - 1.0)
    del_phis = phis_src - phis_dst
    ps_new = ps64 * torch.pow(
        1.0 + (alpha * del_phis).double() / (RD_AIR * t_star), 1.0 / alpha
    )
    return torch.where(del_phis.abs() > PHIS_THRESHOLD, ps_new, ps64)


def hybrid_pressure(a: float, b: float, p0: float, ps: torch.Tensor) -> torch.Tensor:
    """``a * p0 + b * ps`` in float64, as one element where ``b == 0``.

    On pure pressure levels every column has the same pressure, so the float64
    logs, divisions and powers that follow are evaluated once and broadcast.
    """
    if b == 0.0:
        return torch.full((1,), a * p0, dtype=torch.float64, device=ps.device)
    return a * p0 + b * ps


@dataclass
class Stencil:
    """Log-pressure interpolation weights for one target level.

    HICCUP's ``_interp_block`` weights depend only on pressure, so every field
    on a level shares them. ``i0``/``i1`` (1, m) bracket the target in the
    increasing source levels and ``w`` (m,) is the float64 weight, where m is
    the number of columns or 1 on a pure pressure level.
    """

    p: torch.Tensor
    log_p: torch.Tensor
    i0: torch.Tensor
    i1: torch.Tensor
    w: torch.Tensor


def log_pressure_stencil(p: torch.Tensor, log_p_source: torch.Tensor) -> Stencil:
    """Bracket float64 target pressures ``p`` (m,) in ``log_p_source`` (K,)."""
    log_p = torch.log(p)
    i1 = torch.searchsorted(log_p_source, log_p).clamp_(1, log_p_source.numel() - 1)
    i0 = i1 - 1
    x0 = log_p_source[i0]
    w = (log_p - x0) / (log_p_source[i1] - x0)
    return Stencil(p, log_p, i0[None], i1[None], w)


def interp_level(
    stencil: Stencil,
    f_source: torch.Tensor,
    p_source: torch.Tensor,
    log_p_source: torch.Tensor,
    lapse: bool,
) -> torch.Tensor:
    """HICCUP ``_interp_block`` (log pressure, shared source levels) on one level.

    ``f_source`` is (K, n) float32 on the increasing pressures ``p_source``
    (with logs ``log_p_source``). Targets outside the source range take the
    end values; with ``lapse`` (temperature), targets below the lowest source
    level follow ``T_LAPSE_PER_PA`` instead. Returns (n,) float32.
    """
    shape = (1, f_source.shape[1])
    y0 = torch.gather(f_source, 0, stencil.i0.expand(shape))[0]
    dy = torch.gather(f_source, 0, stencil.i1.expand(shape))[0] - y0
    y = y0.double() + stencil.w * dy.double()
    bottom = f_source[-1].double()
    y = torch.where(stencil.log_p < log_p_source[0], f_source[0].double(), y)
    y = torch.where(stencil.log_p > log_p_source[-1], bottom, y)
    if lapse:
        p_bot = p_source[-1]
        y = torch.where(
            stencil.p > p_bot, bottom + T_LAPSE_PER_PA * (stencil.p - p_bot), y
        )
    return y.float()


def remove_supersaturation(
    qv: torch.Tensor, t: torch.Tensor, p_hpa: torch.Tensor
) -> torch.Tensor:
    """HICCUP ``remove_supersaturation``: float32 qv and T, float64 p in hPa.

    Returns float64 qv clipped to Bolton saturation over liquid, with negative
    values set to ``QV_MIN``.
    """
    es = 6.112 * torch.exp(17.67 * (t - 273.0) / (t - 273.0 + 243.5))
    r_sat = ((RD_AIR / R_VAPOR) * es).double() / (p_hpa - es.double())
    qv_sat = r_sat / (1.0 + r_sat)
    qv_sat = torch.where(qv_sat >= 0.0, qv_sat, 1.0)
    qv64 = qv.double()
    rh = qv64 / qv_sat
    out = torch.where(rh > 1.0, qv_sat, qv64)
    return torch.where(rh < 0.0, QV_MIN, out)


def potential_temperature(t: torch.Tensor, p: torch.Tensor, p0: float):
    """``T (P0/p)^(Rd/cp)`` in float64."""
    return t * torch.pow(p0 / p.clamp(min=1.0), KAPPA)


def build_ic(
    era5: Era5,
    remap: torch.Tensor,
    phis_dst: torch.Tensor,
    vert: dict[str, np.ndarray],
    p0: float,
    rows_per_chunk: int,
    emit: Callable[[str, int, torch.Tensor], None],
) -> None:
    """Run the HICCUP steps on device and ``emit(name, j, field)`` each output.

    3D outputs are emitted per output level ``j`` as (ncol,) float32; ``T_2m``
    and ``ps`` are emitted with ``j = 0``. ``phis_dst`` is the float32 SCREAM
    PHIS on device.
    """
    device = phis_dst.device
    nk = era5.plev_pa.size
    n_a = remap.shape[1]
    atm_names = list(ATM_FIELDS)
    src = torch.cat(
        [
            torch.from_numpy(era5.atm[name]).to(device).reshape(nk, n_a).T
            for name in atm_names
        ]
        + [
            torch.from_numpy(era5.sfc[name]).to(device).reshape(n_a, 1)
            for name in SFC_FIELDS
        ],
        dim=1,
    )
    dst = remap_horizontal(remap, src, rows_per_chunk)
    del src
    fields = {name: dst[i * nk : (i + 1) * nk] for i, name in enumerate(atm_names)}
    ps_era5, phis_era5, t_2m = dst[len(atm_names) * nk :]
    p_source = torch.from_numpy(era5.plev_pa).to(device)
    log_p_source = torch.log(p_source)

    ps = adjust_surface_pressure(
        ps_era5, phis_era5, fields["T_mid"][-1], phis_dst, float(era5.plev_pa[-1])
    )
    emit("T_2m", 0, t_2m)
    emit("ps", 0, ps.float())

    hyam, hybm, hyai, hybi = (
        vert[k].tolist() for k in ("hyam", "hybm", "hyai", "hybi")
    )
    z_int = phis_dst.double() / GRAV
    p_int_bot = hybrid_pressure(hyai[NLEV], hybi[NLEV], p0, ps).clamp(min=1.0e-6)
    for k in range(NLEV - 1, -1, -1):
        st = log_pressure_stencil(
            hybrid_pressure(hyam[k], hybm[k], p0, ps), log_p_source
        )
        t = interp_level(st, fields["T_mid"], p_source, log_p_source, lapse=True)
        qv_raw = interp_level(st, fields["qv"], p_source, log_p_source, lapse=False)
        qv = remove_supersaturation(qv_raw, t, st.p / 1e2)
        # hydrostatic heights, summed over the interfaces from the surface up
        tv = t.double() * (1.0 + RV_OVER_RD_M1 * qv.clamp(min=0.0))
        scale = RD * tv / GRAV
        p_int_top = hybrid_pressure(hyai[k], hybi[k], p0, ps).clamp(min=1.0e-6)
        if k % OUTPUT_STRIDE == OUTPUT_LEVEL0:
            j = (k - OUTPUT_LEVEL0) // OUTPUT_STRIDE
            z_mid = z_int + scale * torch.log(p_int_bot / st.p.clamp(min=1.0e-6))
            emit("z_mid", j, z_mid.float())
            theta = potential_temperature(t.double(), st.p, p0)
            emit("PotentialTemperature", j, theta.float())
            emit("qv", j, qv.float())
            for name in ("U", "V", "omega"):
                emit(
                    name,
                    j,
                    interp_level(st, fields[name], p_source, log_p_source, False),
                )
        z_int = z_int + scale * torch.log(p_int_bot / p_int_top)
        p_int_bot = p_int_top


def _chunk_codec(array: zarr.Array) -> Zstd:
    codecs = array.metadata.codecs
    if not (
        len(codecs) == 2
        and isinstance(codecs[0], BytesCodec)
        and codecs[0].endian == Endian.little
        and isinstance(codecs[1], ZstdCodec)
    ):
        raise SystemExit(f"{array.name} codecs {codecs} are not bytes(little)+zstd")
    return Zstd(level=codecs[1].level, checksum=codecs[1].checksum)


class ChunkWriter:
    """Fill a store from ``create_ic_store`` one chunk at a time on a thread pool.

    zarr encodes these ~100 MB chunks one at a time, which takes minutes. Here
    each chunk is encoded with the codec from its array metadata (zstd releases
    the GIL) and written to the key that metadata's chunk key encoding gives.
    ``close`` consolidates the metadata and reads one chunk per array back
    through zarr.
    """

    def __init__(
        self,
        out_path: str,
        ncol: int,
        timestamp: str,
        ic_source: str,
        history: str,
        pool: ThreadPoolExecutor,
    ):
        if sys.byteorder != "little":
            raise SystemExit(
                "chunks are written in host byte order, which must be little"
            )
        self.out_path = out_path
        self.timestamp = timestamp
        self.pool = pool
        root = create_ic_store(out_path, ncol, timestamp, ic_source, history)
        self.arrays = {name: root[name] for name in root.array_keys()}
        self.codecs = {name: _chunk_codec(a) for name, a in self.arrays.items()}
        self.futures: list[Future] = []
        self.first: dict[str, np.ndarray] = {}

    def submit(self, name: str, j: int, data: torch.Tensor | np.ndarray) -> None:
        self.futures.append(self.pool.submit(self._write, name, j, data))

    def _write(self, name: str, j: int, data: torch.Tensor | np.ndarray) -> None:
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        data = np.ascontiguousarray(data, dtype=np.float32)
        array = self.arrays[name]
        coords = (0, j, 0) if name in VARS_3D else (0,) * array.ndim
        key = array.metadata.chunk_key_encoding.encode_chunk_key(coords)
        path = os.path.join(self.out_path, name, *key.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(self.codecs[name].encode(data))
        if j == 0:
            self.first[name] = data

    def close(self) -> None:
        for future in self.futures:
            future.result()
        # one chunk per output level for 3D arrays, one chunk for every other array
        expected = len(self.arrays) + len(VARS_3D) * (NLEV_SUBSET - 1)
        missing = set(self.arrays) - set(self.first)
        if missing or len(self.futures) != expected:
            raise SystemExit(f"incomplete store: {len(self.futures)} chunks, {missing}")
        zarr.consolidate_metadata(self.out_path)
        check = zarr.open_consolidated(self.out_path, mode="r")

        def read_back(name: str) -> None:
            got = check[name][(0, 0) if name in VARS_3D else ...]
            if not np.array_equal(np.asarray(got).ravel(), self.first[name]):
                raise SystemExit(f"{name} chunk read back from {self.out_path} differs")

        list(self.pool.map(read_back, self.arrays))
        validate_zarr(self.out_path, self.timestamp)


def _lower_priority() -> None:
    os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 10)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--timestamp", required=True, nargs="+", help="YYYY-MM-DDTHH:MM:SS (UTC)"
    )
    p.add_argument(
        "--tag",
        nargs="+",
        help="store tags, one per timestamp (default YYYYMMDDHH); the store is "
        "<out-dir>/scream_hiccup_ne1024pg2_<tag>.zarr",
    )
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
    p.add_argument("--era5-zarr", default=ARCO_ERA5)
    p.add_argument("--fetch-workers", type=int, default=8)
    p.add_argument("--write-workers", type=int, default=os.cpu_count())
    p.add_argument("--rows-per-chunk", type=int, default=8 * 1024 * 1024)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    argv = [sys.executable, "-m", "scripts.initial_conditions.fast_ic", *sys.argv[1:]]
    timestamps = [str(np.datetime64(t, "s")) for t in args.timestamp]
    tags = args.tag or [t.replace("-", "").replace("T", "")[:10] for t in timestamps]
    if len(tags) != len(timestamps):
        raise SystemExit("--tag needs one tag per --timestamp")
    if len(set(tags)) != len(tags):
        raise SystemExit(f"tags must be unique (one output store each): {tags}")
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda")
    start = clock = time.perf_counter()

    def stage(name: str) -> None:
        nonlocal clock
        now = time.perf_counter()
        log(f"  {name}: {now - clock:.1f} s")
        clock = now

    background = ThreadPoolExecutor(2)
    # the zstd threads would otherwise starve the thread launching GPU kernels
    writers = ThreadPoolExecutor(args.write_workers, initializer=_lower_priority)
    # finishing one store (its last chunks, consolidation and read-back) overlaps
    # the next IC's GPU work
    closer = ThreadPoolExecutor(1)
    closing: Future | None = None

    def finish(writer: ChunkWriter, out: str, gpu_done: float) -> None:
        writer.close()
        log(f"wrote {out} ({time.perf_counter() - gpu_done:.1f} s after its GPU stage)")

    def fetch(timestamp: str) -> Future:
        return background.submit(
            fetch_era5_arco, timestamp, args.era5_zarr, args.fetch_workers
        )

    # HDF5 is not thread-safe, so every NetCDF read stays on this thread; only
    # the ARCO fetch (Zarr over HTTPS) and coszr (numpy) run in the background.
    era5_next = fetch(timestamps[0])
    lat, lon, lat_deg, lon_deg = load_latlon(args.latlon)
    coszr_next = background.submit(compute_coszr, timestamps[0], lat_deg, lon_deg)
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
    if remap.shape[0] != NCOL:
        raise SystemExit(f"map has n_b={remap.shape[0]}, expected {NCOL}")
    vert, p0 = load_vert(args.vert)
    stage("load map and statics")

    for i, (timestamp, tag) in enumerate(zip(timestamps, tags)):
        log(f"{timestamp} -> {tag}")
        era5 = era5_next.result()
        if i + 1 < len(timestamps):
            era5_next = fetch(timestamps[i + 1])
        if remap.shape[1] != era5.sfc["ps"].size:
            raise SystemExit("map n_a does not match the ERA5 grid")
        coszr = coszr_next.result()
        if i + 1 < len(timestamps):
            coszr_next = background.submit(
                compute_coszr, timestamps[i + 1], lat_deg, lon_deg
            )
        stage("ERA5 read")

        out = os.path.join(args.out_dir, zarr_name(tag))
        writer = ChunkWriter(
            out,
            NCOL,
            timestamp,
            f"ERA5 {args.era5_zarr} via scripts.initial_conditions.fast_ic",
            history_entry(argv),
            writers,
        )
        for name, arr in (*statics.items(), ("coszr", coszr)):
            writer.submit(name, 0, arr)
        build_ic(era5, remap, phis_dst, vert, p0, args.rows_per_chunk, writer.submit)
        torch.cuda.synchronize()
        stage("GPU remap and adjustments")
        if closing is not None:
            closing.result()
        closing = closer.submit(finish, writer, out, clock)
    closing.result()
    background.shutdown()
    writers.shutdown()
    closer.shutdown()
    log(
        f"{len(timestamps)} IC(s) in {time.perf_counter() - start:.1f} s, peak GPU "
        f"memory {torch.cuda.max_memory_allocated() / 2**30:.0f} GiB"
    )


if __name__ == "__main__":
    main()
