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
"""Driver targets built while the run goes ("on the fly") instead of driver stores.

The replay needs the driver only as 8 x 8-pixel block means at its analysis hours, and
the anchored ring of a regional window only as the 8 x 8 block means of the window and
its ring at the driver hours (its ``smooth`` with ``--boundary-block 8``). A producer
process builds them from the source while STRATA runs:

    source state (ERA5 from ARCO, or an ECMWF HRES forecast: open data or WeatherBench2)
      -> fast_ic.build_ic on one GPU, column group by column group
      -> 8 x 8 block means of every replayed channel (float64)
      -> <out-dir>/blocks_<YYYYMMDDHH>.zarr

The fine state (9 GB for the globe) never goes to disk. A regional run's producer
builds only the columns of the window's tiles, and of its ring tiles when the ring is
anchored. Each rank of the run reads, on a background thread (``TargetReader``), the
blocks of its own tiles for ``Replay.add_block_target`` and the blocks of the window and
ring box for ``AnchoredBoundary.add_blocks``. The block means are those
``Replay.add_target`` and the ring take of a driver store built by ``fast_ic`` or
``hres_forecast`` from the same source state.

A target appears atomically: it is written to ``<name>.partial<pid>`` and renamed. It
records its ``origin`` (``target_origin``: the source state and the ``fast_ic``
inputs); a target already in the directory is used only if it was built from this
run's origin and holds its channels and tiles (``check_target``, in the producer and in
every reader). The run's own files are in a run directory (``--run-dir``, by default
the target directory; a run of ``run_screamcast_nudged.py`` uses
``<target dir>/runs/<run>``): the producer writes ``FAILED`` (with the traceback) if it
fails and ``DONE`` after its last target, its log, and the run's markers. A reader
waiting for a target stops on either, or after its timeout. Runs that share the target
directory do not see each other's run files.

Modes (``--driver-producer``):

- ``gated`` (default): the targets needed before the first analysis are built at once
  (during the run's start-up); the GPU stage of each later one starts when the run
  begins the last analysis before it needs that target. The rank hosting the producer
  empties its CUDA cache first, so the build does not meet the model's peak memory of
  start-up and compilation. A run without analyses (a window driven by the ring alone)
  is not gated.
- ``ahead``: every target as early as possible.
- ``serial``: each target only when the run asks for it (``request_<stamp>``), and the
  run waits: the slowest mode, for debugging.

In ``gated`` and ``ahead`` the next source state is fetched while the GPU builds the
current one, and the zarr write runs on its own thread.

Standalone (a dry run, or targets for later runs)::

    python -m screamcast.replay_targets produce --spec SPEC.json --out-dir DIR \\
        --run-init 2024-09-25T12 --hours 6 12 18 \\
        [--region 3,10,14,7,11 --tile-size 128 [--ring]]

    python -m screamcast.replay_targets compare A B

    python -m screamcast.replay_targets ic --spec SPEC.json --out-dir DIR \\
        --time 2024-09-25T12

``compare`` checks that two target stores hold the same values bit for bit. The spec
is JSON: ``source``, ``map`` (the 721 x 1440 -> ne1024pg2 map), ``phis`` (SCREAM phis
Zarr array) and ``vert`` (L128 coefficients), as for ``fast_ic``. The sources:

- ``era5``: hourly ERA5 from ARCO (optional ``era5_zarr``, ``fetch_workers``);
- ``hres``: the HRES forecast from ``init`` in ECMWF open data (00/12 UTC from
  2024-03-05 12 UTC; every 3 h to +144 h, then 6-hourly), GRIB files kept in
  ``cache`` (optional ``opendata_root``);
- ``wb2_hres``: the HRES forecast from ``init`` in WeatherBench2 (00/12 UTC from
  2016-01-01 to 2023-01-10 12 UTC; every 6 h to +240 h; optional ``wb2_zarr``,
  ``fetch_workers``).

``ic`` builds the run's initial condition (``ic_<YYYYMMDDHH>.zarr``, a store for
``--ic-store``, stamped with its origin like the targets); the spec then also names the
statics ``latlon`` and ``scream_ref``, as for ``fast_ic``. The runner builds it itself
with ``--ic-onthefly``.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import zarr

from screamcast.cubesphere_transforms import reorder_cubesphere_to_2d_tensor
from screamcast.distributed_halo import TileTopology, plane_parts
from screamcast.history import history_entry
from screamcast.initial_conditions.fast_ic import (
    ARCO_ERA5,
    ChunkWriter,
    Era5,
    _lower_priority,
    build_ic,
    fetch_era5_arco,
    load_map,
    load_vert,
)
from screamcast.initial_conditions.hres_ic import (
    WB2_HRES_FORECAST,
    WB2_LEADS,
    check_wb2_init,
    fetch_hres_forecast,
)
from screamcast.initial_conditions.ic_store import (
    compute_coszr,
    load_latlon,
    load_static_array,
)
from screamcast.initial_conditions.layout import NLEV_SUBSET, VARS_3D
from screamcast.regional import parse_region, ring_box
from screamcast.replay import BLOCK, ReplayStep, block_mean, channel_level

FAILED = "FAILED"
DONE = "DONE"
PRODUCER_LOG = "producer.log"
MODES = ("gated", "ahead", "serial")
SOURCES = ("era5", "hres", "wb2_hres")
SURFACE = ("T_2m", "ps")  # the 2D fields build_ic emits
SPEC_KEYS = {
    "common": ("source", "map", "phis", "vert"),
    "era5": (),
    "hres": ("init", "cache"),
    "wb2_hres": ("init",),
}
IC_KEYS = ("latlon", "scream_ref")  # the statics of an initial condition
IC_READY = "IC_READY"
# ECMWF open data, HRES "oper" (00 and 12 UTC runs): every 3 h to +144 h, then every
# 6 h to +240 h. The first complete cycle is hres_opendata.FIRST_INIT.
HRES_LEADS = frozenset((*range(0, 145, 3), *range(150, 241, 6)))
CHUNK_BLOCKS = 16  # store chunk side in blocks: one 128-pixel tile


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} [replay targets] {msg}", flush=True)


def hour_stamp(time_: np.datetime64) -> str:
    """``YYYYMMDDHH`` of a time."""
    stamp = np.datetime_as_string(np.datetime64(time_, "h"), unit="h")
    return stamp.replace("-", "").replace("T", "")


def target_path(out_dir: str, valid: np.datetime64) -> str:
    return os.path.join(out_dir, f"blocks_{hour_stamp(valid)}.zarr")


# ------------------------------------------------------------------ sources


def load_spec(path: str, ic: bool = False) -> dict:
    """The spec at ``path``; with ``ic``, also the keys an initial condition needs."""
    with open(path) as f:
        spec = json.load(f)
    source = spec.get("source")
    if source not in SOURCES:
        raise ValueError(f"{path}: source must be one of {SOURCES}, got {source!r}")
    keys = SPEC_KEYS["common"] + SPEC_KEYS[source] + (IC_KEYS if ic else ())
    missing = [k for k in keys if k not in spec]
    if missing:
        raise ValueError(f"{path}: the {source} spec lacks {missing}")
    return spec


def _with_retries(what: str, read: Callable[[], Era5], attempts: int = 3) -> Era5:
    """``read()``, retried after a failure (HTTP errors vary; a missing time is a
    SystemExit and is not retried)."""
    for attempt in range(1, attempts + 1):
        try:
            return read()
        except Exception as err:  # noqa: BLE001
            if attempt == attempts:
                raise
            log(f"{what}: {type(err).__name__} {err}; retry in {10 * attempt} s")
            time.sleep(10 * attempt)
    raise AssertionError("unreachable")


class Era5Source:
    """Hourly ERA5 from the ARCO Zarr store. A failed read is retried."""

    def __init__(self, spec: dict) -> None:
        self.zarr = spec.get("era5_zarr", ARCO_ERA5)
        self.workers = int(spec.get("fetch_workers", 8))

    def has(self, valid: np.datetime64) -> bool:
        return np.datetime64(valid, "h") == np.datetime64(valid, "s")

    def __call__(self, valid: np.datetime64) -> Era5:
        stamp = str(np.datetime64(valid, "s"))
        return _with_retries(
            f"ERA5 {stamp}", lambda: fetch_era5_arco(stamp, self.zarr, self.workers)
        )

    def describe(self, valid: np.datetime64) -> str:
        return f"ERA5 {self.zarr}"


class HresSource:
    """One ECMWF HRES forecast from the open-data bucket (``hres_opendata``)."""

    def __init__(self, spec: dict) -> None:
        from screamcast.initial_conditions.hres_opendata import (
            OPENDATA_ROOT,
            check_init,
        )

        self.init = np.datetime64(spec["init"], "h")
        self.cache = spec["cache"]
        self.root = spec.get("opendata_root", OPENDATA_ROOT)
        check_init(self.init)

    def lead(self, valid: np.datetime64) -> int:
        return int((np.datetime64(valid, "h") - self.init) / np.timedelta64(1, "h"))

    def has(self, valid: np.datetime64) -> bool:
        return (
            np.datetime64(valid, "h") == np.datetime64(valid, "s")
            and self.lead(valid) in HRES_LEADS
        )

    def __call__(self, valid: np.datetime64) -> Era5:
        from screamcast.initial_conditions.hres_opendata import (
            fetch_opendata,
            hres_state,
            read_opendata,
        )

        lead = self.lead(valid)
        path = fetch_opendata(self.init, lead, self.cache, self.root)
        return hres_state(read_opendata(path, self.init, lead))

    def describe(self, valid: np.datetime64) -> str:
        from screamcast.initial_conditions.hres_opendata import opendata_name

        return f"ECMWF HRES open data {self.root}/{opendata_name(self.init, self.lead(valid))}"


class Wb2HresSource:
    """One ECMWF HRES forecast from WeatherBench2's ``hres`` store
    (``hres_ic.fetch_hres_forecast``). A failed read is retried."""

    def __init__(self, spec: dict) -> None:
        self.init = np.datetime64(spec["init"], "h")
        self.zarr = spec.get("wb2_zarr", WB2_HRES_FORECAST)
        self.workers = int(spec.get("fetch_workers", 8))
        check_wb2_init(self.init)

    def lead(self, valid: np.datetime64) -> int:
        return int((np.datetime64(valid, "h") - self.init) / np.timedelta64(1, "h"))

    def has(self, valid: np.datetime64) -> bool:
        return (
            np.datetime64(valid, "h") == np.datetime64(valid, "s")
            and self.lead(valid) in WB2_LEADS
        )

    def __call__(self, valid: np.datetime64) -> Era5:
        lead = self.lead(valid)
        return _with_retries(
            f"HRES {self.init} +{lead} h",
            lambda: fetch_hres_forecast(str(self.init), lead, self.zarr, self.workers),
        )

    def describe(self, valid: np.datetime64) -> str:
        return f"ECMWF HRES {self.zarr} init {self.init} +{self.lead(valid)} h"


Source = Era5Source | HresSource | Wb2HresSource


def make_source(spec: dict) -> Source:
    return {"era5": Era5Source, "hres": HresSource, "wb2_hres": Wb2HresSource}[
        spec["source"]
    ](spec)


def target_origin(spec: dict) -> dict[str, str]:
    """What the targets of ``spec`` are built from: the source state (the ERA5 store, or
    the HRES forecast's init) and the ``fast_ic`` map, phis and vert (absolute paths).
    A target built from another origin is never used."""
    origin = {"source": spec["source"]}
    if spec["source"] == "era5":
        origin["era5_zarr"] = spec.get("era5_zarr", ARCO_ERA5)
    else:
        origin["init"] = str(np.datetime64(spec["init"], "h"))
    if spec["source"] == "wb2_hres":
        origin["wb2_zarr"] = spec.get("wb2_zarr", WB2_HRES_FORECAST)
    for key in ("map", "phis", "vert"):
        origin[key] = os.path.abspath(spec[key])
    return origin


def source_hours(source: Source, run_init: np.datetime64, hours) -> set[int]:
    """The run hours among ``hours`` at which the source has a state (the driver hours
    of ``replay_plan``)."""
    t0 = np.datetime64(run_init, "s")
    return {h for h in hours if source.has(t0 + np.timedelta64(h, "h"))}


def ring_hours(source, run_init: np.datetime64, every: int, end_h: float) -> list[int]:
    """Driver hours of an anchored ring: every ``every`` hours from 0 to the first at or
    after ``end_h`` (the time of the run's last step). The source must have them all."""
    if every <= 0:
        raise ValueError(f"ring driver interval must be positive, got {every}")
    hours = list(range(0, int(np.ceil(end_h / every)) * every + 1, every))
    missing = sorted(set(hours) - source_hours(source, run_init, hours))
    if missing:
        raise ValueError(
            f"the source has no state at ring driver hours {missing} (every {every} h)"
        )
    return hours


def target_schedule(
    plan: list[ReplayStep], ring: list[int] = ()
) -> list[tuple[int, int | None]]:
    """``(hour, gate)`` of every target in the order the run first reads it: the
    analysis hours of ``plan``, and the ``ring`` driver hours a window's steps
    interpolate between (the steps of the window ending at hour ``k`` need the ring
    hours up to the first at or after ``k``). ``gate`` is the last analysis the run
    begins before it reads the target (None: before the first analysis), the one a
    ``gated`` producer waits for."""
    ring = sorted(ring)
    gates: dict[int, int | None] = {}
    last = None

    def need_ring(upto: int) -> None:
        for h in ring:
            gates.setdefault(h, last)
            if h >= upto:
                break

    if ring:
        need_ring(0)
    for step in plan:
        if step.kind == "window" and ring:
            need_ring(step.hour)
        elif step.kind == "analysis":
            gates.setdefault(step.hour, last)
            last = step.hour
    return list(gates.items())


# ------------------------------------------------------------------ build


def target_channels(stems: list[str]) -> list[str]:
    """Channel names of the replayed variables, as the model names its inputs:
    ``<var>_<j>`` for the levels of a 3D variable, the bare name for a surface one."""
    out = []
    for stem in stems:
        if stem in VARS_3D:
            out += [f"{stem}_{j}" for j in range(NLEV_SUBSET)]
        elif stem in SURFACE:
            out.append(stem)
        else:
            raise ValueError(f"{stem} is not a variable build_ic produces")
    return out


def producer_stems(replay_vars: list[str], ring_vars: list[str]) -> list[str]:
    """The variables a producer builds, whole: those of the replayed channels
    (``--replay-vars``, bare or per level like ``U_24``) and of the ring's channels."""
    return sorted({channel_level(name)[0] for name in [*replay_vars, *ring_vars]})


def tile_columns(tiles, tile_size: int, face_size: int) -> np.ndarray:
    """Flat (ne x ne pg2) column indices of the pixels of ``tiles``, tile by tile, each
    row by row: the layout of ``topology.faces_to_local_tiles``."""
    grid = reorder_cubesphere_to_2d_tensor(
        torch.arange(face_size * face_size), ne=face_size // 2, npg=2
    ).numpy()
    ts = tile_size
    return np.concatenate(
        [
            grid[ti * ts : (ti + 1) * ts, tj * ts : (tj + 1) * ts].reshape(-1)
            + face * face_size * face_size
            for face, ti, tj in tiles
        ]
    )


def csr_rows(matrix: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """Rows ``rows`` of a CSR matrix, in that order, as a CSR matrix; each row keeps its
    entries in their order, so its sum is that of the whole matrix."""
    crow, col, val = matrix.crow_indices(), matrix.col_indices(), matrix.values()
    start = crow[rows]
    count = crow[rows + 1] - start
    new_crow = torch.zeros(len(rows) + 1, dtype=crow.dtype, device=crow.device)
    new_crow[1:] = torch.cumsum(count, 0)
    offset = torch.repeat_interleave(start - new_crow[:-1], count)
    idx = offset + torch.arange(int(new_crow[-1]), dtype=crow.dtype, device=crow.device)
    return torch.sparse_csr_tensor(
        new_crow, col[idx], val[idx], size=(len(rows), matrix.shape[1])
    )


class BlockBuilder:
    """``fast_ic.build_ic`` on the columns of ``tiles``, reduced to block means.

    ``build_ic`` works column by column (the map's rows, then each column's physics),
    so it runs on groups of at most ``group_columns`` columns (default 2^21, 1/12 of the
    globe): the GPU holds the map rows, the source and one group, not the 9 GB state.
    Each block mean is ``replay.block_mean`` of the float32 values ``build_ic`` emits,
    as ``Replay.add_target`` takes them from a store.
    """

    def __init__(
        self,
        *,
        remap: torch.Tensor,
        phis: torch.Tensor,
        vert: dict,
        p0: float,
        tiles,
        tile_size: int,
        face_size: int,
        channels: list[str],
        group_columns: int = 1 << 21,
    ) -> None:
        if tile_size % BLOCK:
            raise ValueError(f"tile size {tile_size} is not a multiple of {BLOCK}")
        self.tile_size = tile_size
        self.n_tiles = len(tiles)
        self.vert, self.p0 = vert, p0
        self.channels = list(channels)
        self.index = {name: i for i, name in enumerate(self.channels)}
        columns = torch.from_numpy(tile_columns(tiles, tile_size, face_size))
        per_tile = tile_size * tile_size
        group = max(1, group_columns // per_tile)
        self.groups = []
        for t0 in range(0, self.n_tiles, group):
            t1 = min(t0 + group, self.n_tiles)
            rows = columns[t0 * per_tile : t1 * per_tile].to(remap.device)
            self.groups.append((t0, t1, csr_rows(remap, rows), phis[rows]))

    def __call__(self, state: Era5) -> np.ndarray:
        """Block means ``(channels, tiles, tile/8, tile/8)`` float64 of ``state``."""
        tb = self.tile_size // BLOCK
        out = np.full((len(self.channels), self.n_tiles, tb, tb), np.nan)
        for t0, t1, rows, phis in self.groups:

            def emit(name: str, j: int, field: torch.Tensor, t0=t0, t1=t1) -> None:
                row = self.index.get(name if name in SURFACE else f"{name}_{j}")
                if row is not None:
                    tiles = field.double().reshape(t1 - t0, self.tile_size, -1)
                    out[row, t0:t1] = block_mean(tiles, BLOCK).cpu().numpy()

            if rows.shape[1] != state.sfc["ps"].size:
                raise ValueError(
                    f"map columns {rows.shape[1]} != source points {state.sfc['ps'].size}"
                )
            build_ic(state, rows, phis, self.vert, self.p0, rows.shape[0], emit)
        if not np.isfinite(out).all():
            raise ValueError("build_ic left non-finite block means")
        return out


# ------------------------------------------------------------------ stores


def write_target(
    path: str,
    blocks: np.ndarray,
    *,
    channels: list[str],
    tiles,
    tile_size: int,
    face_size: int,
    valid: np.datetime64,
    history: str,
    source: str,
    origin: dict[str, str],
) -> None:
    """Store ``(channels, tiles, tb, tb)`` block means at ``path`` as one
    ``(channels, 6, face/8, face/8)`` array, NaN off ``tiles``. Written beside ``path``
    and renamed into place, so ``path`` only ever exists complete. If another producer
    (another run sharing the directory) renamed its store there first, that one stays:
    the readers check it like any target they did not build."""
    tmp = f"{path}.partial{os.getpid()}"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    nb = face_size // BLOCK
    chunk = min(CHUNK_BLOCKS, nb)
    group = zarr.open_group(tmp, mode="w")
    array = group.create_array(
        "blocks",
        shape=(len(channels), 6, nb, nb),
        chunks=(len(channels), 1, chunk, chunk),
        dtype="float64",
        fill_value=np.nan,
    )
    tb = tile_size // BLOCK
    for k, (face, ti, tj) in enumerate(tiles):
        array[:, face, ti * tb : (ti + 1) * tb, tj * tb : (tj + 1) * tb] = blocks[:, k]
    group.attrs.update(
        {
            "variables": list(channels),
            "valid_time": str(np.datetime64(valid, "s")),
            "tiles": [[int(v) for v in t] for t in tiles],
            "tile_size": int(tile_size),
            "block": BLOCK,
            "source": source,
            "origin": dict(origin),
            "history": history,
        }
    )
    try:
        os.rename(tmp, path)
    except OSError:
        if not os.path.isdir(path):
            raise
        shutil.rmtree(tmp)


def read_blocks(path: str, channels: list[str], tiles, tile_size: int) -> np.ndarray:
    """Block means ``(channels, tiles, tb, tb)`` of ``tiles`` from a target store."""
    group = zarr.open_group(path, mode="r")
    have = list(group.attrs["variables"])
    missing = [c for c in channels if c not in have]
    if missing:
        raise ValueError(f"{path} lacks {missing}")
    idx = [have.index(c) for c in channels]
    array = group["blocks"]
    tb = tile_size // BLOCK
    return np.stack(
        [
            array[:, face, ti * tb : (ti + 1) * tb, tj * tb : (tj + 1) * tb][idx]
            for face, ti, tj in tiles
        ],
        axis=1,
    )


def read_box_blocks(
    path: str, channels: list[str], topology: TileTopology
) -> np.ndarray:
    """Block means ``(channels, H/8, W/8)`` of a regional run's window and ring box
    (``regional.ring_box``), in the plane's orientation, from a target store."""
    group = zarr.open_group(path, mode="r")
    have = list(group.attrs["variables"])
    missing = [c for c in channels if c not in have]
    if missing:
        raise ValueError(f"{path} lacks {missing}")
    idx = [have.index(c) for c in channels]
    array = group["blocks"]
    (r0, c0), (h, w) = ring_box(topology)
    box = (r0 // BLOCK, (r0 + h) // BLOCK, c0 // BLOCK, (c0 + w) // BLOCK)
    out = np.full((len(channels), h // BLOCK, w // BLOCK), np.nan)
    nb = topology.face_size // BLOCK
    for face, (fr0, fr1, fc0, fc1), (pr0, pr1, pc0, pc1), k in plane_parts(
        topology.region_plane, box, nb
    ):
        part = np.asarray(array[:, face, fr0:fr1, fc0:fc1])[idx]
        out[:, pr0 - box[0] : pr1 - box[0], pc0 - box[2] : pc1 - box[2]] = np.rot90(
            part, k, axes=(-2, -1)
        )
    return out


def _block_set(tiles, tile_size: int) -> set[tuple[int, int, int]]:
    tb = tile_size // BLOCK
    return {
        (face, ti * tb + a, tj * tb + b)
        for face, ti, tj in tiles
        for a in range(tb)
        for b in range(tb)
    }


def check_target(
    path: str,
    *,
    channels: list[str],
    tiles,
    tile_size: int,
    origin: dict[str, str],
) -> None:
    """Raise unless the target store at ``path`` was built from ``origin`` and holds
    ``channels`` on every block of ``tiles`` (the producer's tiles, the same on every
    rank, so every rank reaches the same verdict)."""
    attrs = zarr.open_group(path, mode="r").attrs
    why = None
    built_from = attrs.get("origin")
    if built_from != origin:
        why = f"built from {built_from}, this run's targets from {origin}"
    elif not set(channels) <= set(attrs["variables"]):
        why = f"lacks channels {sorted(set(channels) - set(attrs['variables']))}"
    elif not _block_set(tiles, tile_size) <= _block_set(
        [tuple(t) for t in attrs["tiles"]], int(attrs["tile_size"])
    ):
        why = "does not cover this run's tiles"
    if why is not None:
        raise ValueError(
            f"{path} cannot serve this run ({why}): remove it or use another "
            "target directory"
        )


def compare_stores(a: str, b: str) -> list[str]:
    """Differences between two target stores: variables, or values bit for bit (NaN
    equal to NaN)."""
    ga, gb = zarr.open_group(a, mode="r"), zarr.open_group(b, mode="r")
    if list(ga.attrs["variables"]) != list(gb.attrs["variables"]):
        return ["variables differ"]
    x, y = np.asarray(ga["blocks"][...]), np.asarray(gb["blocks"][...])
    if x.shape != y.shape:
        return [f"shape {x.shape} vs {y.shape}"]
    if np.array_equal(x, y, equal_nan=True):
        return []
    d = np.abs(x - y)
    return [f"max |a - b| {np.nanmax(d):.3g}, {int((d > 0).sum())} values differ"]


# ------------------------------------------------------------------ initial condition


def ic_path(out_dir: str, valid: np.datetime64) -> str:
    return os.path.join(out_dir, f"ic_{hour_stamp(valid)}.zarr")


def ic_origin(spec: dict, valid: np.datetime64) -> dict[str, str]:
    """What the initial condition at ``valid`` is built from: the targets' origin, its
    time and the statics (absolute paths)."""
    origin = target_origin(spec)
    origin["valid_time"] = str(np.datetime64(valid, "s"))
    for key in IC_KEYS:
        origin[key] = os.path.abspath(spec[key])
    return origin


def check_ic(path: str, origin: dict[str, str]) -> None:
    """Raise unless the store at ``path`` is an initial condition built from ``origin``."""
    built_from = zarr.open_group(path, mode="r").attrs.get("origin")
    if built_from != origin:
        raise ValueError(
            f"{path} cannot serve this run (built from {built_from}, this run's "
            f"initial condition from {origin}): remove it or use another target "
            "directory"
        )


def build_ic_store(
    spec: dict,
    valid: np.datetime64,
    out_dir: str,
    *,
    device: torch.device,
    history: str,
    write_workers: int = 32,
    rows_per_chunk: int = 8 * 1024 * 1024,
) -> str:
    """The one-frame initial condition at ``valid`` from ``spec``'s source, as
    ``fast_ic`` writes it (a store for ``--ic-store``): its path in ``out_dir``. A store
    already there is kept if it was built from the same origin (``ic_origin``), and
    stops the build otherwise. A new store is written beside its path, stamped with its
    origin and renamed into place; if another run renamed one there first, that one is
    checked and kept."""
    path = ic_path(out_dir, valid)
    origin = ic_origin(spec, valid)
    if os.path.isdir(path):
        check_ic(path, origin)
        log(f"{path} exists, kept")
        return path
    source = make_source(spec)
    if not source.has(valid):
        raise ValueError(f"{spec['source']} has no state at {valid}")
    stamp = str(np.datetime64(valid, "s"))
    start = time.perf_counter()
    state = source(valid)
    t_source = time.perf_counter() - start
    lat, lon, lat_deg, lon_deg = load_latlon(spec["latlon"])
    statics = {
        "phis": np.asarray(zarr.open_array(spec["phis"], mode="r")[:], np.float32),
        "lat": lat,
        "lon": lon,
        "sgh30": load_static_array(spec["scream_ref"], "sgh30"),
        "landfrac": load_static_array(spec["scream_ref"], "landfrac"),
    }
    remap = load_map(spec["map"], device)
    ncol = remap.shape[0]
    if remap.shape[1] != state.sfc["ps"].size:
        raise ValueError(
            f"map columns {remap.shape[1]} != source points {state.sfc['ps'].size}"
        )
    for name, arr in statics.items():
        if arr.shape != (ncol,):
            raise ValueError(f"{name} shape {arr.shape}, the map has {ncol} rows")
    vert, p0 = load_vert(spec["vert"])
    os.makedirs(out_dir, exist_ok=True)
    tmp = f"{path}.partial{os.getpid()}"
    with ThreadPoolExecutor(write_workers, initializer=_lower_priority) as writers:
        writer = ChunkWriter(
            tmp,
            ncol,
            stamp,
            f"{source.describe(valid)} via screamcast.replay_targets",
            history,
            writers,
        )
        zarr.open_group(tmp, mode="r+").attrs["origin"] = origin
        for name, arr in (
            *statics.items(),
            ("coszr", compute_coszr(stamp, lat_deg, lon_deg)),
        ):
            writer.submit(name, 0, arr)
        phis = torch.from_numpy(statics["phis"]).to(device)
        build_ic(state, remap, phis, vert, p0, rows_per_chunk, writer.submit)
        if device.type == "cuda":
            torch.cuda.synchronize()
        writer.close()
    try:
        os.rename(tmp, path)
    except OSError:
        if not os.path.isdir(path):
            raise
        shutil.rmtree(tmp)
        check_ic(path, origin)
    log(
        f"{path}: source {t_source:.1f} s, "
        f"total {time.perf_counter() - start:.1f} s"
    )
    return path


def wait_for_ic(run_dir: str, timeout: float | None, poll: float = 0.5) -> str:
    """The initial condition's path once the rank building it has written ``IC_READY``
    in ``run_dir``; raises at once if it wrote ``FAILED``, and after ``timeout``
    seconds unless it is None."""
    ready = os.path.join(run_dir, IC_READY)
    failed = os.path.join(run_dir, FAILED)
    start = time.perf_counter()
    while not os.path.exists(ready):
        if os.path.exists(failed):
            with open(failed) as f:
                raise RuntimeError(f"initial condition build failed:\n{f.read()}")
        if timeout is not None and time.perf_counter() - start > timeout:
            raise TimeoutError(f"no initial condition after {timeout:.0f} s")
        time.sleep(poll)
    with open(ready) as f:
        return f.read()


# ------------------------------------------------------------------ markers


def marker_path(run_dir: str, kind: str, valid: np.datetime64) -> str:
    return os.path.join(run_dir, f"{kind}_{hour_stamp(valid)}")


def mark(run_dir: str, kind: str, valid: np.datetime64) -> None:
    """Tell the producer that the run begins the analysis at (``analysis``) or waits for
    the target of (``request``) ``valid``."""
    open(marker_path(run_dir, kind, valid), "w").close()


def wait_for_markers(
    paths: list[str], check: Callable[[], None], poll: float = 0.2
) -> None:
    """Wait until one of ``paths`` exists; ``check()`` runs at each poll and raises to
    stop waiting."""
    while not any(os.path.exists(p) for p in paths):
        check()
        time.sleep(poll)


def wait_for_target(
    out_dir: str,
    valid: np.datetime64,
    timeout: float | None,
    poll: float = 0.5,
    stop: threading.Event | None = None,
    run_dir: str | None = None,
) -> float:
    """Block until the target of ``valid`` exists; the seconds waited. Raises at once
    if the producer wrote ``FAILED`` or finished (``DONE``) without it in ``run_dir``
    (default ``out_dir``), or if ``stop`` is set; after ``timeout`` seconds unless it is
    None."""
    path = target_path(out_dir, valid)
    run_dir = out_dir if run_dir is None else run_dir
    start = time.perf_counter()
    while not os.path.isdir(path):
        failed = os.path.join(run_dir, FAILED)
        if os.path.exists(failed):
            with open(failed) as f:
                raise RuntimeError(f"replay target producer failed:\n{f.read()}")
        if os.path.exists(os.path.join(run_dir, DONE)):
            raise RuntimeError(f"replay target producer finished without {path}")
        if timeout is not None and time.perf_counter() - start > timeout:
            raise TimeoutError(f"no replay target {path} after {timeout:.0f} s")
        if stop is not None and stop.wait(poll):
            raise RuntimeError(f"stopped waiting for replay target {path}")
        if stop is None:
            time.sleep(poll)
    return time.perf_counter() - start


# ------------------------------------------------------------------ producer


def producer_tiles(
    region: str | None, tile_size: int, face_size: int, ring: bool = False
) -> list:
    """The tiles a producer builds: the window's (and its ring's, for an anchored
    ring), or every tile of the cube."""
    n = face_size // tile_size
    if not region:
        if ring:
            raise ValueError("a ring needs a --region window")
        return [(f, i, j) for f in range(6) for i in range(n) for j in range(n)]
    window = parse_region(region, n)
    if not ring:
        return list(window)
    topology = TileTopology(
        world_size=1,
        rank=0,
        face_size=face_size,
        tile_size=tile_size,
        halo_width=0,
        active_tiles=window,
    )
    return [*window, *topology.ring_tiles]


def produce(args: argparse.Namespace) -> None:
    out_dir = args.out_dir
    run_dir = args.run_dir or out_dir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    for name in (FAILED, DONE):
        if os.path.exists(os.path.join(run_dir, name)):
            os.remove(os.path.join(run_dir, name))
    history = history_entry(
        [sys.executable, "-m", "screamcast.replay_targets", *sys.argv[1:]]
    )
    start = time.perf_counter()
    parent = os.getppid()
    try:
        spec = load_spec(args.spec)
        source = make_source(spec)
        origin = target_origin(spec)
        run_init = np.datetime64(args.run_init, "s")
        valids = [run_init + np.timedelta64(h, "h") for h in args.hours]
        gates = (
            [None if g < 0 else g for g in args.gates]
            if args.gates
            else [None, *args.hours[:-1]]
        )
        if len(gates) != len(valids):
            raise ValueError(f"{len(gates)} gates for {len(valids)} targets")
        missing = [str(v) for v in valids if not source.has(v)]
        if missing:
            raise ValueError(f"{spec['source']} has no state at {missing}")
        tiles = producer_tiles(args.region, args.tile_size, args.face_size, args.ring)
        channels = target_channels(args.vars.split(","))
        todo = []
        for v in valids:
            path = target_path(out_dir, v)
            if not os.path.isdir(path):
                todo.append(v)
            else:
                check_target(
                    path,
                    channels=channels,
                    tiles=tiles,
                    tile_size=args.tile_size,
                    origin=origin,
                )
                log(f"{hour_stamp(v)} exists, kept")
        device = torch.device(args.device)
        on_cuda = device.type == "cuda"
        phis = np.asarray(zarr.open_array(spec["phis"], mode="r")[:], np.float32)
        vert, p0 = load_vert(spec["vert"])
        builder = BlockBuilder(
            remap=load_map(spec["map"], device),
            phis=torch.from_numpy(phis).to(device),
            vert=vert,
            p0=p0,
            tiles=tiles,
            tile_size=args.tile_size,
            face_size=args.face_size,
            channels=channels,
            group_columns=args.group_columns,
        )
        if on_cuda:
            torch.cuda.empty_cache()  # the whole map is gone; its rows stay
        log(
            f"{spec['source']}: {len(todo)} of {len(valids)} target(s) to build "
            f"{[hour_stamp(v) for v in todo]} -> {out_dir}, mode {args.mode}, "
            f"{len(tiles)} tiles of {args.tile_size}, {len(channels)} channels, "
            f"ready in {time.perf_counter() - start:.1f} s"
        )
        fetcher = ThreadPoolExecutor(1)
        writer = ThreadPoolExecutor(1)
        pending: list[Future] = []
        prefetch = args.mode != "serial"

        def check() -> None:
            """Raise a failed write, or stop if the run that started us is gone."""
            for future in pending:
                if future.done():
                    future.result()
            if os.getppid() != parent:
                raise RuntimeError("the process that started the producer has exited")

        def gate(k: int) -> None:
            if args.mode == "gated" and gates[k] is not None:
                wait_for_markers(
                    [
                        marker_path(
                            run_dir,
                            "analysis",
                            run_init + np.timedelta64(gates[k], "h"),
                        ),
                        marker_path(run_dir, "request", valids[k]),
                    ],
                    check,
                )

        def build(state: Era5, k: int) -> np.ndarray:
            """The GPU stage, retried when the GPU is idle if it runs out of memory."""
            for attempt in range(3):
                try:
                    return builder(state)
                except torch.OutOfMemoryError as err:
                    torch.cuda.empty_cache()
                    if attempt == 2:
                        raise
                    log(f"{hour_stamp(valids[k])}: {str(err)[:120]}; retry when idle")
                    if args.mode == "ahead":
                        time.sleep(30)
                    else:
                        wait_for_markers(
                            [
                                marker_path(run_dir, "analysis", valids[k]),
                                marker_path(run_dir, "request", valids[k]),
                            ],
                            check,
                        )
            raise AssertionError("unreachable")

        def finish(valid: np.datetime64, blocks: np.ndarray, t_ready: float) -> None:
            t = time.perf_counter()
            write_target(
                target_path(out_dir, valid),
                blocks,
                channels=channels,
                tiles=tiles,
                tile_size=args.tile_size,
                face_size=args.face_size,
                valid=valid,
                history=history,
                source=source.describe(valid),
                origin=origin,
            )
            log(
                f"{hour_stamp(valid)} written at {time.perf_counter() - start:.1f} s "
                f"(write {time.perf_counter() - t:.1f} s, built {t_ready:.1f} s)"
            )

        nxt = fetcher.submit(source, todo[0]) if todo and prefetch else None
        for i, valid in enumerate(todo):
            k = valids.index(valid)
            check()
            t = time.perf_counter()
            if not prefetch:
                wait_for_markers([marker_path(run_dir, "request", valid)], check)
                state = source(valid)
            else:
                state = nxt.result()
                if i + 1 < len(todo):
                    nxt = fetcher.submit(source, todo[i + 1])
            t_source = time.perf_counter() - t
            t = time.perf_counter()
            gate(k)
            t_gate = time.perf_counter() - t
            t = time.perf_counter()
            blocks = build(state, k)
            del state
            if on_cuda:
                torch.cuda.empty_cache()  # hold only the map rows between targets
            log(
                f"{hour_stamp(valid)}: source {t_source:.1f} s, gate {t_gate:.1f} s, "
                f"build {time.perf_counter() - t:.1f} s"
            )
            pending.append(
                writer.submit(finish, valid, blocks, time.perf_counter() - start)
            )
            if not prefetch:
                pending[-1].result()
        for future in pending:
            future.result()
        fetcher.shutdown()
        writer.shutdown()
        open(os.path.join(run_dir, DONE), "w").close()
        log(f"done: {len(todo)} target(s) in {time.perf_counter() - start:.1f} s")
    except BaseException:
        with open(os.path.join(run_dir, FAILED), "w") as f:
            f.write(traceback.format_exc())
        raise


class ProducerProcess:
    """The producer as a child process of one rank, on that rank's GPU, in the run's
    working directory (relative paths mean what they mean to the run). A monitor thread
    writes ``FAILED`` if the child dies without doing so, so ranks waiting for a target
    stop instead of timing out. ``FAILED``, ``DONE``, the markers and ``producer.log``
    are in ``run_dir``."""

    def __init__(
        self,
        *,
        spec: str,
        out_dir: str,
        run_dir: str,
        run_init: np.datetime64,
        hours: list[int],
        gates: list[int | None],
        mode: str,
        stems: list[str],
        tile_size: int,
        face_size: int,
        region: str | None,
        ring: bool,
        gpu: str | None,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"producer mode must be one of {MODES}, got {mode!r}")
        self.run_dir = run_dir
        cmd = [
            sys.executable,
            "-P",  # import screamcast from PYTHONPATH below, not from the run's cwd
            "-m",
            "screamcast.replay_targets",
            "produce",
            "--spec",
            spec,
            "--out-dir",
            out_dir,
            "--run-dir",
            run_dir,
            "--run-init",
            str(np.datetime64(run_init, "s")),
            "--mode",
            mode,
            "--vars",
            ",".join(stems),
            "--tile-size",
            str(tile_size),
            "--face-size",
            str(face_size),
            "--hours",
            *map(str, hours),
            "--gates",
            *(str(-1 if g is None else g) for g in gates),
        ]
        if region:
            cmd += ["--region", region]
        if ring:
            cmd.append("--ring")
        env = dict(os.environ)
        # a standalone process: none of the rank's launcher or torch.distributed identity
        for key in (
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
            "LOCAL_WORLD_SIZE",
            "GROUP_RANK",
            "ROLE_RANK",
            "ROLE_WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
            "TORCHELASTIC_RUN_ID",
            "PMIX_RANK",
            "SLURM_PROCID",
            "SLURM_LOCALID",
            "OMPI_COMM_WORLD_RANK",
            "OMPI_COMM_WORLD_SIZE",
            "OMPI_COMM_WORLD_LOCAL_RANK",
        ):
            env.pop(key, None)
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = gpu
        env["PYTHONUNBUFFERED"] = "1"
        root = str(Path(__file__).resolve().parents[1])
        env["PYTHONPATH"] = os.pathsep.join(
            [root, *filter(None, [env.get("PYTHONPATH")])]
        )
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(run_dir, exist_ok=True)
        # an earlier verdict and markers in this run directory must not reach this run
        for name in os.listdir(run_dir):
            if name in (FAILED, DONE) or name.startswith(("analysis_", "request_")):
                os.remove(os.path.join(run_dir, name))
        self.log = open(os.path.join(run_dir, PRODUCER_LOG), "a")
        self.log.write(f"# gpu {gpu}: {' '.join(cmd)}\n")
        self.log.flush()
        self.proc = subprocess.Popen(
            cmd,  # noqa: S603 (our own interpreter and module, arguments from the run)
            env=env,
            stdout=self.log,
            stderr=subprocess.STDOUT,
        )
        self.monitor = threading.Thread(target=self._monitor, daemon=True)
        self.monitor.start()

    def _monitor(self) -> None:
        code = self.proc.wait()
        failed = os.path.join(self.run_dir, FAILED)
        if code != 0 and not os.path.exists(failed):
            with open(failed, "w") as f:
                f.write(f"producer exited with code {code} (see {PRODUCER_LOG})\n")

    def close(self, timeout: float = 120.0) -> int:
        """Wait for the producer to finish, then stop it if it has not."""
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.log.close()
        return self.proc.returncode


# ------------------------------------------------------------------ run side


class TargetReader:
    """A rank's reader of the targets, ``read(path)`` of each (its own tiles' blocks for
    the replay, the window and ring box for the ring), on a background thread
    (``prefetch``) so the run does not wait on the file system. With a ``serial``
    producer nothing is prefetched: ``get`` asks for each target.

    ``timeout`` counts from the ``get`` call: a prefetched target may take as long as
    the run takes to need it. The background thread is a daemon and stops waiting at
    ``close``, so a run that fails does not wait for targets on its way out. Each
    target passes ``check`` (``check_target``) before it is read, also one the producer
    kept from an earlier run. ``FAILED``, ``DONE`` and the requests are in
    ``run_dir``."""

    def __init__(
        self,
        out_dir: str,
        run_dir: str,
        run_init: np.datetime64,
        read: Callable[[str], np.ndarray],
        *,
        check: Callable[[str], None],
        mode: str,
        timeout: float,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"producer mode must be one of {MODES}, got {mode!r}")
        self.out_dir = out_dir
        self.run_dir = run_dir
        self.check = check
        self.run_init = np.datetime64(run_init, "s")
        self.read = read
        self.mode = mode
        self.timeout = timeout
        self.stop = threading.Event()
        self.queue: queue.SimpleQueue[tuple[int, Future]] = queue.SimpleQueue()
        self.worker: threading.Thread | None = None
        self.futures: dict[int, Future] = {}

    def valid(self, hour: int) -> np.datetime64:
        return self.run_init + np.timedelta64(hour, "h")

    def _load(self, hour: int, timeout: float | None) -> tuple[np.ndarray, dict]:
        waited = wait_for_target(
            self.out_dir,
            self.valid(hour),
            timeout,
            stop=self.stop,
            run_dir=self.run_dir,
        )
        t = time.perf_counter()
        path = target_path(self.out_dir, self.valid(hour))
        self.check(path)
        blocks = self.read(path)
        return blocks, {"waited_s": waited, "read_s": time.perf_counter() - t}

    def _work(self) -> None:
        while not self.stop.is_set():
            hour, future = self.queue.get()
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(self._load(hour, None))
            except BaseException as e:  # noqa: BLE001 (handed to the caller of get)
                future.set_exception(e)

    def prefetch(self, hours: list[int]) -> None:
        if self.mode == "serial":
            return
        if self.worker is None:
            self.worker = threading.Thread(target=self._work, daemon=True)
            self.worker.start()
        for hour in hours:
            if hour not in self.futures:
                self.futures[hour] = Future()
                self.queue.put((hour, self.futures[hour]))

    def get(self, hour: int) -> tuple[np.ndarray, dict]:
        """``read`` of the target at ``+hour`` and timings, with ``blocked_s`` the
        seconds the caller waited here (at most ``timeout``)."""
        start = time.perf_counter()
        if not os.path.isdir(target_path(self.out_dir, self.valid(hour))):
            mark(self.run_dir, "request", self.valid(hour))
        future = self.futures.pop(hour, None)
        if future is None:
            blocks, info = self._load(hour, self.timeout)
        else:
            try:
                blocks, info = future.result(timeout=self.timeout)
            except TimeoutError:
                raise TimeoutError(
                    f"no replay target "
                    f"{target_path(self.out_dir, self.valid(hour))} after "
                    f"{self.timeout:.0f} s"
                ) from None
        return blocks, {**info, "blocked_s": time.perf_counter() - start}

    def close(self) -> None:
        self.stop.set()
        for future in self.futures.values():
            future.cancel()


# ------------------------------------------------------------------ CLI


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare", help="bitwise comparison of two target stores")
    c.add_argument("a")
    c.add_argument("b")
    q = sub.add_parser("produce", help="build the targets of a run")
    q.add_argument("--spec", required=True, help="source spec (JSON)")
    q.add_argument("--out-dir", required=True, help="the target directory")
    q.add_argument(
        "--run-dir",
        default=None,
        help="FAILED, DONE and the run's markers (default: --out-dir)",
    )
    q.add_argument("--run-init", required=True, help="run start, YYYY-MM-DDTHH (UTC)")
    q.add_argument("--hours", required=True, nargs="+", type=int, help="run hours")
    q.add_argument(
        "--gates",
        type=int,
        nargs="+",
        help="per target, the analysis hour a gated producer waits for (-1: none); "
        "default the previous target's",
    )
    q.add_argument("--mode", choices=MODES, default="ahead")
    q.add_argument(
        "--vars",
        default="U,V,PotentialTemperature,qv,z_mid,omega,T_2m",
        help="replayed variables (all levels of the 3D ones)",
    )
    q.add_argument("--region", default=None, help="build only this --region window")
    q.add_argument(
        "--ring",
        action="store_true",
        help="also the window's ring tiles (anchored ring)",
    )
    q.add_argument("--tile-size", type=int, default=128)
    q.add_argument("--face-size", type=int, default=2048)
    q.add_argument("--group-columns", type=int, default=1 << 21)
    q.add_argument("--device", default="cuda")
    i = sub.add_parser("ic", help="build the initial condition of a run")
    i.add_argument("--spec", required=True, help="source spec (JSON) with the statics")
    i.add_argument("--out-dir", required=True, help="the target directory")
    i.add_argument("--time", required=True, help="YYYY-MM-DDTHH (UTC)")
    i.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.cmd == "compare":
        bad = compare_stores(args.a, args.b)
        for line in bad:
            print(line)
        if bad:
            raise SystemExit(f"TARGETS_DIFFER {args.a} {args.b}")
        print(f"TARGETS_EQUAL {args.a} {args.b}")
    elif args.cmd == "ic":
        argv = [sys.executable, "-m", "screamcast.replay_targets", *sys.argv[1:]]
        build_ic_store(
            load_spec(args.spec, ic=True),
            np.datetime64(args.time, "h"),
            args.out_dir,
            device=torch.device(args.device),
            history=history_entry(argv),
        )
    else:
        produce(args)


if __name__ == "__main__":
    main()
