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
"""Replay targets built while the run goes ("on the fly") instead of driver stores.

A replay needs the driver only as 8 x 8-pixel block means at its analysis hours. A
producer process builds them from the source while STRATA runs:

    source state (ERA5 from ARCO, or ECMWF HRES open data)
      -> fast_ic.build_ic on one GPU, column group by column group
      -> 8 x 8 block means of every replayed channel (float64)
      -> <out-dir>/blocks_<YYYYMMDDHH>.zarr

The fine state (9 GB for the globe) never goes to disk. A regional run's producer
builds only the columns of the window's tiles. Each rank of the run reads the blocks of
its own tiles (``TargetReader``, in a background thread so the analysis does not wait on
the read) and hands them to ``Replay.add_block_target``. The block means are those
``Replay.add_target`` takes of a driver store built by ``fast_ic`` or ``hres_forecast``
from the same source state.

A target appears atomically: it is written to ``<name>.partial<pid>`` and renamed. The
producer writes ``FAILED`` (with the traceback) if it fails and ``DONE`` after its last
target; a reader waiting for a target stops on either, or after its timeout.

Modes (``--replay-producer``):

- ``gated`` (default): the first target is built at once (during the run's start-up);
  the GPU stage of each later one starts when the run begins the analysis before it.
  The rank hosting the producer empties its CUDA cache first, so the build does not
  meet the model's peak memory of start-up and compilation.
- ``ahead``: every target as early as possible.
- ``serial``: each target only when the run asks for it (``request_<stamp>``), and the
  run waits: the slowest mode, for debugging.

In ``gated`` and ``ahead`` the next source state is fetched while the GPU builds the
current one, and the zarr write runs on its own thread.

Standalone (a dry run, or targets for later runs)::

    python -m screamcast.replay_targets produce --spec SPEC.json --out-dir DIR \\
        --run-init 2024-09-25T12 --hours 6 12 18 [--region 3,10,14,7,11 --tile-size 128]

    python -m screamcast.replay_targets compare A B

``compare`` checks that two target stores hold the same values bit for bit. The spec
is JSON: ``source`` (``era5`` or ``hres``), ``map`` (the 721 x 1440 -> ne1024pg2 map),
``phis`` (SCREAM phis Zarr array) and ``vert`` (L128 coefficients), as for ``fast_ic``;
``init`` and ``cache`` for ``hres`` (optional ``opendata_root``); optional
``era5_zarr`` and ``fetch_workers`` for ``era5``.
"""

from __future__ import annotations

import argparse
import json
import os
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
from screamcast.distributed_halo import TileTopology
from screamcast.history import history_entry
from screamcast.initial_conditions.fast_ic import (
    ARCO_ERA5,
    Era5,
    build_ic,
    fetch_era5_arco,
    load_map,
    load_vert,
)
from screamcast.initial_conditions.layout import NLEV_SUBSET, VARS_3D
from screamcast.regional import parse_region
from screamcast.replay import BLOCK, ReplayStep, block_mean

FAILED = "FAILED"
DONE = "DONE"
PRODUCER_LOG = "producer.log"
MODES = ("gated", "ahead", "serial")
SOURCES = ("era5", "hres")
SURFACE = ("T_2m", "ps")  # the 2D fields build_ic emits
SPEC_KEYS = {
    "common": ("source", "map", "phis", "vert"),
    "era5": (),
    "hres": ("init", "cache"),
}
# ECMWF open data, HRES "oper" (00 and 12 UTC runs): every 3 h to +144 h, then every
# 6 h to +240 h. The first cycle with every field build_ic needs is 2024-03-05 12 UTC
# (from 2024-02-29 the files lack four pressure levels and w). Checked 2026-11-19.
HRES_LEADS = frozenset((*range(0, 145, 3), *range(150, 241, 6)))
HRES_FIRST_INIT = np.datetime64("2024-03-05T12", "h")
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


def load_spec(path: str) -> dict:
    with open(path) as f:
        spec = json.load(f)
    source = spec.get("source")
    if source not in SOURCES:
        raise ValueError(f"{path}: source must be one of {SOURCES}, got {source!r}")
    missing = [k for k in SPEC_KEYS["common"] + SPEC_KEYS[source] if k not in spec]
    if missing:
        raise ValueError(f"{path}: the {source} spec lacks {missing}")
    return spec


class Era5Source:
    """Hourly ERA5 from the ARCO Zarr store. A failed read is retried."""

    def __init__(self, spec: dict) -> None:
        self.zarr = spec.get("era5_zarr", ARCO_ERA5)
        self.workers = int(spec.get("fetch_workers", 8))
        self.attempts = 3

    def has(self, valid: np.datetime64) -> bool:
        return np.datetime64(valid, "h") == np.datetime64(valid, "s")

    def __call__(self, valid: np.datetime64) -> Era5:
        for attempt in range(1, self.attempts + 1):
            try:
                return fetch_era5_arco(
                    str(np.datetime64(valid, "s")), self.zarr, self.workers
                )
            except Exception as err:  # noqa: BLE001 (HTTP errors vary; a missing time is SystemExit)
                if attempt == self.attempts:
                    raise
                log(
                    f"ERA5 {valid}: {type(err).__name__} {err}; retry in {10 * attempt} s"
                )
                time.sleep(10 * attempt)
        raise AssertionError("unreachable")

    def describe(self, valid: np.datetime64) -> str:
        return f"ERA5 {self.zarr}"


class HresSource:
    """One ECMWF HRES forecast from the open-data bucket (``hres_opendata``)."""

    def __init__(self, spec: dict) -> None:
        from screamcast.initial_conditions.hres_opendata import OPENDATA_ROOT

        self.init = np.datetime64(spec["init"], "h")
        self.cache = spec["cache"]
        self.root = spec.get("opendata_root", OPENDATA_ROOT)
        hour = int(np.datetime_as_string(self.init, unit="h")[11:13])
        if hour not in (0, 12):
            raise ValueError(
                f"HRES init {self.init}: open data holds the 00 and 12 UTC forecasts "
                "(oper) only"
            )
        if self.init < HRES_FIRST_INIT:
            raise ValueError(
                f"HRES init {self.init}: the open-data files are complete from "
                f"{HRES_FIRST_INIT} on"
            )

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


def make_source(spec: dict) -> Era5Source | HresSource:
    return Era5Source(spec) if spec["source"] == "era5" else HresSource(spec)


def source_hours(
    source: Era5Source | HresSource, run_init: np.datetime64, hours
) -> set[int]:
    """The run hours among ``hours`` at which the source has a state (the driver hours
    of ``replay_plan``)."""
    t0 = np.datetime64(run_init, "s")
    return {h for h in hours if source.has(t0 + np.timedelta64(h, "h"))}


def target_gates(plan: list[ReplayStep]) -> list[tuple[int, int | None]]:
    """``(analysis hour, gate)`` in the order the run reads the targets. ``gate`` is the
    analysis before it (None for the first): a ``gated`` producer builds the target
    once the run has begun that analysis."""
    hours = [s.hour for s in plan if s.kind == "analysis"]
    return list(zip(hours, [None, *hours[:-1]]))


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
) -> None:
    """Store ``(channels, tiles, tb, tb)`` block means at ``path`` as one
    ``(channels, 6, face/8, face/8)`` array, NaN off ``tiles``. Written beside ``path``
    and renamed into place, so ``path`` only ever exists complete."""
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
            "history": history,
        }
    )
    os.rename(tmp, path)


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


def _block_set(tiles, tile_size: int) -> set[tuple[int, int, int]]:
    tb = tile_size // BLOCK
    return {
        (face, ti * tb + a, tj * tb + b)
        for face, ti, tj in tiles
        for a in range(tb)
        for b in range(tb)
    }


def covers(path: str, channels: list[str], tiles, tile_size: int) -> bool:
    """Whether the target store at ``path`` holds ``channels`` on every block of
    ``tiles`` (an earlier producer's store kept for a rerun)."""
    attrs = zarr.open_group(path, mode="r").attrs
    if not set(channels) <= set(attrs["variables"]):
        return False
    return _block_set(tiles, tile_size) <= _block_set(
        [tuple(t) for t in attrs["tiles"]], int(attrs["tile_size"])
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


# ------------------------------------------------------------------ markers


def marker_path(out_dir: str, kind: str, valid: np.datetime64) -> str:
    return os.path.join(out_dir, f"{kind}_{hour_stamp(valid)}")


def mark(out_dir: str, kind: str, valid: np.datetime64) -> None:
    """Tell the producer that the run begins the analysis at (``analysis``) or waits for
    the target of (``request``) ``valid``."""
    open(marker_path(out_dir, kind, valid), "w").close()


def wait_for_markers(
    paths: list[str], check: Callable[[], None], poll: float = 0.2
) -> None:
    """Wait until one of ``paths`` exists; ``check()`` runs at each poll and raises to
    stop waiting."""
    while not any(os.path.exists(p) for p in paths):
        check()
        time.sleep(poll)


def wait_for_target(
    out_dir: str, valid: np.datetime64, timeout: float, poll: float = 0.5
) -> float:
    """Block until the target of ``valid`` exists; the seconds waited. Raises at once
    if the producer wrote ``FAILED`` or finished (``DONE``) without it."""
    path = target_path(out_dir, valid)
    start = time.perf_counter()
    while not os.path.isdir(path):
        failed = os.path.join(out_dir, FAILED)
        if os.path.exists(failed):
            with open(failed) as f:
                raise RuntimeError(f"replay target producer failed:\n{f.read()}")
        if os.path.exists(os.path.join(out_dir, DONE)):
            raise RuntimeError(f"replay target producer finished without {path}")
        if time.perf_counter() - start > timeout:
            raise TimeoutError(f"no replay target {path} after {timeout:.0f} s")
        time.sleep(poll)
    return time.perf_counter() - start


# ------------------------------------------------------------------ producer


def producer_tiles(region: str | None, tile_size: int, face_size: int) -> list:
    """The tiles a producer builds: the window's, or every tile of the cube."""
    n = face_size // tile_size
    if region:
        return list(parse_region(region, n))
    return [(f, i, j) for f in range(6) for i in range(n) for j in range(n)]


def produce(args: argparse.Namespace) -> None:
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    for name in (FAILED, DONE):
        if os.path.exists(os.path.join(out_dir, name)):
            os.remove(os.path.join(out_dir, name))
    history = history_entry(
        [sys.executable, "-m", "screamcast.replay_targets", *sys.argv[1:]]
    )
    start = time.perf_counter()
    parent = os.getppid()
    try:
        spec = load_spec(args.spec)
        source = make_source(spec)
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
        tiles = producer_tiles(args.region, args.tile_size, args.face_size)
        channels = target_channels(args.vars.split(","))
        todo = []
        for v in valids:
            path = target_path(out_dir, v)
            if not os.path.isdir(path):
                todo.append(v)
            elif covers(path, channels, tiles, args.tile_size):
                log(f"{hour_stamp(v)} exists, kept")
            else:
                raise ValueError(
                    f"{path} does not hold the channels and tiles this run needs: "
                    "remove it or use another directory"
                )
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
                            out_dir,
                            "analysis",
                            run_init + np.timedelta64(gates[k], "h"),
                        ),
                        marker_path(out_dir, "request", valids[k]),
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
                                marker_path(out_dir, "analysis", valids[k]),
                                marker_path(out_dir, "request", valids[k]),
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
                wait_for_markers([marker_path(out_dir, "request", valid)], check)
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
        open(os.path.join(out_dir, DONE), "w").close()
        log(f"done: {len(todo)} target(s) in {time.perf_counter() - start:.1f} s")
    except BaseException:
        with open(os.path.join(out_dir, FAILED), "w") as f:
            f.write(traceback.format_exc())
        raise


class ProducerProcess:
    """The producer as a child process of one rank, on that rank's GPU. A monitor thread
    writes ``FAILED`` if the child dies without doing so, so ranks waiting for a target
    stop instead of timing out."""

    def __init__(
        self,
        *,
        spec: str,
        out_dir: str,
        run_init: np.datetime64,
        hours: list[int],
        gates: list[int | None],
        mode: str,
        stems: list[str],
        tile_size: int,
        face_size: int,
        region: str | None,
        gpu: str | None,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"producer mode must be one of {MODES}, got {mode!r}")
        self.out_dir = out_dir
        cmd = [
            sys.executable,
            "-m",
            "screamcast.replay_targets",
            "produce",
            "--spec",
            spec,
            "--out-dir",
            out_dir,
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
        os.makedirs(out_dir, exist_ok=True)
        # an earlier run's verdict and markers must not reach this one; its targets stay
        for name in os.listdir(out_dir):
            if name in (FAILED, DONE) or name.startswith(("analysis_", "request_")):
                os.remove(os.path.join(out_dir, name))
        self.log = open(os.path.join(out_dir, PRODUCER_LOG), "a")
        self.log.write(f"# gpu {gpu}: {' '.join(cmd)}\n")
        self.log.flush()
        self.proc = subprocess.Popen(
            cmd,  # noqa: S603 (our own interpreter and module, arguments from the run)
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            stdout=self.log,
            stderr=subprocess.STDOUT,
        )
        self.monitor = threading.Thread(target=self._monitor, daemon=True)
        self.monitor.start()

    def _monitor(self) -> None:
        code = self.proc.wait()
        failed = os.path.join(self.out_dir, FAILED)
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
    """One rank's reader of the targets: the blocks of its own tiles, read on a
    background thread (``prefetch``) so the analysis does not wait on the file system.
    With a ``serial`` producer nothing is prefetched: ``get`` asks for each target."""

    def __init__(
        self,
        out_dir: str,
        channels: list[str],
        topology: TileTopology,
        run_init: np.datetime64,
        *,
        mode: str,
        timeout: float,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"producer mode must be one of {MODES}, got {mode!r}")
        self.out_dir = out_dir
        self.channels = list(channels)
        self.tiles = list(topology.local_tiles)
        self.tile_size = topology.tile_size
        self.run_init = np.datetime64(run_init, "s")
        self.mode = mode
        self.timeout = timeout
        self.pool = ThreadPoolExecutor(1)
        self.futures: dict[int, Future] = {}

    def valid(self, hour: int) -> np.datetime64:
        return self.run_init + np.timedelta64(hour, "h")

    def _load(self, hour: int) -> tuple[np.ndarray, dict]:
        waited = wait_for_target(self.out_dir, self.valid(hour), self.timeout)
        t = time.perf_counter()
        blocks = read_blocks(
            target_path(self.out_dir, self.valid(hour)),
            self.channels,
            self.tiles,
            self.tile_size,
        )
        return blocks, {"waited_s": waited, "read_s": time.perf_counter() - t}

    def prefetch(self, hours: list[int]) -> None:
        if self.mode == "serial":
            return
        for hour in hours:
            if hour not in self.futures:
                self.futures[hour] = self.pool.submit(self._load, hour)

    def get(self, hour: int) -> tuple[np.ndarray, dict]:
        """``(channels, local tiles, tb, tb)`` block means of the target at ``+hour``
        and timings, with ``blocked_s`` the seconds the caller waited here."""
        start = time.perf_counter()
        if not os.path.isdir(target_path(self.out_dir, self.valid(hour))):
            mark(self.out_dir, "request", self.valid(hour))
        future = self.futures.pop(hour, None)
        blocks, info = future.result() if future is not None else self._load(hour)
        return blocks, {**info, "blocked_s": time.perf_counter() - start}

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)


# ------------------------------------------------------------------ CLI


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare", help="bitwise comparison of two target stores")
    c.add_argument("a")
    c.add_argument("b")
    q = sub.add_parser("produce", help="build the targets of a run")
    q.add_argument("--spec", required=True, help="source spec (JSON)")
    q.add_argument("--out-dir", required=True)
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
    q.add_argument("--tile-size", type=int, default=128)
    q.add_argument("--face-size", type=int, default=2048)
    q.add_argument("--group-columns", type=int, default=1 << 21)
    q.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.cmd == "compare":
        bad = compare_stores(args.a, args.b)
        for line in bad:
            print(line)
        if bad:
            raise SystemExit(f"TARGETS_DIFFER {args.a} {args.b}")
        print(f"TARGETS_EQUAL {args.a} {args.b}")
    else:
        produce(args)


if __name__ == "__main__":
    main()
