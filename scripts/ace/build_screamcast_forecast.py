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
"""
Build a global +6 hour screamcast forecast dataset on the native SCREAM grid.

Each ``tile_size x tile_size`` cubesphere tile is rolled out independently.
When launched under ``torchrun``, the tile list ``[(face, x0, y0)]`` is sharded
across ranks. Each rank writes its disjoint tile chunks directly into the zarr.

The output is xarray-compatible:
- 3D variables: ``(time, level, face, x, y)``
- 2D variables: ``(time, face, x, y)``
- coordinates: ``time``, ``face``, ``x``, ``y``, ``lat``, ``lon``, ``level``,
  ``hyam``, ``hybm``

Example:
    torchrun --nproc_per_node=8 scripts/ace/build_screamcast_forecast.py \
        --checkpoint /path/to/best.pth \
        --output /path/to/forecast.6hr.cubesphere.zarr
"""

from __future__ import annotations

import argparse
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
import zarr
from physicsnemo.distributed import DistributedManager
from tqdm import tqdm

import data_catalog
from screamcast.cubesphere_transforms import unstructured_to_6faces
from screamcast.earth2studio_wrappers import ScreamcastModel
from screamcast.history import history_entry
from screamcast.zarr_writer import ZarrWriter

catalog_entry = data_catalog.scream_sdecadal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Screamcast checkpoint")
    parser.add_argument("--output", required=True, help="Output zarr path")
    parser.add_argument("--time-start", type=int, default=0)
    parser.add_argument("--time-end", type=int, default=None)
    parser.add_argument("--n-times", type=int, default=None)
    parser.add_argument(
        "--time-stride",
        type=int,
        default=36,
        help="Stride in native 10-minute steps between initial times",
    )
    parser.add_argument(
        "--forecast-steps",
        type=int,
        default=36,
        help="Number of screamcast steps to run per tile (default: 36 = +6 hr)",
    )
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=None,
        help="Optional global limit on the number of (face, x0, y0) tiles to process",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--writer-queue-size",
        type=int,
        default=2,
        help="Max steps buffered in the async zarr writer queue (default: 2)",
    )
    return parser.parse_args()


def _barrier(distributed: bool) -> None:
    if distributed:
        dist.barrier()


def _load_face_latlon(catalog_entry, nside: int) -> tuple[np.ndarray, np.ndarray]:
    npg = 2
    ds = catalog_entry.to_xarray(chunks=None)
    lat = ds["lat"].values
    lon = ds["lon"].values
    if nside % npg != 0:
        raise ValueError(f"Expected nside={nside} to be divisible by npg={npg}.")
    ne = nside // npg
    # The raw grid file is stored in per-face PG2 ordering. Forecast tiles are
    # written in the reordered 2D face layout produced by ScreamDataSource, so
    # lat/lon must be run through the same cubesphere->2D reorder.
    lat_faces = (
        unstructured_to_6faces(torch.from_numpy(lat), ne=ne, npg=npg)
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    lon_faces = (
        unstructured_to_6faces(torch.from_numpy(lon), ne=ne, npg=npg)
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    return lat_faces, lon_faces


def _split_variable_levels(
    variable_names: list[str],
) -> OrderedDict[str, list[tuple[int | None, int]]]:
    grouped: OrderedDict[str, list[tuple[int | None, int]]] = OrderedDict()
    for channel_index, name in enumerate(variable_names):
        stem, sep, maybe_level = name.rpartition("_")
        if sep and maybe_level.isdigit():
            base = stem
            level = int(maybe_level)
        else:
            base = name
            level = None
        grouped.setdefault(base, []).append((level, channel_index))
    return grouped


def _prepare_output_store(
    *,
    output_path: str,
    grouped_variables: OrderedDict[str, list[tuple[int | None, int]]],
    n_times: int,
    n_steps: int,
    dt: np.timedelta64,
    nside: int,
    tile_size: int,
    time_values: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    hyam: np.ndarray,
    hybm: np.ndarray,
    history_entry: str,
    args: argparse.Namespace,
) -> zarr.Group:
    store = zarr.open_group(output_path, mode="w")
    store.attrs.update(
        {
            "history": history_entry,
            "dataset_type": "screamcast_forecast_on_native_cubesphere_grid",
            "forecast_steps": int(args.forecast_steps),
            "lead_time_hours": float(args.forecast_steps * 10.0 / 60.0),
            "tile_size": int(args.tile_size),
            "output_definition": "all_screamcast_forecast_steps",
            "pixel_ordering": "cubesphere_faces_2d",
        }
    )

    time = store.create_array(
        "time",
        data=time_values.astype("datetime64[s]"),
        chunks=(n_times,),
        dimension_names=("time",),
    )
    time.attrs.update(
        {
            "standard_name": "time",
        }
    )

    store.create_array(
        "step",
        data=np.arange(1, n_steps + 1) * dt,
        dimension_names=("step",),
    )
    store.create_array(
        "face",
        data=np.arange(6, dtype=np.int32),
        dimension_names=("face",),
    )
    store.create_array(
        "x",
        data=np.arange(nside, dtype=np.int32),
        dimension_names=("x",),
    )
    store.create_array(
        "y",
        data=np.arange(nside, dtype=np.int32),
        dimension_names=("y",),
    )

    lat_arr = store.create_array(
        "lat",
        data=lat.astype(np.float32),
        chunks=(1, tile_size, tile_size),
        dimension_names=("face", "x", "y"),
    )
    lat_arr.attrs.update(
        {
            "standard_name": "latitude",
            "units": "degrees_north",
        }
    )
    lon_arr = store.create_array(
        "lon",
        data=lon.astype(np.float32),
        chunks=(1, tile_size, tile_size),
        dimension_names=("face", "x", "y"),
    )
    lon_arr.attrs.update(
        {
            "standard_name": "longitude",
            "units": "degrees_east",
        }
    )

    first_3d = next(
        (
            entries
            for entries in grouped_variables.values()
            if entries[0][0] is not None
        ),
        None,
    )
    if first_3d is not None:
        levels = np.asarray([level for level, _ in first_3d], dtype=np.int32)
    else:
        levels = np.arange(len(hyam), dtype=np.int32)

    store.create_array("level", data=levels, dimension_names=("level",))

    store.create_array(
        "hyam",
        data=hyam[levels].astype(np.float32),
        dimension_names=("level",),
    )
    store.create_array(
        "hybm",
        data=hybm[levels].astype(np.float32),
        dimension_names=("level",),
    )

    for base_name, entries in grouped_variables.items():
        if entries[0][0] is None:
            store.create_array(
                base_name,
                shape=(n_times, n_steps, 6, nside, nside),
                chunks=(1, 1, 1, tile_size, tile_size),
                dtype="f4",
                dimension_names=("time", "step", "face", "x", "y"),
            )
        else:
            store.create_array(
                base_name,
                shape=(n_times, n_steps, len(entries), 6, nside, nside),
                chunks=(1, 1, len(entries), 1, tile_size, tile_size),
                dtype="f4",
                dimension_names=("time", "step", "level", "face", "x", "y"),
            )

    return store


def _run_rollout(
    model: "ScreamcastModel",
    x0: torch.Tensor,
    coords: OrderedDict,
    n_steps: int,
    writer: "ZarrWriter",
    out_index: int,
    face: int,
    x0_tile: int,
    y0_tile: int,
    pbar=None,
) -> None:
    in_vars = list(coords["variable"])
    out_vars = list(model.output_coords(coords)["variable"])
    out_var_idx = {name: i for i, name in enumerate(out_vars)}
    input_from_output = [i for i, name in enumerate(in_vars) if name in out_var_idx]
    output_from_output = [out_var_idx[in_vars[i]] for i in input_from_output]

    state = x0
    with torch.inference_mode():
        for step_index in range(n_steps):
            out_tensor, out_coords = model(state, coords)
            next_state = state.clone()
            next_state[:, input_from_output] = out_tensor[:, output_from_output]
            state = next_state
            writer.write(
                out_index, step_index, face, x0_tile, y0_tile, out_tensor[0, :, 0]
            )
            coords = OrderedDict(out_coords)
            coords["batch"] = np.empty(x0.shape[0])
            if pbar is not None:
                pbar.update(1)


def _tile_triplets(nside: int, tile_size: int) -> list[tuple[int, int, int]]:
    starts = list(range(0, nside, tile_size))
    return [(face, x0, y0) for face in range(6) for x0 in starts for y0 in starts]


def main() -> None:
    args = parse_args()
    run_history = history_entry()

    print(f"output: {args.output}", flush=True)

    DistributedManager.initialize()
    dist_mgr = DistributedManager()
    rank = dist_mgr.rank
    world_size = dist_mgr.world_size
    distributed = dist_mgr.distributed
    device = dist_mgr.device if args.device == "cuda" else torch.device(args.device)

    model = ScreamcastModel.from_checkpoint(args.checkpoint, bf16=True).to(device)
    model.eval()
    model.set_tile_size(args.tile_size)
    model.compile()

    in_coords = OrderedDict(model.input_coords())
    in_variable_names = list(in_coords["variable"])
    out_variable_names = list(model.output_coords(in_coords)["variable"])
    grouped_variables = _split_variable_levels(out_variable_names)

    ds = catalog_entry.to_data_source()
    native_time = catalog_entry.time
    nside = model.latlon.shape[1] if model.latlon is not None else 2048
    if nside % args.tile_size != 0:
        raise ValueError(f"tile_size={args.tile_size} must divide nside={nside}")

    total_native_times = len(native_time)
    tile_triplets = _tile_triplets(nside, args.tile_size)
    if args.max_tiles is not None:
        if args.max_tiles <= 0:
            raise ValueError(f"max_tiles must be positive, got {args.max_tiles}")
        tile_triplets = tile_triplets[: args.max_tiles]
    local_tiles = tile_triplets[rank::world_size]

    t_start = args.time_start
    if args.n_times is not None:
        t_end = t_start + args.n_times * args.time_stride
    elif args.time_end is not None:
        t_end = args.time_end
    else:
        t_end = total_native_times - args.forecast_steps
    t_end = min(t_end, total_native_times - args.forecast_steps)
    initial_indices = list(range(t_start, t_end, args.time_stride))
    initial_times = native_time[initial_indices]
    valid_indices = np.asarray(initial_indices, dtype=np.int64) + args.forecast_steps
    valid_times = native_time[valid_indices]

    hyam, hybm, _ = catalog_entry.to_hybrid_vertical_coordinates()
    hyam = hyam.astype(np.float32)
    hybm = hybm.astype(np.float32)

    lat, lon = _load_face_latlon(catalog_entry, nside)
    if rank == 0:
        _prepare_output_store(
            output_path=args.output,
            grouped_variables=grouped_variables,
            n_times=len(initial_indices),
            n_steps=args.forecast_steps,
            dt=model.dt,
            nside=nside,
            tile_size=args.tile_size,
            time_values=valid_times,
            lat=lat,
            lon=lon,
            hyam=hyam,
            hybm=hybm,
            history_entry=run_history,
            args=args,
        )
        zarr.consolidate_metadata(args.output)
    _barrier(distributed)
    store = zarr.open_group(args.output, mode="a")
    writer = ZarrWriter(
        store, grouped_variables, args.tile_size, maxsize=args.writer_queue_size
    )

    total_steps = len(initial_indices) * len(local_tiles) * args.forecast_steps
    with tqdm(
        total=total_steps, desc="rollout", disable=rank != 0, unit="step"
    ) as pbar:
        for out_index, init_time in enumerate(initial_times):
            if rank == 0:
                pbar.set_postfix(
                    t=str(init_time),
                    tile=f"{out_index * len(local_tiles) + 1}/{len(initial_indices) * len(local_tiles)}",
                )

            for face, x0, y0 in local_tiles:
                x_slice = np.arange(x0, x0 + args.tile_size, dtype=np.int64)
                y_slice = np.arange(y0, y0 + args.tile_size, dtype=np.int64)
                fetch_coords = {
                    "time": np.array([init_time]),
                    "variable": in_variable_names,
                    "face": np.array([face]),
                    "x": x_slice,
                    "y": y_slice,
                }
                x_init = torch.from_numpy(ds(fetch_coords).values).to(device)
                coords_model = OrderedDict(in_coords)
                coords_model["time"] = np.array([init_time])
                coords_model["face"] = np.array([face])
                coords_model["x"] = x_slice
                coords_model["y"] = y_slice
                coords_model["batch"] = np.empty(1)
                _run_rollout(
                    model=model,
                    x0=x_init,
                    coords=coords_model,
                    n_steps=args.forecast_steps,
                    writer=writer,
                    out_index=out_index,
                    face=face,
                    x0_tile=x0,
                    y0_tile=y0,
                    pbar=pbar,
                )
            _barrier(distributed)

    writer.close()

    if rank == 0:
        print(f"wrote {args.output}", flush=True)

    dist_mgr.cleanup()


if __name__ == "__main__":
    main()
