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
"""Async zarr writer for screamcast forecast output."""

from __future__ import annotations

import logging
import queue
import threading
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
import zarr
from einops import rearrange

logger = logging.getLogger(__name__)


def _split_variable_levels(
    variable_names: list[str],
) -> OrderedDict[str, list[tuple[int | None, int]]]:
    """Group flat channel names (e.g. U_0, U_1, PRECT) by base variable."""
    grouped: OrderedDict[str, list[tuple[int | None, int]]] = OrderedDict()
    for channel_index, name in enumerate(variable_names):
        stem, sep, maybe_level = name.rpartition("_")
        if sep and maybe_level.isdigit():
            base, level = stem, int(maybe_level)
        else:
            base, level = name, None
        grouped.setdefault(base, []).append((level, channel_index))
    return grouped


def prepare_output_store(
    *,
    output_path: str,
    grouped_variables: OrderedDict,
    n_times: int,
    n_steps: int,
    nside: int,
    tile_size: int,
    time_values: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    attrs: dict | None = None,
    dt: np.timedelta64 | None = None,
    hyam: np.ndarray | None = None,
    hybm: np.ndarray | None = None,
) -> zarr.Group:
    """Create and pre-allocate a zarr store for screamcast forecast output.

    Schema
    ------
    Coordinates: ``time``, ``step``, ``face``, ``x``, ``y``
    Optional coordinates: ``level``, ``hyam``, ``hybm`` (when *hyam*/*hybm* provided)
    Spatial metadata: ``lat``, ``lon`` with shape ``(face, x, y)``
    Variables
      - 2-D: shape ``(time, step, face, x, y)``
      - 3-D: shape ``(time, step, level, face, x, y)``

    Args:
        output_path: Path to the output zarr store (created fresh).
        grouped_variables: Mapping from base variable name to
            ``[(level_or_None, channel_index), ...]`` as produced by
            ``_split_variable_levels``.
        n_times: Number of initial times.
        n_steps: Number of forecast steps.
        nside: Full-face grid size (e.g. 2048 for ne=1024, npg=2).
        tile_size: Spatial tile size used for zarr chunk size.
        time_values: 1-D array of initial times (shape ``(n_times,)``).
        lat: Face lat array, shape ``(6, nside, nside)``, degrees north.
        lon: Face lon array, shape ``(6, nside, nside)``, degrees east.
        attrs: Optional global attributes written to ``store.attrs``.
        dt: Time delta per step; used to create the ``step`` coordinate.
            When ``None`` the step coordinate is stored as integer indices.
        hyam: Hybrid-A coefficients, shape ``(n_levels,)``.  When provided
            together with *hybm*, ``level`` / ``hyam`` / ``hybm`` coordinates
            are added.
        hybm: Hybrid-B coefficients, same shape as *hyam*.

    Returns:
        The opened zarr group (mode ``"w"`` — freshly created).
    """
    store = zarr.open_group(output_path, mode="w")
    if attrs:
        store.attrs.update(attrs)

    store.create_array(
        "time",
        data=time_values.astype("datetime64[s]"),
        chunks=(n_times,),
        dimension_names=("time",),
    ).attrs.update({"standard_name": "time"})

    if dt is not None:
        store.create_array(
            "step",
            data=np.arange(1, n_steps + 1) * dt,
            dimension_names=("step",),
        )
    else:
        store.create_array(
            "step",
            data=np.arange(1, n_steps + 1, dtype=np.int32),
            dimension_names=("step",),
        )

    store.create_array(
        "face", data=np.arange(6, dtype=np.int32), dimension_names=("face",)
    )
    store.create_array(
        "x", data=np.arange(nside, dtype=np.int32), dimension_names=("x",)
    )
    store.create_array(
        "y", data=np.arange(nside, dtype=np.int32), dimension_names=("y",)
    )

    store.create_array(
        "lat",
        data=lat.astype(np.float32),
        chunks=(1, tile_size, tile_size),
        shards=lat.shape,
        dimension_names=("face", "x", "y"),
    ).attrs.update({"standard_name": "latitude", "units": "degrees_north"})

    store.create_array(
        "lon",
        data=lon.astype(np.float32),
        chunks=(1, tile_size, tile_size),
        shards=lon.shape,
        dimension_names=("face", "x", "y"),
    ).attrs.update({"standard_name": "longitude", "units": "degrees_east"})

    if hyam is not None and hybm is not None:
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
            "hyam", data=hyam[levels].astype(np.float32), dimension_names=("level",)
        )
        store.create_array(
            "hybm", data=hybm[levels].astype(np.float32), dimension_names=("level",)
        )

    for base_name, entries in grouped_variables.items():
        if entries[0][0] is None:
            store.create_array(
                base_name,
                shape=(n_times, n_steps, 6, nside, nside),
                chunks=(1, 1, 1, tile_size, tile_size),
                shards=(1, 1, *lon.shape),
                dtype="f4",
                dimension_names=("time", "step", "face", "x", "y"),
            )
        else:
            store.create_array(
                base_name,
                shape=(n_times, n_steps, len(entries), 6, nside, nside),
                chunks=(1, 1, 1, 1, tile_size, tile_size),
                shards=(1, 1, 1, *lon.shape),
                dtype="f4",
                dimension_names=("time", "step", "level", "face", "x", "y"),
            )

    return store


def prepare_latlon_plev_store(
    *,
    output_path: str,
    grouped_variables: OrderedDict,
    n_times: int,
    n_steps: int,
    time_values: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    plev: np.ndarray,
    attrs: dict | None = None,
    dt: np.timedelta64 | None = None,
) -> zarr.Group:
    """Create and pre-allocate a zarr store on a regular lat/lon + pressure grid.

    Schema
    ------
    Coordinates: ``time``, ``step``, ``lat``, ``lon``, ``plev``
    Variables
      - 2-D: shape ``(time, step, lat, lon)``
      - 3-D: shape ``(time, step, plev, lat, lon)``

    Args:
        output_path: Path to the output zarr store (created fresh).
        grouped_variables: Mapping from base variable name to
            ``[(level_or_None, channel_index), ...]`` as produced by
            ``_split_variable_levels``.  Only the level-or-None slot is
            inspected to decide 2-D vs 3-D layout.
        n_times: Number of initial times.
        n_steps: Number of forecast output steps.
        time_values: 1-D array of initial times (shape ``(n_times,)``).
        lat: Target latitudes in degrees north (1-D).
        lon: Target longitudes in degrees east (1-D).
        plev: Target pressure levels in Pa (1-D).
        attrs: Optional global attributes written to ``store.attrs``.
        dt: Time delta per output step; when ``None`` the step coordinate is
            stored as integer indices.

    Returns:
        The opened zarr group (mode ``"w"`` — freshly created).
    """
    store = zarr.open_group(output_path, mode="w")
    if attrs:
        store.attrs.update(attrs)

    n_lat = int(lat.shape[0])
    n_lon = int(lon.shape[0])
    n_plev = int(plev.shape[0])

    store.create_array(
        "time",
        data=time_values.astype("datetime64[s]"),
        chunks=(n_times,),
        dimension_names=("time",),
    ).attrs.update({"standard_name": "time"})

    if dt is not None:
        store.create_array(
            "step",
            data=np.arange(1, n_steps + 1) * dt,
            dimension_names=("step",),
        )
    else:
        store.create_array(
            "step",
            data=np.arange(1, n_steps + 1, dtype=np.int32),
            dimension_names=("step",),
        )

    store.create_array(
        "lat",
        data=lat.astype(np.float32),
        dimension_names=("lat",),
    ).attrs.update({"standard_name": "latitude", "units": "degrees_north"})

    store.create_array(
        "lon",
        data=lon.astype(np.float32),
        dimension_names=("lon",),
    ).attrs.update({"standard_name": "longitude", "units": "degrees_east"})

    store.create_array(
        "plev",
        data=plev.astype(np.float32),
        dimension_names=("plev",),
    ).attrs.update({"standard_name": "air_pressure", "units": "Pa", "positive": "down"})

    for base_name, entries in grouped_variables.items():
        if entries[0][0] is None:
            store.create_array(
                base_name,
                shape=(n_times, n_steps, n_lat, n_lon),
                chunks=(1, 1, n_lat, n_lon),
                dtype="f4",
                dimension_names=("time", "step", "lat", "lon"),
            )
        else:
            store.create_array(
                base_name,
                shape=(n_times, n_steps, n_plev, n_lat, n_lon),
                chunks=(1, 1, 1, n_lat, n_lon),
                dtype="f4",
                dimension_names=("time", "step", "plev", "lat", "lon"),
            )

    return store


def partition_grouped_variables(
    grouped_variables: OrderedDict[str, list[tuple[int | None, int]]],
    world_size: int,
) -> list[OrderedDict[str, list[tuple[int | None, int]]]]:
    """Partition base variables across ranks while preserving input order."""
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    items = list(grouped_variables.items())
    base = len(items) // world_size
    remainder = len(items) % world_size
    partitions: list[OrderedDict[str, list[tuple[int | None, int]]]] = []
    start = 0
    for rank in range(world_size):
        stop = start + base + (1 if rank < remainder else 0)
        partitions.append(OrderedDict(items[start:stop]))
        start = stop
    return partitions


def reindex_grouped_variables(
    grouped_variables: OrderedDict[str, list[tuple[int | None, int]]],
) -> OrderedDict[str, list[tuple[int | None, int]]]:
    """Rewrite grouped channel indices to a dense local ``0..N-1`` range."""
    reindexed: OrderedDict[str, list[tuple[int | None, int]]] = OrderedDict()
    next_channel = 0
    for base_name, entries in grouped_variables.items():
        local_entries = []
        for level, _ in entries:
            local_entries.append((level, next_channel))
            next_channel += 1
        reindexed[base_name] = local_entries
    return reindexed


def grouped_channel_indices(
    grouped_variables: OrderedDict[str, list[tuple[int | None, int]]],
) -> list[int]:
    """Flatten grouped variable metadata into channel-index order."""
    return [
        channel_index
        for entries in grouped_variables.values()
        for _, channel_index in entries
    ]


class FaceZarrWriteStep:
    """Rollout callback that writes rank-owned variables as full face slabs.

    The callback gathers the full local output tensor into cubed-sphere faces so
    every rank enters the collective with the same channel shape, then slices to
    the channels owned by this rank and writes those base variables into the raw
    output zarr store. Channel ownership is derived from the distributed
    ``topology`` and the global ``grouped_variables`` metadata.
    """

    def __init__(
        self,
        *,
        output_steps: int,
        topology,
        grouped_variables: OrderedDict[str, list[tuple[int | None, int]]],
        store: zarr.Group,
        maxsize: int = 2,
    ):
        self.output_steps = output_steps
        self.topology = topology
        self.grouped_variable_partitions = partition_grouped_variables(
            grouped_variables, topology.world_size
        )
        self.channel_indices_by_rank = [
            grouped_channel_indices(partition)
            for partition in self.grouped_variable_partitions
        ]
        self.owned_grouped_variables = self.grouped_variable_partitions[topology.rank]
        self.store = store
        self.local_grouped_variables = reindex_grouped_variables(
            self.owned_grouped_variables
        )
        self.owned_channel_indices = self.channel_indices_by_rank[topology.rank]
        self._tile_n_elements = (
            topology.tiles_per_rank * topology.tile_size * topology.tile_size
        )
        self._send_sizes = [
            len(channel_indices) * self._tile_n_elements
            for channel_indices in self.channel_indices_by_rank
        ]
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _exchange_owned_faces(self, out_tensor: torch.Tensor) -> torch.Tensor:
        if out_tensor.ndim != 5:
            raise ValueError(
                "Expected out_tensor with shape "
                "(batch, channel, tile, x, y), "
                f"got {tuple(out_tensor.shape)}"
            )

        batch, n_channels, tiles_per_rank, nx, ny = out_tensor.shape
        if tiles_per_rank != self.topology.tiles_per_rank:
            raise ValueError(
                f"Expected {self.topology.tiles_per_rank} local tiles, got {tiles_per_rank}"
            )
        if nx != self.topology.tile_size or ny != self.topology.tile_size:
            raise ValueError(
                f"Expected tile shape ({self.topology.tile_size}, {self.topology.tile_size}), "
                f"got ({nx}, {ny})"
            )
        if self.topology.face_size % self.topology.tile_size != 0:
            raise ValueError(
                "face_size must be divisible by tile_size, got "
                f"{self.topology.face_size} and {self.topology.tile_size}"
            )
        tiles_per_dim = self.topology.face_size // self.topology.tile_size
        expected_tiles = self.topology.n_faces * tiles_per_dim * tiles_per_dim
        if self.topology.total_tiles != expected_tiles:
            raise ValueError(
                f"Expected topology.total_tiles={expected_tiles}, got {self.topology.total_tiles}"
            )
        if self.owned_channel_indices:
            max_channel_index = max(
                channel_index
                for channel_indices in self.channel_indices_by_rank
                for channel_index in channel_indices
            )
            if max_channel_index >= n_channels:
                raise ValueError(
                    f"Channel index {max_channel_index} is out of range for "
                    f"{n_channels} output channels"
                )

        if self.topology.world_size == 1:
            owned_tiles = out_tensor[:, self.owned_channel_indices].contiguous()
        else:
            send_parts = []
            for channel_indices in self.channel_indices_by_rank:
                send_parts.append(  # noqa: PERF401
                    out_tensor[:, channel_indices].contiguous().reshape(batch, -1)
                )
            send_tensor = torch.cat(send_parts, dim=1)
            send_1d = send_tensor.transpose(0, 1).contiguous().reshape(-1)

            recv_chunk_size = len(self.owned_channel_indices) * self._tile_n_elements
            recv_total = recv_chunk_size * self.topology.world_size
            recv_1d = torch.empty(
                recv_total * batch,
                dtype=out_tensor.dtype,
                device=out_tensor.device,
            )

            dist.all_to_all_single(
                recv_1d,
                send_1d,
                output_split_sizes=[recv_chunk_size * batch] * self.topology.world_size,
                input_split_sizes=[send_size * batch for send_size in self._send_sizes],
            )

            recv_tensor = recv_1d.reshape(recv_total, batch).transpose(0, 1)
            owned_tiles = recv_tensor.reshape(
                batch,
                self.topology.world_size,
                len(self.owned_channel_indices),
                self.topology.tiles_per_rank,
                self.topology.tile_size,
                self.topology.tile_size,
            )
            owned_tiles = owned_tiles.permute(0, 2, 1, 3, 4, 5).reshape(
                batch,
                len(self.owned_channel_indices),
                self.topology.total_tiles,
                self.topology.tile_size,
                self.topology.tile_size,
            )

        return rearrange(
            owned_tiles,
            "b c (f tx ty) x y -> b c f (tx x) (ty y)",
            f=self.topology.n_faces,
            tx=tiles_per_dim,
            ty=tiles_per_dim,
        )

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                break
            time_index, out_index, faces = item
            logger.debug(
                "face-zarr rank=%s step=%d starting store write",
                self.topology.rank,
                out_index * self.output_steps,
            )
            self._write_faces(time_index, out_index, faces)
            logger.debug(
                "face-zarr rank=%s step=%d store write done",
                self.topology.rank,
                out_index * self.output_steps,
            )

    def _write_faces(
        self,
        time_index: int,
        step_index: int,
        faces: torch.Tensor,
    ) -> None:
        faces_np = faces.detach().cpu().numpy().astype(np.float32)
        for base_name, entries in self.local_grouped_variables.items():
            channel_indices = [ci for _, ci in entries]
            if entries[0][0] is None:
                self.store[base_name][time_index, step_index] = faces_np[
                    channel_indices[0]
                ]
            else:
                self.store[base_name][time_index, step_index] = faces_np[
                    channel_indices
                ]

    def __call__(self, step_index: int, out_tensor: torch.Tensor) -> None:
        """Write one output step when ``step_index`` matches ``output_steps``."""
        if step_index % self.output_steps != 0:
            return
        out_index = step_index // self.output_steps
        logger.debug(
            "face-zarr rank=%s step=%d starting channel exchange",
            self.topology.rank,
            step_index,
        )
        owned_faces = self._exchange_owned_faces(out_tensor)
        logger.debug(
            "face-zarr rank=%s step=%d channel exchange done",
            self.topology.rank,
            step_index,
        )
        if self.owned_channel_indices:
            self._queue.put((0, out_index, owned_faces[0]))

    def close(self) -> None:
        """Flush all pending face writes and stop the worker thread."""
        self._queue.put(None)
        self._thread.join()


class ZarrWriter:
    """Background thread that writes tensors to zarr as they arrive.

    Tensors are enqueued from the GPU-side rollout loop and written to zarr by a
    worker thread, so disk I/O overlaps GPU computation. The bounded queue provides
    backpressure so GPU memory use stays bounded.

    Args:
        store: Open zarr group with pre-created variable arrays.
        grouped_variables: Mapping from base variable name to a list of
            ``(level_or_None, channel_index)`` pairs, as produced by
            ``_split_variable_levels``.
        tile_size: Spatial tile size used when slicing the zarr arrays.
        maxsize: Maximum number of pending write items in the queue.
    """

    def __init__(
        self, store, grouped_variables: OrderedDict, tile_size: int, maxsize: int = 2
    ):
        self.store = store
        self.grouped_variables = grouped_variables
        self.tile_size = tile_size
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                break
            out_index, step_index, face, x0, y0, tensor = item
            tile_np = tensor.detach().cpu().numpy().astype(np.float32)
            ts = self.tile_size
            for base_name, entries in self.grouped_variables.items():
                channel_indices = [ci for _, ci in entries]
                if entries[0][0] is None:
                    self.store[base_name][
                        out_index, step_index, face, x0 : x0 + ts, y0 : y0 + ts
                    ] = tile_np[channel_indices[0]]
                else:
                    self.store[base_name][
                        out_index, step_index, :, face, x0 : x0 + ts, y0 : y0 + ts
                    ] = tile_np[channel_indices]
            self._queue.task_done()

    def write(
        self,
        out_index: int,
        step_index: int,
        face: int,
        x0: int,
        y0: int,
        tensor: torch.Tensor,
    ) -> None:
        """Enqueue a tensor for writing (blocks if the queue is full)."""
        self._queue.put((out_index, step_index, face, x0, y0, tensor))

    def close(self) -> None:
        """Flush all pending writes and stop the worker thread."""
        self._queue.put(None)
        self._thread.join()
