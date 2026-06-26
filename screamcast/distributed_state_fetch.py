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
from __future__ import annotations

import logging

import numpy as np
import torch

from screamcast.distributed_halo import TileTopology

logger = logging.getLogger(__name__)


def _partition_channel_ranges(
    n_channels: int,
    world_size: int,
) -> list[tuple[int, int]]:
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    base = n_channels // world_size
    remainder = n_channels % world_size
    ranges = []
    start = 0
    for rank in range(world_size):
        stop = start + base + (1 if rank < remainder else 0)
        ranges.append((start, stop))
        start = stop
    return ranges


def _fetch_full_face_channel_shard(
    truth_time,
    variables,
    channel_range: tuple[int, int],
    nside: int,
    ds,
    device: torch.device,
    channel_chunk: int,
) -> torch.Tensor:
    start, stop = channel_range
    if channel_chunk <= 0:
        raise ValueError(f"channel_chunk must be positive, got {channel_chunk}")
    if start == stop:
        return torch.empty((1, 0, 6, nside, nside), device=device)

    vars_all = list(variables)
    x_chunks = []
    for chunk_start in range(start, stop, channel_chunk):
        chunk_stop = min(chunk_start + channel_chunk, stop)
        vars_chunk = vars_all[chunk_start:chunk_stop]
        logger.info(
            "fetch_full_face_channel_shard fetching vars %d:%d at time=%s",
            chunk_start,
            chunk_stop,
            truth_time,
        )
        coords = {
            "time": np.array([truth_time]),
            "variable": vars_chunk,
            "face": np.arange(6),
            "x": np.arange(nside),
            "y": np.arange(nside),
        }
        chunk_np = ds(coords).values
        x_chunks.append(torch.from_numpy(chunk_np).to(device))
    return torch.cat(x_chunks, dim=1)


def _exchange_channel_shards(
    local_shard: torch.Tensor,
    channel_ranges: list[tuple[int, int]],
    rank: int,
) -> torch.Tensor:
    local_start, local_stop = channel_ranges[rank]
    expected_channels = local_stop - local_start
    if local_shard.shape[1] != expected_channels:
        raise ValueError(
            f"Expected local shard to have {expected_channels} channels, "
            f"got {local_shard.shape[1]}"
        )
    if len(channel_ranges) == 1:
        return local_shard

    max_channels = max(stop - start for start, stop in channel_ranges)
    if max_channels == 0:
        return local_shard

    padded = local_shard.new_zeros(
        local_shard.shape[0],
        max_channels,
        local_shard.shape[2],
        local_shard.shape[3],
        local_shard.shape[4],
    )
    if expected_channels > 0:
        padded[:, :expected_channels] = local_shard
    tensor_list = [torch.zeros_like(padded) for _ in range(len(channel_ranges))]
    torch.distributed.all_gather(tensor_list, padded.contiguous())

    shards = []
    for gathered, (start, stop) in zip(tensor_list, channel_ranges, strict=True):
        n_channels = stop - start
        if n_channels > 0:
            shards.append(gathered[:, :n_channels])
    if not shards:
        return local_shard
    return torch.cat(shards, dim=1)


def _extract_local_tiles_from_faces(
    full_faces: torch.Tensor,
    topology: TileTopology,
) -> torch.Tensor:
    if full_faces.shape[-3:] != (
        topology.n_faces,
        topology.face_size,
        topology.face_size,
    ):
        raise ValueError(
            "Expected full face tensor shaped "
            f"(..., {topology.n_faces}, {topology.face_size}, {topology.face_size}), "
            f"got {tuple(full_faces.shape)}"
        )

    batch = full_faces.shape[0]
    channels = full_faces.shape[1]
    tiles = full_faces.new_empty(
        batch,
        channels,
        topology.tiles_per_rank,
        topology.tile_size,
        topology.tile_size,
    )
    ts = topology.tile_size
    for local_idx, (face, tile_i, tile_j) in enumerate(topology.local_tiles):
        i0 = tile_i * ts
        j0 = tile_j * ts
        tiles[:, :, local_idx] = full_faces[:, :, face, i0 : i0 + ts, j0 : j0 + ts]
    return tiles


def load_full_face_scream_state(
    truth_time,
    variables,
    nside,
    ds,
    device,
    channel_chunk,
    world_size,
    rank,
):
    """Load one SCREAM state time as a full-face tensor on every rank.

    Each rank fetches only its assigned variable shard over the full cubed-sphere
    grid, then exchanges those channel shards so all ranks reconstruct the full
    ``(1, n_channels, 6, nside, nside)`` tensor locally.
    """
    vars_all = list(variables)
    channel_ranges = _partition_channel_ranges(len(vars_all), world_size)
    logger.info(
        "load_full_face_scream_state start time=%s n_vars=%d chunk=%d nside=%d rank=%d/%d",
        truth_time,
        len(vars_all),
        channel_chunk,
        nside,
        rank,
        world_size,
    )
    local_shard = _fetch_full_face_channel_shard(
        truth_time=truth_time,
        variables=vars_all,
        channel_range=channel_ranges[rank],
        nside=nside,
        ds=ds,
        device=device,
        channel_chunk=channel_chunk,
    )
    out = _exchange_channel_shards(local_shard, channel_ranges, rank)
    logger.info("load_full_face_scream_state done shape=%s", tuple(out.shape))
    return out


def fetch_local_tiles(
    ds,
    in_coords,
    topology: TileTopology,
    device: torch.device,
    t0,
    channel_chunk: int,
) -> tuple[torch.Tensor, dict]:
    """Fetch one time slice and return only this rank's local interior tiles.

    The underlying data read is performed via channel-sharded full-face fetches.
    After the full-face state is reconstructed locally, the tensor is sliced
    into this rank's tile allocation from ``topology``.
    """
    # performs all gather here, but could be optimized to use all to all instead
    full_faces = load_full_face_scream_state(
        truth_time=t0,
        variables=in_coords["variable"],
        nside=topology.face_size,
        ds=ds,
        device=device,
        channel_chunk=channel_chunk,
        world_size=topology.world_size,
        rank=topology.rank,
    )
    x_tensor = _extract_local_tiles_from_faces(full_faces, topology)
    coords = topology.interior_coords(in_coords, time=np.array([t0]))
    return x_tensor, coords
