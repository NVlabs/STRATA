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
import numpy as np
import torch

from screamcast import distributed_state_fetch
from screamcast.distributed_halo import TileTopology


class _ArrayBox:
    def __init__(self, values: np.ndarray):
        self.values = values


class FakeDataSource:
    def __init__(self, variable_names: list[str], face_size: int):
        self.variable_index = {name: i for i, name in enumerate(variable_names)}
        self.face_size = face_size
        self.calls: list[dict] = []

    def __call__(self, coords: dict) -> _ArrayBox:
        self.calls.append(
            {
                "time": coords["time"].copy(),
                "variable": list(coords["variable"]),
                "face": coords["face"].copy(),
                "x": coords["x"].copy(),
                "y": coords["y"].copy(),
            }
        )
        face = coords["face"].astype(np.float32)[None, None, :, None, None]
        x = coords["x"].astype(np.float32)[None, None, None, :, None]
        y = coords["y"].astype(np.float32)[None, None, None, None, :]
        data = np.empty(
            (
                1,
                len(coords["variable"]),
                len(coords["face"]),
                len(coords["x"]),
                len(coords["y"]),
            ),
            dtype=np.float32,
        )
        for channel_index, name in enumerate(coords["variable"]):
            base = self.variable_index[name] * 1000.0
            data[:, channel_index] = base + face + 10.0 * x + y
        return _ArrayBox(data)


def _make_rank_shard(
    variable_names: list[str],
    channel_range: tuple[int, int],
    face_size: int,
) -> torch.Tensor:
    ds = FakeDataSource(variable_names, face_size)
    return distributed_state_fetch._fetch_full_face_channel_shard(
        truth_time=np.datetime64("2020-10-13T00:00:00"),
        variables=variable_names,
        channel_range=channel_range,
        nside=face_size,
        ds=ds,
        device=torch.device("cpu"),
        channel_chunk=max(channel_range[1] - channel_range[0], 1),
    )


def test_partition_channel_ranges_balances_remainder():
    assert distributed_state_fetch._partition_channel_ranges(7, 3) == [
        (0, 3),
        (3, 5),
        (5, 7),
    ]


def test_fetch_full_face_channel_shard_reads_only_local_variable_range():
    variable_names = ["a", "b", "c", "d", "e"]
    ds = FakeDataSource(variable_names, face_size=4)

    shard = distributed_state_fetch._fetch_full_face_channel_shard(
        truth_time=np.datetime64("2020-10-13T00:00:00"),
        variables=variable_names,
        channel_range=(1, 4),
        nside=4,
        ds=ds,
        device=torch.device("cpu"),
        channel_chunk=2,
    )

    assert shard.shape == (1, 3, 6, 4, 4)
    assert [call["variable"] for call in ds.calls] == [["b", "c"], ["d"]]
    torch.testing.assert_close(
        shard[0, :, 0, 0, 0], torch.tensor([1000.0, 2000.0, 3000.0])
    )


def test_exchange_channel_shards_reassembles_full_channel_order(monkeypatch):
    variable_names = ["a", "b", "c", "d", "e"]
    face_size = 4
    channel_ranges = distributed_state_fetch._partition_channel_ranges(
        len(variable_names), 3
    )
    rank = 1
    local_shard = _make_rank_shard(variable_names, channel_ranges[rank], face_size)
    rank_shards = [
        _make_rank_shard(variable_names, channel_range, face_size)
        for channel_range in channel_ranges
    ]

    def fake_all_gather(tensor_list, padded_local):
        for tensor, shard in zip(tensor_list, rank_shards, strict=True):
            tensor.zero_()
            tensor[:, : shard.shape[1]] = shard

    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)

    full_faces = distributed_state_fetch._exchange_channel_shards(
        local_shard=local_shard,
        channel_ranges=channel_ranges,
        rank=rank,
    )

    assert full_faces.shape == (1, 5, 6, 4, 4)
    torch.testing.assert_close(
        full_faces[0, :, 0, 0, 0],
        torch.tensor([0.0, 1000.0, 2000.0, 3000.0, 4000.0]),
    )


def test_extract_local_tiles_from_faces_uses_topology_tile_assignment():
    topology = TileTopology(
        world_size=2,
        rank=1,
        face_size=4,
        tile_size=2,
        halo_width=0,
    )
    full_faces = torch.zeros(
        1, 1, topology.n_faces, topology.face_size, topology.face_size
    )
    ts = topology.tile_size
    for global_index, (face, tile_i, tile_j) in enumerate(topology.global_tiles):
        full_faces[
            :,
            :,
            face,
            tile_i * ts : (tile_i + 1) * ts,
            tile_j * ts : (tile_j + 1) * ts,
        ] = float(global_index)

    local_tiles = distributed_state_fetch._extract_local_tiles_from_faces(
        full_faces, topology
    )

    assert local_tiles.shape == (1, 1, topology.tiles_per_rank, ts, ts)
    start = topology.rank * topology.tiles_per_rank
    for local_index in range(topology.tiles_per_rank):
        torch.testing.assert_close(
            local_tiles[0, 0, local_index],
            torch.full((ts, ts), float(start + local_index)),
        )


def test_fetch_local_tiles_reads_shard_then_extracts_local_tiles(monkeypatch):
    variable_names = ["a", "b", "c", "d", "e"]
    topology = TileTopology(
        world_size=2,
        rank=1,
        face_size=4,
        tile_size=2,
        halo_width=0,
    )
    ds = FakeDataSource(variable_names, face_size=topology.face_size)
    channel_ranges = distributed_state_fetch._partition_channel_ranges(
        len(variable_names), topology.world_size
    )
    rank_shards = [
        _make_rank_shard(variable_names, channel_range, topology.face_size)
        for channel_range in channel_ranges
    ]

    def fake_all_gather(tensor_list, padded_local):
        for tensor, shard in zip(tensor_list, rank_shards, strict=True):
            tensor.zero_()
            tensor[:, : shard.shape[1]] = shard

    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)

    x_tensor, coords = distributed_state_fetch.fetch_local_tiles(
        ds=ds,
        in_coords={"variable": variable_names},
        topology=topology,
        device=torch.device("cpu"),
        t0=np.datetime64("2020-10-13T00:00:00"),
        channel_chunk=1,
    )

    assert [call["variable"] for call in ds.calls] == [["d"], ["e"]]
    assert x_tensor.shape == (
        1,
        len(variable_names),
        topology.tiles_per_rank,
        topology.tile_size,
        topology.tile_size,
    )
    assert coords["face"].tolist() == list(range(topology.tiles_per_rank))
    assert coords["x"].tolist() == [0, 1]
    assert coords["y"].tolist() == [0, 1]

    ts = topology.tile_size
    for local_index, (face, tile_i, tile_j) in enumerate(topology.local_tiles):
        i0 = tile_i * ts
        j0 = tile_j * ts
        for channel_index in range(len(variable_names)):
            expected = torch.tensor(
                channel_index * 1000.0
                + face
                + 10.0 * np.arange(i0, i0 + ts, dtype=np.float32)[:, None]
                + np.arange(j0, j0 + ts, dtype=np.float32)[None, :],
                dtype=torch.float32,
            )
            torch.testing.assert_close(
                x_tensor[0, channel_index, local_index], expected
            )
