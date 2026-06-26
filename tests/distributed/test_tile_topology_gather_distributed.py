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
import torch
import torch.distributed as dist

from screamcast.distributed_halo import TileTopology


def _expected_faces(topology: TileTopology) -> torch.Tensor:
    expected = torch.zeros(
        (1, topology.n_faces, topology.face_size, topology.face_size)
    )
    for global_index, (face, ti, tj) in enumerate(topology.global_tiles):
        value = float(global_index + 1)
        x0 = ti * topology.tile_size
        y0 = tj * topology.tile_size
        expected[
            0, face, x0 : x0 + topology.tile_size, y0 : y0 + topology.tile_size
        ] = value
    return expected


def test_tile_topology_gather_tiles_to_faces_distributed(distributed_cuda_context):
    rank, world_size, device = distributed_cuda_context

    topology = TileTopology(
        world_size=world_size,
        rank=rank,
        face_size=4,
        tile_size=2,
        halo_width=0,
    )
    assert topology.tiles_per_rank == 3

    local_tiles = torch.zeros(
        (1, topology.tiles_per_rank, topology.tile_size, topology.tile_size),
        device=device,
        dtype=torch.float32,
    )
    for local_idx, global_index in enumerate(
        range(rank * topology.tiles_per_rank, (rank + 1) * topology.tiles_per_rank)
    ):
        local_tiles[0, local_idx] = float(global_index + 1)

    dist.barrier()

    faces = topology.gather_tiles_to_faces(local_tiles)

    expected = _expected_faces(topology).to(device)
    torch.testing.assert_close(faces, expected)

    dist.barrier()


def test_tile_topology_faces_to_local_tiles_distributed(distributed_cuda_context):
    """``faces_to_local_tiles`` should be a local (no-collective) inverse of the
    scatter half of ``gather_tiles_to_faces``."""
    rank, world_size, device = distributed_cuda_context

    topology = TileTopology(
        world_size=world_size,
        rank=rank,
        face_size=4,
        tile_size=2,
        halo_width=0,
    )

    faces = _expected_faces(topology).to(device)

    local_tiles = topology.faces_to_local_tiles(faces)

    assert local_tiles.shape == (
        1,
        topology.tiles_per_rank,
        topology.tile_size,
        topology.tile_size,
    )

    for local_idx, global_index in enumerate(
        range(rank * topology.tiles_per_rank, (rank + 1) * topology.tiles_per_rank)
    ):
        expected_value = float(global_index + 1)
        torch.testing.assert_close(
            local_tiles[0, local_idx],
            torch.full_like(local_tiles[0, local_idx], expected_value),
        )

    # Round trip: gather(scatter) == identity on face layout.
    round_trip = topology.gather_tiles_to_faces(local_tiles)
    torch.testing.assert_close(round_trip, faces)

    dist.barrier()
