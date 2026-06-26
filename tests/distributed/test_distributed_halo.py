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

from screamcast.distributed_halo import (
    DistributedTileKNNHaloPadding_AllGather,
    NormalizedHaloAdjoint,
    TileTopology,
)


def test_distributed_halo_exchange_main_path(distributed_cuda_context):
    rank, world_size, device = distributed_cuda_context
    topology = TileTopology(
        world_size=world_size,
        rank=rank,
        face_size=8,
        tile_size=2,
        halo_width=1,
    )

    numtiles = len(topology.local_tiles)
    b = 1
    c = 1
    np = topology.face_size + 2 * topology.halo_width
    lon = torch.randn([6, np, np])
    lat = torch.randn([6, np, np]) * 180 - 90
    local_tiles = torch.randn([b, c, numtiles, topology.tile_size, topology.tile_size])

    halo_exchange = DistributedTileKNNHaloPadding_AllGather.from_padded_face_grid(
        topology=topology,
        lon=lon,
        lat=lat,
        pad_width_data=topology.halo_width,
        device="cuda",
    )

    local_tiles = local_tiles.cuda()
    halo_exchange.cuda()
    padded = halo_exchange(local_tiles)
    assert padded.shape == (
        1,
        local_tiles.shape[0],
        topology.tiles_per_rank,
        topology.padded_tile_size,
        topology.padded_tile_size,
    )
    torch.testing.assert_close(topology.crop(padded), local_tiles, atol=1e-3, rtol=1e-3)
    dist.barrier()


def test_halo_transpose(distributed_cuda_context):
    rank, world_size, device = distributed_cuda_context
    topology = TileTopology(
        world_size=world_size,
        rank=rank,
        face_size=8,
        tile_size=2,
        halo_width=1,
    )

    numtiles = len(topology.local_tiles)
    b = 1
    c = 1
    np = topology.face_size + 2 * topology.halo_width
    lon = torch.randn([6, np, np])
    lat = torch.randn([6, np, np]) * 180 - 90
    local_tiles = torch.randn([b, c, numtiles, topology.tile_size, topology.tile_size])

    halo_exchange = DistributedTileKNNHaloPadding_AllGather.from_padded_face_grid(
        topology=topology,
        lon=lon,
        lat=lat,
        pad_width_data=topology.halo_width,
        device="cuda",
    )

    local_tiles = local_tiles.to(device)
    halo_exchange.to(device)
    halo_pseudoinverse = NormalizedHaloAdjoint(halo_exchange)

    with torch.inference_mode():
        padded = halo_exchange(local_tiles)
        unpadded = halo_pseudoinverse(padded)
    assert unpadded.shape == local_tiles.shape
    assert torch.isfinite(unpadded).all()

    # Constant fields round-trip exactly through any normalized interpolation.
    const = torch.ones_like(local_tiles)
    with torch.inference_mode():
        round_trip_const = halo_pseudoinverse(halo_exchange(const))
    torch.testing.assert_close(round_trip_const, const, atol=1e-3, rtol=1e-3)

    dist.barrier()
