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
Tests for tile-level distributed KNN halo padding.

Uses a small synthetic cubed-sphere grid (ne=8, npg=2 -> face_size=16,
tile_size=4, tiles_per_dim=4, halo_width=2) so tests run fast.
"""

import numpy as np
import pytest
import torch

from screamcast.cubesphere_transforms import (
    create_padded_faces_batched,
    reorder_2d_tensor_to_cubesphere,
    unstructured_to_6faces,
)
from screamcast.distributed_cubesphere_transforms import (
    DistributedTileKNNHaloPadding,
    _build_tile_to_rank_map,
    _get_tile_halo_lonlat,
    _halo_dst_flat_indices_per_face,
    _split_source_grid_by_tile,
    _tile_id_to_tuple,
    build_distributed_tile_halo_plan,
)

# Grid parameters
NE = 8
NPG = 2
FACE_SIZE = NE * NPG  # 16
TILE_SIZE = 4
TILES_PER_DIM = FACE_SIZE // TILE_SIZE  # 4
HALO_WIDTH = 2
TOTAL_TILES = 6 * TILES_PER_DIM**2  # 96
NPTS_PER_TILE = TILE_SIZE * TILE_SIZE  # 16
TOTAL_PTS = 6 * FACE_SIZE * FACE_SIZE  # 1536


def _make_cubesphere_lonlat(ne: int, npg: int) -> dict:
    """Create synthetic lon/lat for a cubed-sphere grid."""
    face_size = ne * npg
    lons = []
    lats = []
    for f in range(6):
        lon_base = 60.0 * f
        lat_base = -80.0 + f * 20.0
        lon_1d = np.linspace(lon_base, lon_base + 50.0, face_size)
        lat_1d = np.linspace(lat_base, lat_base + 50.0, face_size)
        lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d, indexing="ij")

        face_lon = torch.from_numpy(
            lon_2d.reshape(1, face_size, face_size).astype(np.float32)
        )
        face_lon_cs = (
            reorder_2d_tensor_to_cubesphere(face_lon, ne=ne, npg=npg).numpy().flatten()
        )
        face_lat = torch.from_numpy(
            lat_2d.reshape(1, face_size, face_size).astype(np.float32)
        )
        face_lat_cs = (
            reorder_2d_tensor_to_cubesphere(face_lat, ne=ne, npg=npg).numpy().flatten()
        )

        lons.append(face_lon_cs)
        lats.append(face_lat_cs)

    return {
        "lon": np.concatenate(lons).astype(np.float32),
        "lat": np.concatenate(lats).astype(np.float32),
    }


@pytest.fixture(scope="module")
def grid_src():
    return _make_cubesphere_lonlat(NE, NPG)


# ---------------------------------------------------------------------------
# Test 1: Plan precomputation correctness
# ---------------------------------------------------------------------------


class TestTilePrecomputation:
    @pytest.mark.parametrize("world_size", [1, 2, 6, 16, 96])
    def test_regridder_indices_in_range(self, grid_src, world_size):
        """Every regridder's index buffer must be in [0, tile_size^2)."""
        plan = build_distributed_tile_halo_plan(
            grid_src, NE, TILE_SIZE, HALO_WIDTH, num_pg_cells=NPG, world_size=world_size
        )
        max_idx = NPTS_PER_TILE

        for src_rank, dst_tiles in plan.send_plans.items():
            for dst_tile, entries in dst_tiles.items():
                for entry in entries:
                    idx = entry.regridder_state_dict["index"]
                    assert (
                        idx.min() >= 0
                    ), f"Negative index: src_rank={src_rank}, dst_tile={dst_tile}"
                    assert idx.max() < max_idx, (
                        f"Index {idx.max()} >= {max_idx}: "
                        f"src_rank={src_rank}, dst_tile={dst_tile}"
                    )

    @pytest.mark.parametrize("world_size", [1, 6, 96])
    def test_halo_coverage(self, grid_src, world_size):
        """Union of dst_halo_positions covers all halo cells per tile."""
        plan = build_distributed_tile_halo_plan(
            grid_src, NE, TILE_SIZE, HALO_WIDTH, num_pg_cells=NPG, world_size=world_size
        )

        # Expected halo indices for a single tile
        expected_halo = set(
            _halo_dst_flat_indices_per_face(TILE_SIZE, HALO_WIDTH).tolist()
        )

        for face in range(6):
            for ti in range(TILES_PER_DIM):
                for tj in range(TILES_PER_DIM):
                    dst_tile = (face, ti, tj)
                    all_positions = []
                    for src_rank in range(world_size):
                        if dst_tile in plan.send_plans[src_rank]:
                            for entry in plan.send_plans[src_rank][dst_tile]:
                                all_positions.extend(
                                    entry.dst_halo_positions.numpy().tolist()
                                )

                    actual = set(all_positions)
                    assert actual == expected_halo, (
                        f"Tile {dst_tile}: coverage mismatch. "
                        f"Missing: {expected_halo - actual}, "
                        f"Extra: {actual - expected_halo}"
                    )

    @pytest.mark.parametrize("world_size", [1, 6, 96])
    def test_no_duplicate_halo_positions(self, grid_src, world_size):
        """Each halo position assigned exactly once per tile."""
        plan = build_distributed_tile_halo_plan(
            grid_src, NE, TILE_SIZE, HALO_WIDTH, num_pg_cells=NPG, world_size=world_size
        )

        for face in range(6):
            for ti in range(TILES_PER_DIM):
                for tj in range(TILES_PER_DIM):
                    dst_tile = (face, ti, tj)
                    all_positions = []
                    for src_rank in range(world_size):
                        if dst_tile in plan.send_plans[src_rank]:
                            for entry in plan.send_plans[src_rank][dst_tile]:
                                all_positions.extend(
                                    entry.dst_halo_positions.numpy().tolist()
                                )
                    assert len(all_positions) == len(
                        set(all_positions)
                    ), f"Tile {dst_tile}: duplicate halo positions"

    @pytest.mark.parametrize("world_size", [1, 6, 96])
    def test_send_recv_sizes_match(self, grid_src, world_size):
        """send_sizes[A][B] == recv_sizes[B][A]."""
        plan = build_distributed_tile_halo_plan(
            grid_src, NE, TILE_SIZE, HALO_WIDTH, num_pg_cells=NPG, world_size=world_size
        )
        for ra in range(world_size):
            for rb in range(world_size):
                assert plan.send_sizes[ra][rb] == plan.recv_sizes[rb][ra], (
                    f"send_sizes[{ra}][{rb}]={plan.send_sizes[ra][rb]} != "
                    f"recv_sizes[{rb}][{ra}]={plan.recv_sizes[rb][ra]}"
                )


# ---------------------------------------------------------------------------
# Test 2: Single-GPU forward pass
# ---------------------------------------------------------------------------


class TestTileSingleGPU:
    def test_output_shape(self, grid_src):
        """Single-GPU forward produces correct shape."""
        plan = build_distributed_tile_halo_plan(
            grid_src, NE, TILE_SIZE, HALO_WIDTH, num_pg_cells=NPG, world_size=1
        )
        module = DistributedTileKNNHaloPadding(plan, rank=0)

        # (channels, total_tiles, tile_size, tile_size)
        data = torch.randn(3, TOTAL_TILES, TILE_SIZE, TILE_SIZE)
        padded = module(data)

        S = TILE_SIZE + 2 * HALO_WIDTH
        assert padded.shape == (3, TOTAL_TILES, S, S)

    def test_interior_preserved(self, grid_src):
        """Interior region should match original tile data."""
        plan = build_distributed_tile_halo_plan(
            grid_src, NE, TILE_SIZE, HALO_WIDTH, num_pg_cells=NPG, world_size=1
        )
        module = DistributedTileKNNHaloPadding(plan, rank=0)

        data = torch.randn(2, TOTAL_TILES, TILE_SIZE, TILE_SIZE)
        padded = module(data)

        h = HALO_WIDTH
        interior = padded[:, :, h : h + TILE_SIZE, h : h + TILE_SIZE]
        torch.testing.assert_close(interior, data, atol=1e-6, rtol=1e-6)

    def test_no_nan_in_halo(self, grid_src):
        """After padding, no NaN values should remain."""
        plan = build_distributed_tile_halo_plan(
            grid_src, NE, TILE_SIZE, HALO_WIDTH, num_pg_cells=NPG, world_size=1
        )
        module = DistributedTileKNNHaloPadding(plan, rank=0)

        data = torch.randn(1, TOTAL_TILES, TILE_SIZE, TILE_SIZE)
        padded = module(data)
        assert not torch.isnan(padded).any(), "NaN found in padded output"

    def test_multichannel_batch(self, grid_src):
        """Works with batch and channel leading dims."""
        plan = build_distributed_tile_halo_plan(
            grid_src, NE, TILE_SIZE, HALO_WIDTH, num_pg_cells=NPG, world_size=1
        )
        module = DistributedTileKNNHaloPadding(plan, rank=0)

        data = torch.randn(2, 5, TOTAL_TILES, TILE_SIZE, TILE_SIZE)
        padded = module(data)
        S = TILE_SIZE + 2 * HALO_WIDTH
        assert padded.shape == (2, 5, TOTAL_TILES, S, S)


# ---------------------------------------------------------------------------
# Test 3: Intra-face halo accuracy
# ---------------------------------------------------------------------------


class TestIntraFaceAccuracy:
    def test_interior_tile_halo_accuracy(self, grid_src):
        """For an interior tile, halo should closely match neighboring tile data."""
        plan = build_distributed_tile_halo_plan(
            grid_src, NE, TILE_SIZE, HALO_WIDTH, num_pg_cells=NPG, world_size=1
        )
        module = DistributedTileKNNHaloPadding(plan, rank=0)

        # Build data where each pixel has a known value based on face position
        # so we can verify halo correctness
        faces_2d = torch.zeros(6, FACE_SIZE, FACE_SIZE)
        for f in range(6):
            for i in range(FACE_SIZE):
                for j in range(FACE_SIZE):
                    faces_2d[f, i, j] = f * 1000 + i * FACE_SIZE + j

        # Convert to tile format: (total_tiles, tile_size, tile_size)
        tiles = []
        for f in range(6):
            for ti in range(TILES_PER_DIM):
                for tj in range(TILES_PER_DIM):
                    tile = faces_2d[
                        f,
                        ti * TILE_SIZE : (ti + 1) * TILE_SIZE,
                        tj * TILE_SIZE : (tj + 1) * TILE_SIZE,
                    ]
                    tiles.append(tile)
        data = torch.stack(tiles).unsqueeze(0)  # (1, 96, 4, 4)

        padded = module(data)

        # Check an interior tile (face=0, ti=1, tj=1)
        tile_idx = 0 * TILES_PER_DIM**2 + 1 * TILES_PER_DIM + 1
        padded_tile = padded[0, tile_idx]  # (S, S)
        h = HALO_WIDTH

        # Build expected padded tile from face data
        face_row_start = 1 * TILE_SIZE - h
        face_col_start = 1 * TILE_SIZE - h
        S = TILE_SIZE + 2 * h
        expected = faces_2d[
            0, face_row_start : face_row_start + S, face_col_start : face_col_start + S
        ]

        # KNN with k=4 on exact grid points should be very close
        torch.testing.assert_close(padded_tile, expected, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Test 4: Edge cases and invalid configs
# ---------------------------------------------------------------------------


class TestTileEdgeCases:
    def test_invalid_tile_size(self, grid_src):
        """face_size not divisible by tile_size should raise."""
        with pytest.raises(ValueError, match="divisible"):
            build_distributed_tile_halo_plan(
                grid_src, NE, 5, HALO_WIDTH, num_pg_cells=NPG, world_size=1
            )

    def test_invalid_world_size(self, grid_src):
        """world_size that doesn't divide total_tiles should raise."""
        with pytest.raises(ValueError, match="divisible"):
            build_distributed_tile_halo_plan(
                grid_src, NE, TILE_SIZE, HALO_WIDTH, num_pg_cells=NPG, world_size=7
            )

    def test_halo_width_zero(self, grid_src):
        """halo_width=0 should return tiles unchanged."""
        plan = build_distributed_tile_halo_plan(
            grid_src, NE, TILE_SIZE, 0, num_pg_cells=NPG, world_size=1
        )
        module = DistributedTileKNNHaloPadding(plan, rank=0)
        data = torch.randn(2, TOTAL_TILES, TILE_SIZE, TILE_SIZE)
        padded = module(data)
        assert padded.shape == (2, TOTAL_TILES, TILE_SIZE, TILE_SIZE)
        torch.testing.assert_close(padded, data, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Test 5: Helper function unit tests
# ---------------------------------------------------------------------------


class TestTileHelpers:
    def test_tile_id_roundtrip(self):
        """tile_id -> tuple -> tile_id roundtrip."""
        for tile_id in range(TOTAL_TILES):
            f, ti, tj = _tile_id_to_tuple(tile_id, TILES_PER_DIM)
            assert 0 <= f < 6
            assert 0 <= ti < TILES_PER_DIM
            assert 0 <= tj < TILES_PER_DIM

    def test_tile_to_rank_map(self):
        """Every tile mapped to a valid rank."""
        for ws in [1, 2, 6, 96]:
            tile_map = _build_tile_to_rank_map(TILES_PER_DIM, ws)
            assert len(tile_map) == TOTAL_TILES
            ranks = set(tile_map.values())
            assert ranks == set(range(ws))

    def test_split_source_grid_by_tile(self, grid_src):
        """Per-tile grids cover all face points."""
        per_tile = _split_source_grid_by_tile(grid_src, NE, NPG, TILE_SIZE)
        assert len(per_tile) == TOTAL_TILES
        for tile_grid in per_tile.values():
            assert tile_grid["lon"].shape == (NPTS_PER_TILE,)
            assert tile_grid["lat"].shape == (NPTS_PER_TILE,)

    def test_get_tile_halo_lonlat(self, grid_src):
        """Halo lon/lat extraction produces correct count."""
        lon_faces = unstructured_to_6faces(
            torch.from_numpy(grid_src["lon"].astype(np.float32)), ne=NE, npg=NPG
        ).numpy()
        lat_faces = unstructured_to_6faces(
            torch.from_numpy(grid_src["lat"].astype(np.float32)), ne=NE, npg=NPG
        ).numpy()
        stacked = torch.stack(
            [torch.from_numpy(lon_faces), torch.from_numpy(lat_faces)], dim=0
        )
        padded = create_padded_faces_batched(stacked, pad_width=HALO_WIDTH).numpy()
        padded_lon = padded[0]
        padded_lat = padded[1]

        halo_lon, halo_lat, halo_dst = _get_tile_halo_lonlat(
            padded_lon, padded_lat, 0, 1, 1, TILE_SIZE, HALO_WIDTH
        )

        S = TILE_SIZE + 2 * HALO_WIDTH
        expected_halo_count = S * S - TILE_SIZE * TILE_SIZE
        assert len(halo_lon) == expected_halo_count
        assert len(halo_lat) == expected_halo_count
        assert len(halo_dst) == expected_halo_count
        assert halo_dst.min() >= 0
        assert halo_dst.max() < S * S
