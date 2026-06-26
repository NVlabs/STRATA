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
Distributed KNN halo padding for cubed-sphere grids.

Each face is divided into `T x T` tiles; each rank owns 1+ tiles.
All halo cells (both intra-face and cross-face) use KNN interpolation
for uniform load balancing across tiles.

Two-phase workflow:
  1. Precompute: `build_distributed_tile_halo_plan()` builds routing tables and
     per-(source_tile, dest_tile) Regridder instances. Run once, save to disk.
  2. Runtime: `DistributedTileKNNHaloPadding.forward()` fills interiors locally,
     runs local KNN interpolation for halo cells, and communicates via all-to-all.

Setting tile_size=face_size gives 1 tile per face, equivalent to face-level padding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
from earth2grid import KNNS2Interpolator
from earth2grid._regrid import Regridder
from scipy import spatial

from screamcast.cubesphere_transforms import (
    create_padded_faces_batched,
    unstructured_to_6faces,
)


@dataclass
class TileSendEntry:
    """One send entry in a tile-level plan (src_tile -> subset of dst_tile's halo)."""

    src_tile: tuple[int, int, int]
    regridder_state_dict: dict[str, torch.Tensor]
    dst_halo_positions: torch.Tensor


@dataclass
class TileHaloPlan:
    """Precomputed routing plan for tile-level distributed halo padding."""

    face_size: int
    tile_size: int
    tiles_per_dim: int
    halo_width: int
    ne: int
    npg: int
    world_size: int
    tiles_per_rank: int
    # tile_to_rank[(face, ti, tj)] = rank
    tile_to_rank: dict[tuple[int, int, int], int]
    # send_plans[src_rank][(dst_face, dst_ti, dst_tj)] = list of TileSendEntry
    send_plans: dict[int, dict[tuple[int, int, int], list[TileSendEntry]]]
    # recv_counts[(dst_face, dst_ti, dst_tj)][src_rank] = count
    recv_counts: dict[tuple[int, int, int], dict[int, int]]
    # send_sizes[src_rank][dst_rank] = total count
    send_sizes: dict[int, dict[int, int]]
    # recv_sizes[dst_rank][src_rank] = total count
    recv_sizes: dict[int, dict[int, int]]


def _split_source_grid_by_face(
    grid_src: dict, ne: int, npg: int
) -> list[dict[str, np.ndarray]]:
    """
    Split global source grid lon/lat into 6 per-face arrays.

    Args:
        grid_src: {"lon": (6*N*N,), "lat": (6*N*N,)} in degrees
        ne: number of elements per face dimension
        npg: number of physics grid cells per element dimension

    Returns:
        List of 6 dicts, each {"lon": (N*N,), "lat": (N*N,)} in degrees
    """
    face_size = ne * npg
    per_face = []
    for key in ("lon", "lat"):
        arr = grid_src[key].astype(np.float32)
        faces_2d = unstructured_to_6faces(
            torch.from_numpy(arr), ne=ne, npg=npg
        ).numpy()  # (6, face_size, face_size)
        per_face.append(faces_2d.reshape(6, face_size * face_size))

    return [{"lon": per_face[0][f], "lat": per_face[1][f]} for f in range(6)]


def _assign_halo_cells_to_source_faces(
    halo_lon: np.ndarray,
    halo_lat: np.ndarray,
    src_grids_per_face: list[dict[str, np.ndarray]],
) -> np.ndarray:
    """
    For each halo cell, find which source face contains its nearest neighbor.

    Args:
        halo_lon: (Nhalo,) degrees
        halo_lat: (Nhalo,) degrees
        src_grids_per_face: list of 6 dicts with "lon", "lat" arrays (degrees)

    Returns:
        face_assignment: (Nhalo,) int array, values in [0, 5]
    """
    # Build KDTree for each face using 3D Cartesian coords on unit sphere
    trees = []
    for face_grid in src_grids_per_face:
        lon_rad = np.deg2rad(face_grid["lon"])
        lat_rad = np.deg2rad(face_grid["lat"])
        x = np.cos(lat_rad) * np.cos(lon_rad)
        y = np.cos(lat_rad) * np.sin(lon_rad)
        z = np.sin(lat_rad)
        vecs = np.stack([x, y, z], axis=-1)
        trees.append(spatial.KDTree(vecs))

    # Query point coords
    hlon_rad = np.deg2rad(halo_lon)
    hlat_rad = np.deg2rad(halo_lat)
    qx = np.cos(hlat_rad) * np.cos(hlon_rad)
    qy = np.cos(hlat_rad) * np.sin(hlon_rad)
    qz = np.sin(hlat_rad)
    query_vecs = np.stack([qx, qy, qz], axis=-1)

    # Find nearest neighbor distance from each face's tree
    best_face = np.zeros(len(halo_lon), dtype=np.int64)
    best_dist = np.full(len(halo_lon), np.inf)

    for face_id, tree in enumerate(trees):
        dists, _ = tree.query(query_vecs, k=1)
        closer = dists < best_dist
        best_dist[closer] = dists[closer]
        best_face[closer] = face_id

    return best_face


def _halo_dst_flat_indices_per_face(face_size: int, halo_width: int) -> np.ndarray:
    """
    Return flattened halo indices for a SINGLE face, into a flat array of
    size S*S where S = face_size + 2*halo_width.

    Returns:
        (Nhalo_per_face,) int64 array
    """
    S = face_size + 2 * halo_width
    ii, jj = np.meshgrid(np.arange(S), np.arange(S), indexing="ij")
    halo = (
        (ii < halo_width)
        | (ii >= halo_width + face_size)
        | (jj < halo_width)
        | (jj >= halo_width + face_size)
    )
    return (ii[halo] * S + jj[halo]).astype(np.int64)


def _tile_id_to_tuple(tile_id: int, tiles_per_dim: int) -> tuple[int, int, int]:
    """Convert linear tile ID to (face, ti, tj)."""
    tiles_per_face = tiles_per_dim * tiles_per_dim
    face = tile_id // tiles_per_face
    remainder = tile_id % tiles_per_face
    ti = remainder // tiles_per_dim
    tj = remainder % tiles_per_dim
    return (face, ti, tj)


def _tile_tuple_to_id(face: int, ti: int, tj: int, tiles_per_dim: int) -> int:
    """Convert (face, ti, tj) to linear tile ID."""
    return face * tiles_per_dim * tiles_per_dim + ti * tiles_per_dim + tj


def _build_tile_to_rank_map(
    tiles_per_dim: int, world_size: int
) -> dict[tuple[int, int, int], int]:
    """Build mapping from (face, ti, tj) -> rank."""
    total_tiles = 6 * tiles_per_dim * tiles_per_dim
    tiles_per_rank = total_tiles // world_size
    tile_to_rank = {}
    for tile_id in range(total_tiles):
        tile = _tile_id_to_tuple(tile_id, tiles_per_dim)
        tile_to_rank[tile] = tile_id // tiles_per_rank
    return tile_to_rank


def _split_source_grid_by_tile(
    grid_src: dict, ne: int, npg: int, tile_size: int
) -> dict[tuple[int, int, int], dict[str, np.ndarray]]:
    """
    Split global source grid lon/lat into per-tile arrays.

    Returns:
        dict: {(face, ti, tj): {"lon": (tile_size², ), "lat": (tile_size², )}}
    """
    face_size = ne * npg
    tiles_per_dim = face_size // tile_size

    src_per_tile = {}
    for key in ("lon", "lat"):
        arr = grid_src[key].astype(np.float32)
        faces_2d = unstructured_to_6faces(
            torch.from_numpy(arr), ne=ne, npg=npg
        ).numpy()  # (6, face_size, face_size)

        for face in range(6):
            for ti in range(tiles_per_dim):
                for tj in range(tiles_per_dim):
                    tile_key = (face, ti, tj)
                    row_s = ti * tile_size
                    col_s = tj * tile_size
                    tile_data = faces_2d[
                        face, row_s : row_s + tile_size, col_s : col_s + tile_size
                    ].flatten()
                    if tile_key not in src_per_tile:
                        src_per_tile[tile_key] = {}
                    src_per_tile[tile_key][key] = tile_data

    return src_per_tile


def _get_tile_halo_lonlat(
    padded_lon: np.ndarray,
    padded_lat: np.ndarray,
    face: int,
    ti: int,
    tj: int,
    tile_size: int,
    halo_width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract halo cell lon/lat for one tile from the face-padded lon/lat arrays.

    Args:
        padded_lon: (6, S_face, S_face) padded face lon in degrees
        padded_lat: (6, S_face, S_face) padded face lat in degrees
        face, ti, tj: tile identification
        tile_size: tile side length
        halo_width: halo width

    Returns:
        halo_lon: (Nhalo,) lon of halo cells in degrees
        halo_lat: (Nhalo,) lat of halo cells in degrees
        halo_dst_flat: (Nhalo,) flat indices into padded tile (S_tile * S_tile)
    """
    h = halo_width
    T = tile_size
    S_tile = T + 2 * h

    # Tile's padded region in the face-padded array.
    # The face interior starts at index h in the padded face. The tile interior
    # starts at (h + ti*T, h + tj*T). The tile's halo starts h before that,
    # so (h + ti*T - h, h + tj*T - h) = (ti*T, tj*T). The two h's cancel.
    row_start = ti * T
    col_start = tj * T
    tile_pad_lon = padded_lon[
        face, row_start : row_start + S_tile, col_start : col_start + S_tile
    ]
    tile_pad_lat = padded_lat[
        face, row_start : row_start + S_tile, col_start : col_start + S_tile
    ]

    # Halo mask: everything outside the interior [h, h+T) x [h, h+T)
    ii, jj = np.meshgrid(np.arange(S_tile), np.arange(S_tile), indexing="ij")
    halo_mask = (ii < h) | (ii >= h + T) | (jj < h) | (jj >= h + T)

    halo_lon = tile_pad_lon[halo_mask].astype(np.float32)
    halo_lat = tile_pad_lat[halo_mask].astype(np.float32)
    halo_dst_flat = (ii[halo_mask] * S_tile + jj[halo_mask]).astype(np.int64)

    return halo_lon, halo_lat, halo_dst_flat


def _assign_halo_to_source_tiles(
    halo_lon: np.ndarray,
    halo_lat: np.ndarray,
    src_grids_per_face: list[dict[str, np.ndarray]],
    face_trees: list[spatial.KDTree],
    tiles_per_dim: int,
    tile_size: int,
) -> np.ndarray:
    """
    Assign each halo cell to a source tile (face, ti, tj).

    1. Assign to source face via KDTree (reusing existing face-level logic)
    2. Within the assigned face, find nearest grid point → derive tile indices

    Args:
        halo_lon, halo_lat: (Nhalo,) degrees
        src_grids_per_face: list of 6 per-face grid dicts
        face_trees: list of 6 KDTrees built from per-face 2D grid points
        face_lon_2d, face_lat_2d: (6, face_size, face_size) 2D face grids in radians
        tiles_per_dim: T = face_size / tile_size
        tile_size: tile side length

    Returns:
        tile_assignments: (Nhalo, 3) int array, each row is (face, ti, tj)
    """
    # Step 1: assign to face
    face_assignment = _assign_halo_cells_to_source_faces(
        halo_lon, halo_lat, src_grids_per_face
    )

    # Step 2: within the assigned face, find nearest grid point
    hlon_rad = np.deg2rad(halo_lon)
    hlat_rad = np.deg2rad(halo_lat)
    qx = np.cos(hlat_rad) * np.cos(hlon_rad)
    qy = np.cos(hlat_rad) * np.sin(hlon_rad)
    qz = np.sin(hlat_rad)
    query_vecs = np.stack([qx, qy, qz], axis=-1)

    face_size = tiles_per_dim * tile_size
    tile_assignments = np.zeros((len(halo_lon), 3), dtype=np.int64)
    tile_assignments[:, 0] = face_assignment

    for face_id in range(6):
        mask = face_assignment == face_id
        if not mask.any():
            continue
        _, indices = face_trees[face_id].query(query_vecs[mask], k=1)
        # indices is flat index into face_size*face_size
        indices = np.asarray(indices).flatten()
        row = indices // face_size
        col = indices % face_size
        tile_assignments[mask, 1] = row // tile_size
        tile_assignments[mask, 2] = col // tile_size

    return tile_assignments


def _padded_lonlat_from_target_grid(
    grid_tgt: dict,
    face_size: int,
    halo_width: int,
    num_pg_cells: int,
) -> np.ndarray:
    """
    Extract padded face lon/lat from a target SCRIP grid.

    The target grid is expected to be a padded cubed-sphere grid with
    6 * face_size_file^2 cells, where face_size_file >= face_size + 2*halo_width.

    Returns:
        (2, 6, S, S) numpy array where S = face_size + 2*halo_width.
        Index 0 is lon, index 1 is lat, both in degrees.
    """
    lon_tgt = grid_tgt["lon"].astype(np.float32)
    lat_tgt = grid_tgt["lat"].astype(np.float32)

    # Determine the file's face size
    n_cells = len(lon_tgt)
    face_size_file = int((n_cells / 6) ** 0.5)
    if face_size_file * face_size_file * 6 != n_cells:
        raise ValueError(
            f"Target grid has {n_cells} cells, not compatible with 6 square faces"
        )
    ne_file = face_size_file // num_pg_cells

    # Convert to 2D face layout: (6, face_size_file, face_size_file)
    lon_faces_tgt = unstructured_to_6faces(
        torch.from_numpy(lon_tgt), ne=ne_file, npg=num_pg_cells
    ).numpy()
    lat_faces_tgt = unstructured_to_6faces(
        torch.from_numpy(lat_tgt), ne=ne_file, npg=num_pg_cells
    ).numpy()

    # The file may have a larger halo than requested; extract the relevant window
    halo_width_file = (face_size_file - face_size) // 2
    if halo_width > halo_width_file:
        raise ValueError(
            f"Requested halo_width={halo_width} exceeds target grid's "
            f"halo_width_file={halo_width_file}"
        )

    # Window: [halo_width_file - halo_width : halo_width_file + face_size + halo_width]
    i0 = halo_width_file - halo_width
    i1 = halo_width_file + face_size + halo_width
    S = face_size + 2 * halo_width

    padded_lon = lon_faces_tgt[:, i0:i1, i0:i1]  # (6, S, S)
    padded_lat = lat_faces_tgt[:, i0:i1, i0:i1]

    if padded_lon.shape != (6, S, S):
        raise ValueError(f"Expected padded shape (6, {S}, {S}), got {padded_lon.shape}")

    return np.stack([padded_lon, padded_lat], axis=0)  # (2, 6, S, S)


def build_distributed_tile_halo_plan(
    grid_src: dict,
    num_elements: int,
    tile_size: int,
    halo_width: int,
    num_pg_cells: int = 2,
    k: int = 4,
    eps: float = 1e-7,
    world_size: int = 96,
    grid_tgt: dict | None = None,
) -> TileHaloPlan:
    """
    Precompute the distributed tile-level KNN halo routing plan.

    Args:
        grid_src: {"lon": (6*N*N,), "lat": (6*N*N,)} source grid in degrees
        num_elements: ne (elements per face dimension)
        tile_size: tile side length in grid cells
        halo_width: halo width in grid cells
        num_pg_cells: npg
        k: KNN neighbors
        eps: regularization
        world_size: number of GPUs
        grid_tgt: optional {"lon": (...,), "lat": (...,)} padded target grid in
            degrees (e.g., from a SCRIP file like ne1024halo256pg2_scrip.nc).
            When provided, halo cell lon/lat are extracted from this grid via
            KNN interpolation (duo padding), giving more accurate coordinates
            at face corners. When None, falls back to create_padded_faces_batched
            on the source lon/lat.

    Returns:
        plan dict suitable for DistributedTileKNNHaloPadding
    """
    face_size = num_elements * num_pg_cells
    if face_size % tile_size != 0:
        raise ValueError(
            f"face_size={face_size} must be divisible by tile_size={tile_size}"
        )
    tiles_per_dim = face_size // tile_size
    total_tiles = 6 * tiles_per_dim * tiles_per_dim
    if total_tiles % world_size != 0:
        raise ValueError(
            f"total_tiles={total_tiles} must be divisible by world_size={world_size}"
        )
    tiles_per_rank = total_tiles // world_size
    tile_to_rank = _build_tile_to_rank_map(tiles_per_dim, world_size)

    # Build per-tile source grids
    src_per_tile = _split_source_grid_by_tile(
        grid_src, num_elements, num_pg_cells, tile_size
    )

    # Build per-face source grids (for face-level KDTree assignment)
    src_per_face = _split_source_grid_by_face(grid_src, num_elements, num_pg_cells)

    # Build face-level KDTrees for nearest-grid-point lookup
    face_trees = []
    for face_grid in src_per_face:
        lon_rad = np.deg2rad(face_grid["lon"])
        lat_rad = np.deg2rad(face_grid["lat"])
        x = np.cos(lat_rad) * np.cos(lon_rad)
        y = np.cos(lat_rad) * np.sin(lon_rad)
        z = np.sin(lat_rad)
        face_trees.append(spatial.KDTree(np.stack([x, y, z], axis=-1)))

    # Get face-level 2D lon/lat arrays
    lon_faces = unstructured_to_6faces(
        torch.from_numpy(grid_src["lon"].astype(np.float32)),
        ne=num_elements,
        npg=num_pg_cells,
    ).numpy()
    lat_faces = unstructured_to_6faces(
        torch.from_numpy(grid_src["lat"].astype(np.float32)),
        ne=num_elements,
        npg=num_pg_cells,
    ).numpy()

    # Build padded face lon/lat: (6, S_face, S_face) where S_face = face_size + 2*halo_width
    if grid_tgt is not None:
        # Duo padding: use the target SCRIP grid for halo cell coordinates.
        # This gives KNN-interpolated lon/lat at halo positions, which is
        # more accurate than the naive padding heuristic at face corners.
        padded_lonlat = _padded_lonlat_from_target_grid(
            grid_tgt, face_size, halo_width, num_pg_cells
        )
    else:
        # Fallback: pad source lon/lat using face-level copy+rotate+corner heuristic
        stacked = torch.stack(
            [
                torch.from_numpy(lon_faces),
                torch.from_numpy(lat_faces),
            ],
            dim=0,
        )  # (2, 6, face_size, face_size)
        padded_lonlat = create_padded_faces_batched(
            stacked, pad_width=halo_width
        ).numpy()  # (2, 6, S_face, S_face)
    padded_lon = padded_lonlat[0]
    padded_lat = padded_lonlat[1]

    # Build plan
    send_plans: dict[int, dict] = {r: {} for r in range(world_size)}
    recv_counts: dict[tuple, dict] = {}

    for face in range(6):
        for ti in range(tiles_per_dim):
            for tj in range(tiles_per_dim):
                dst_tile = (face, ti, tj)

                # Get halo cell lon/lat for this tile
                halo_lon, halo_lat, halo_dst_flat = _get_tile_halo_lonlat(
                    padded_lon, padded_lat, face, ti, tj, tile_size, halo_width
                )

                if len(halo_lon) == 0:
                    recv_counts[dst_tile] = {}
                    continue

                # Assign halo cells to source tiles
                tile_assignments = _assign_halo_to_source_tiles(
                    halo_lon,
                    halo_lat,
                    src_per_face,
                    face_trees,
                    tiles_per_dim,
                    tile_size,
                )

                # Group by source tile and build regridders
                recv_counts[dst_tile] = {}

                # Get unique source tiles
                unique_src = set(map(tuple, tile_assignments.tolist()))

                for src_tuple in unique_src:
                    src_tile = (int(src_tuple[0]), int(src_tuple[1]), int(src_tuple[2]))
                    src_rank = tile_to_rank[src_tile]

                    mask = np.all(tile_assignments == np.array(src_tuple), axis=1)
                    count = int(mask.sum())

                    if count == 0:
                        continue

                    recv_counts[dst_tile].setdefault(src_rank, 0)
                    recv_counts[dst_tile][src_rank] += count

                    subset_lon = halo_lon[mask]
                    subset_lat = halo_lat[mask]
                    subset_dst = halo_dst_flat[mask]

                    src_grid = src_per_tile[src_tile]
                    regridder = KNNS2Interpolator(
                        torch.from_numpy(src_grid["lon"]),
                        torch.from_numpy(src_grid["lat"]),
                        torch.from_numpy(subset_lon),
                        torch.from_numpy(subset_lat),
                        k=k,
                        eps=eps,
                    )

                    entry = TileSendEntry(
                        src_tile=src_tile,
                        regridder_state_dict=regridder.state_dict(),
                        dst_halo_positions=torch.from_numpy(subset_dst),
                    )

                    # A dst_tile may have multiple source tiles from the same rank
                    if dst_tile not in send_plans[src_rank]:
                        send_plans[src_rank][dst_tile] = []
                    send_plans[src_rank][dst_tile].append(entry)

    # Compute send_sizes and recv_sizes
    send_sizes: dict[int, dict[int, int]] = {}
    recv_sizes: dict[int, dict[int, int]] = {}

    for sr in range(world_size):
        send_sizes[sr] = {}
        for dr in range(world_size):
            total = 0
            for dst_tile, entries in send_plans[sr].items():
                if tile_to_rank[dst_tile] == dr:
                    total += sum(e.dst_halo_positions.shape[0] for e in entries)
            send_sizes[sr][dr] = total

    for dr in range(world_size):
        recv_sizes[dr] = {}
        for sr in range(world_size):
            recv_sizes[dr][sr] = send_sizes[sr][dr]

    return TileHaloPlan(
        face_size=face_size,
        tile_size=tile_size,
        tiles_per_dim=tiles_per_dim,
        halo_width=halo_width,
        ne=num_elements,
        npg=num_pg_cells,
        world_size=world_size,
        tiles_per_rank=tiles_per_rank,
        tile_to_rank=tile_to_rank,
        send_plans=send_plans,
        recv_counts=recv_counts,
        send_sizes=send_sizes,
        recv_sizes=recv_sizes,
    )


class DistributedTileKNNHaloPadding(torch.nn.Module):
    """
    Distributed tile-level halo exchange with KNN interpolation.

    Each rank owns tiles_per_rank tiles. All halo cells (both intra-face
    and cross-face) are filled via KNN interpolation for uniform load balancing.

    Args:
        plan: TileHaloPlan from build_distributed_tile_halo_plan
        rank: this GPU's rank (0-indexed)
        group: optional torch.distributed ProcessGroup
    """

    def __init__(
        self,
        plan: TileHaloPlan,
        rank: int,
        group: dist.ProcessGroup | None = None,
    ):
        super().__init__()
        self.rank = rank
        self.world_size = plan.world_size
        self.tile_size = plan.tile_size
        self.halo_width = plan.halo_width
        self.tiles_per_rank = plan.tiles_per_rank
        self.tiles_per_dim = plan.tiles_per_dim
        self.padded_size = self.tile_size + 2 * self.halo_width
        self.group = group

        total_tiles = 6 * self.tiles_per_dim**2

        # Determine which tiles this rank owns (in linear order)
        self.my_tiles = []
        for tile_id in range(total_tiles):
            tile = _tile_id_to_tuple(tile_id, self.tiles_per_dim)
            if plan.tile_to_rank[tile] == rank:
                self.my_tiles.append(tile)
        self.tile_to_local_idx = {t: i for i, t in enumerate(self.my_tiles)}

        # Load send regridders
        self.send_regridders = torch.nn.ModuleDict()
        self.send_order = (
            []
        )  # [(dst_rank, dst_tile, src_local_idx, regridder_key), ...]

        my_send_plans = plan.send_plans[rank]
        # Order by dest_rank, then by dest_tile
        ordered_dst_tiles = []
        for dr in range(self.world_size):
            for dst_tile, entries in my_send_plans.items():
                if plan.tile_to_rank[dst_tile] == dr:
                    ordered_dst_tiles.append((dr, dst_tile, entries))

        regridder_idx = 0
        for dr, dst_tile, entries in ordered_dst_tiles:
            for entry in entries:
                src_local_idx = self.tile_to_local_idx[entry.src_tile]
                key = f"send_{regridder_idx}"
                regridder = Regridder.from_state_dict(entry.regridder_state_dict)
                self.send_regridders[key] = regridder
                self.send_order.append((dr, dst_tile, src_local_idx, key))
                regridder_idx += 1

        # Load recv scatter indices
        # Order: src_rank ascending, then dst_tile in self.my_tiles order
        recv_buf_idx = 0
        self.recv_order = []  # [(tile_local_idx, buf_key, count), ...]

        for sr in range(self.world_size):
            for tile_local_idx, my_tile in enumerate(self.my_tiles):
                if my_tile not in plan.send_plans[sr]:
                    continue
                entries = plan.send_plans[sr][my_tile]
                for entry in entries:
                    positions = entry.dst_halo_positions
                    buf_key = f"recv_pos_{recv_buf_idx}"
                    self.register_buffer(buf_key, positions.long(), persistent=False)
                    self.recv_order.append(
                        (tile_local_idx, buf_key, positions.shape[0])
                    )
                    recv_buf_idx += 1

        # Precompute split sizes
        self._send_sizes = [plan.send_sizes[rank][dr] for dr in range(self.world_size)]
        self._recv_sizes = [plan.recv_sizes[rank][sr] for sr in range(self.world_size)]

    def forward(self, data_local: torch.Tensor) -> torch.Tensor:
        """
        Distributed tile-level KNN halo padding.

        Args:
            data_local: (..., tiles_per_rank, tile_size, tile_size)

        Returns:
            (..., tiles_per_rank, tile_size + 2h, tile_size + 2h)
        """
        T = self.tile_size
        h = self.halo_width
        S = self.padded_size

        leading_shape = data_local.shape[:-3]
        device = data_local.device
        dtype = data_local.dtype

        # Step 1: Fill interiors
        fill_value = float("nan") if data_local.is_floating_point() else 0
        padded = torch.full(
            (*leading_shape, self.tiles_per_rank, S, S),
            fill_value,
            dtype=dtype,
            device=device,
        )
        padded[..., :, h : h + T, h : h + T] = data_local

        # Step 2: Compute send buffers via KNN interpolation
        send_chunks_by_rank: dict[int, list[torch.Tensor]] = {}

        for dst_rank, dst_tile, src_local_idx, key in self.send_order:
            tile_flat = data_local[..., src_local_idx, :, :].reshape(
                *leading_shape, T * T
            )
            regridder = self.send_regridders[key]
            interpolated = regridder(tile_flat)
            send_chunks_by_rank.setdefault(dst_rank, []).append(interpolated)

        # Pack send buffer
        send_parts = []
        for dr in range(self.world_size):
            if dr in send_chunks_by_rank:
                send_parts.append(torch.cat(send_chunks_by_rank[dr], dim=-1))
            else:
                send_parts.append(data_local.new_empty(*leading_shape, 0))
        send_tensor = torch.cat(send_parts, dim=-1)

        # Step 3: All-to-all communication
        if self.world_size == 1:
            recv_tensor = send_tensor
        else:
            L = int(np.prod(leading_shape)) if len(leading_shape) > 0 else 1
            send_2d = send_tensor.reshape(L, -1)
            send_1d = send_2d.t().contiguous().reshape(-1)

            recv_total = sum(self._recv_sizes)
            recv_1d = torch.empty(recv_total * L, dtype=dtype, device=device)

            dist.all_to_all_single(
                recv_1d,
                send_1d,
                output_split_sizes=[s * L for s in self._recv_sizes],
                input_split_sizes=[s * L for s in self._send_sizes],
                group=self.group,
            )

            recv_tensor = (
                recv_1d.reshape(recv_total, L).t().reshape(*leading_shape, recv_total)
            )

        # Step 4: Scatter received values into padded tiles
        padded_flat_all = padded.reshape(
            *leading_shape, self.tiles_per_rank, S * S
        ).clone()

        offset = 0
        for tile_local_idx, buf_key, count in self.recv_order:
            pos_buf = getattr(self, buf_key)
            chunk = recv_tensor[..., offset : offset + count]
            idx = pos_buf.expand(*leading_shape, count)
            padded_flat_all[..., tile_local_idx, :].scatter_(-1, idx, chunk)
            offset += count

        return padded_flat_all.reshape(*leading_shape, self.tiles_per_rank, S, S)
