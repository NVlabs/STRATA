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
"""Distributed cubed-sphere halo exchange based on all-gather.

Due to the expense of creating the distributed plan,
./distributed_cubesphere_transforms.py was not feasible for the distributed
tiled rollout at full resolution, so this all gather implementation is used
instead, and seems to have minimal overead up to at least 32 GPUS.

Example:
    >>> topology = TileTopology(
    ...     world_size=world_size,
    ...     rank=rank,
    ...     face_size=2048,
    ...     tile_size=128,
    ...     halo_width=32,
    ... )
    >>> print(lon_faces_padded.shape)
    (6, 2112, 2112)
    >>> print(lat_faces_padded.shape)
    (6, 2112, 2112)
    >>> halo_exchange = DistributedTileKNNHaloPadding_AllGather.from_padded_face_grid(
    ...     topology=topology,
    ...     lon=lon_faces_padded,
    ...     lat=lat_faces_padded,
    ...     pad_width_data=pad_width_data,
    ...     device=device,
    ... )
    >>> padded_tiles = halo_exchange(local_tiles)
    >>> interior_tiles = topology.crop(padded_tiles)
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import cached_property
from typing import Callable, TypeAlias

import numpy as np
import torch
from earth2grid import KNNS2Interpolator

Tile: TypeAlias = tuple[int, int, int]

# A regional run's region is a rectangle on a "plane": one face, or an equatorial face
# (0-3) and one neighbour joined across their shared edge: the next equatorial face to
# the east (rows line up one to one), the north polar face 5 (its edge that meets
# equatorial face f's north edge lies there after f quarter turns) or the south polar
# face 4 (below face f after -f quarter turns). Each block is ``(face, block row, block
# col, k)``: the face's data sits in the plane as ``torch.rot90(data, k)``. The first
# block is the plane's base face; below a south join it sits at block row 1.
# tests/test_regional.py checks the joins on the grid.
PlaneBlock: TypeAlias = tuple[int, int, int, int]
NORTH_POLAR_FACE = 5
SOUTH_POLAR_FACE = 4
# (normal, row direction, column direction) of each face of ne1024pg2: pixel (p, q) of
# a face lies toward normal + x_p row + y_q column, x and y running from -1 to 1 along
# the face. Rows run north on the equatorial faces. tests/test_regional.py checks them on
# the grid.
FACE_AXES = (
    ((1, 0, 0), (0, 0, 1), (0, 1, 0)),
    ((0, 1, 0), (0, 0, 1), (-1, 0, 0)),
    ((-1, 0, 0), (0, 0, 1), (0, -1, 0)),
    ((0, -1, 0), (0, 0, 1), (1, 0, 0)),
    ((0, 0, -1), (1, 0, 0), (0, 1, 0)),
    ((0, 0, 1), (-1, 0, 0), (0, 1, 0)),
)


def plane_blocks(
    face: int, north: bool = False, east: bool = False, south: bool = False
) -> list[PlaneBlock]:
    """The faces of the plane based on ``face``, with its north, east or south
    neighbour."""
    if north + east + south > 1:
        raise ValueError("a region plane joins one neighbour, north, east or south")
    if (north or east or south) and not 0 <= face < 4:
        raise ValueError(f"only an equatorial face (0-3) joins a neighbour, not {face}")
    if south:
        return [(face, 1, 0, 0), (SOUTH_POLAR_FACE, 0, 0, -face % 4)]
    blocks = [(face, 0, 0, 0)]
    if east:
        blocks.append(((face + 1) % 4, 0, 1, 0))
    if north:
        blocks.append((NORTH_POLAR_FACE, 1, 0, face))
    return blocks


def base_offset(blocks: list[PlaneBlock], n: int) -> tuple[int, int]:
    """Plane position of the base face's ``(0, 0)``, ``n`` per face side: a
    ``--region`` spec counts from there, so a south join's rows are negative."""
    _, br, bc, _ = blocks[0]
    return br * n, bc * n


def tile_corners(tile: Tile, tiles_per_dim: int) -> set[tuple[int, int, int]]:
    """The four corners of ``tile`` as integer points of the cube ``[-T, T]^3`` (``T``
    tiles per face side), from ``FACE_AXES``: tiles of any faces touch where they share
    a corner. The equiangular grid puts a face's tile edges at the same angles on every
    face, so the integer lattice labels the shared edges alike."""
    face, ti, tj = tile
    t = tiles_per_dim
    normal, row, col = FACE_AXES[face]
    return {
        tuple(
            t * a + (2 * r - t) * b + (2 * c - t) * d
            for a, b, d in zip(normal, row, col)
        )
        for r in (ti, ti + 1)
        for c in (tj, tj + 1)
    }


def touching_tiles(tiles: list[Tile], tiles_per_dim: int) -> list[Tile]:
    """The tiles, not in ``tiles``, that share an edge or a corner with one of them, on
    any face (sorted)."""
    own = set(tiles)
    corners = set().union(*(tile_corners(t, tiles_per_dim) for t in tiles))
    return [
        t
        for t in itertools.product(range(6), range(tiles_per_dim), range(tiles_per_dim))
        if t not in own and tile_corners(t, tiles_per_dim) & corners
    ]


def region_plane_for(tiles: list[Tile], tiles_per_dim: int) -> list[PlaneBlock]:
    """The plane (``plane_blocks``) of a region made of ``tiles``.

    Tiles on two faces: the equatorial face joined to that neighbour. Tiles on one
    face: that face, unless the region is flush with an equatorial face's north, east or
    west edge, where its one-tile ring lies on the neighbour; the plane then joins that
    neighbour (the west neighbour as the base of an east join). The ring of a region
    flush with a south edge, with two edges (a cube corner) or with a polar face's edge
    is not on any plane; ``TileTopology.ring_tiles`` then takes it from every face it
    touches.
    """
    faces = sorted({f for f, _, _ in tiles})
    if len(faces) == 1:
        face, last = faces[0], tiles_per_dim - 1
        if face < 4:
            if any(i == last for _, i, _ in tiles):
                return plane_blocks(face, north=True)
            if any(j == last for _, _, j in tiles):
                return plane_blocks(face, east=True)
            if any(j == 0 for _, _, j in tiles):
                return plane_blocks((face - 1) % 4, east=True)
        return plane_blocks(face)
    if len(faces) == 2:
        a, b = faces
        if b == NORTH_POLAR_FACE and a < 4:
            return plane_blocks(a, north=True)
        if b == SOUTH_POLAR_FACE and a < 4:
            return plane_blocks(a, south=True)
        if b < 4 and (a + 1) % 4 == b:
            return plane_blocks(a, east=True)
        if b < 4 and (b + 1) % 4 == a:
            return plane_blocks(b, east=True)
    raise ValueError(
        f"active tiles span faces {faces}: need one face, or an equatorial face "
        "and its east neighbour or a polar face"
    )


def plane_shape(blocks: list[PlaneBlock], n: int) -> tuple[int, int]:
    """Rows and columns of the plane, ``n`` per face side."""
    return (
        n * (1 + max(br for _, br, _, _ in blocks)),
        n * (1 + max(bc for _, _, bc, _ in blocks)),
    )


def rot_position(p: int, q: int, n: int, k: int) -> tuple[int, int]:
    """Where element ``(p, q)`` of an ``n x n`` array lands in ``torch.rot90(a, k)``."""
    for _ in range(k % 4):
        p, q = n - 1 - q, p
    return p, q


def to_plane(
    blocks: list[PlaneBlock], face: int, p: int, q: int, n: int
) -> tuple[int, int]:
    """Plane position of ``(p, q)`` on ``face`` (``n`` per face side)."""
    for f, br, bc, k in blocks:
        if f == face:
            a, b = rot_position(p, q, n, k)
            return br * n + a, bc * n + b
    raise ValueError(f"face {face} is not in the plane {blocks}")


def from_plane(
    blocks: list[PlaneBlock], P: int, Q: int, n: int
) -> tuple[int, int, int]:
    """``(face, p, q)`` at plane position ``(P, Q)``."""
    for f, br, bc, k in blocks:
        if P // n == br and Q // n == bc and P >= 0 and Q >= 0:
            return (f, *rot_position(P - br * n, Q - bc * n, n, -k))
    raise ValueError(f"plane position {(P, Q)} lies outside the plane {blocks}")


def faces_to_plane(x: torch.Tensor, blocks: list[PlaneBlock]) -> torch.Tensor:
    """``(..., 6, n, n)`` face-major fields as the ``(..., rows, cols)`` plane."""
    n = x.shape[-1]
    out = x.new_zeros(*x.shape[:-3], *plane_shape(blocks, n))
    for f, br, bc, k in blocks:
        out[..., br * n : (br + 1) * n, bc * n : (bc + 1) * n] = torch.rot90(
            x[..., f, :, :], k, dims=(-2, -1)
        )
    return out


def plane_parts(
    blocks: list[PlaneBlock], box: tuple[int, int, int, int], n: int
) -> list[tuple[int, tuple[int, int, int, int], tuple[int, int, int, int], int]]:
    """The plane rectangle ``box = (r0, r1, c0, c1)`` split by face:
    ``(face, face rows/cols (half-open), plane rows/cols, k)``; the plane part is the
    face part turned ``k`` quarter turns (``torch.rot90(face part, k)``)."""
    r0, r1, c0, c1 = box
    rows, cols = plane_shape(blocks, n)
    if not (0 <= r0 < r1 <= rows and 0 <= c0 < c1 <= cols):
        raise ValueError(f"box {box} is not inside the plane ({rows} x {cols})")
    parts = []
    for f, br, bc, k in blocks:
        pr0, pr1 = max(r0, br * n), min(r1, (br + 1) * n)
        pc0, pc1 = max(c0, bc * n), min(c1, (bc + 1) * n)
        if pr0 >= pr1 or pc0 >= pc1:
            continue
        corners = [
            rot_position(a - br * n, b - bc * n, n, -k)
            for a in (pr0, pr1 - 1)
            for b in (pc0, pc1 - 1)
        ]
        fr = [p for p, _ in corners]
        fc = [q for _, q in corners]
        parts.append(
            (
                f,
                (min(fr), max(fr) + 1, min(fc), max(fc) + 1),
                (pr0, pr1, pc0, pc1),
                k,
            )
        )
    return parts


@dataclass(frozen=True)
class RingPiece:
    """A rectangle of tiles on one plane that the anchored ring is smoothed on: plane
    pixels ``origin`` to ``origin + size`` of ``blocks``. ``tiles`` are the ring tiles
    it supplies (in ``ring_tiles`` order)."""

    blocks: tuple[PlaneBlock, ...]
    origin: tuple[int, int]
    size: tuple[int, int]
    tiles: tuple[Tile, ...]


@dataclass(frozen=True)
class TileTopology:
    """Describe the distributed cubed-sphere tile decomposition.

    ``active_tiles`` restricts the rollout to a region (regional mode): only
    those tiles are predicted and distributed over the ranks, in the given
    order; ``global_tiles``/``total_tiles``/``local_tiles`` then refer to the
    active set. The region must be a full rectangle of tiles on a region plane
    (one face, or an equatorial face joined to its east, north or south neighbour,
    see ``region_plane_for``). Its ``ring_tiles`` are the tiles that touch it, on the
    plane or, at a face edge the plane does not cross, on the faces beyond; the ring is
    not predicted, it only supplies the halo sources at the region boundary. ``None``
    is the full cube.
    """

    world_size: int
    rank: int
    face_size: int
    tile_size: int
    halo_width: int
    n_faces: int = 6
    active_tiles: tuple[Tile, ...] | None = None

    def __post_init__(self) -> None:
        if self.face_size % self.tile_size != 0:
            raise ValueError(
                f"face_size={self.face_size} must be divisible by tile_size={self.tile_size}"
            )
        if self.active_tiles is not None:
            # region_box validates the rectangle.
            self.region_box  # noqa: B018
        if self.total_tiles % self.world_size != 0:
            raise ValueError(
                f"Number of tiles ({self.total_tiles}) must be divisible by "
                f"world_size ({self.world_size})"
            )

    @property
    def tiles_per_dim(self) -> int:
        return self.face_size // self.tile_size

    @property
    def total_tiles(self) -> int:
        if self.active_tiles is not None:
            return len(self.active_tiles)
        return self.n_faces * self.tiles_per_dim * self.tiles_per_dim

    @property
    def is_regional(self) -> bool:
        return self.active_tiles is not None

    @property
    def tiles_per_rank(self) -> int:
        return self.total_tiles // self.world_size

    @property
    def padded_tile_size(self) -> int:
        return self.tile_size + 2 * self.halo_width

    @property
    def global_tiles(self) -> list[Tile]:
        """Predicted tiles in rank order: the active tiles, or the full cube."""
        if self.active_tiles is not None:
            return list(self.active_tiles)
        return self.full_cube_tiles

    @property
    def full_cube_tiles(self) -> list[Tile]:
        return list(
            itertools.product(
                range(self.n_faces),
                range(self.tiles_per_dim),
                range(self.tiles_per_dim),
            )
        )

    @cached_property
    def region_plane(self) -> list[PlaneBlock]:
        """The plane (``plane_blocks``) the active tiles lie on."""
        if self.active_tiles is None:
            raise ValueError("region_plane is only defined for a regional topology")
        return region_plane_for(list(self.active_tiles), self.tiles_per_dim)

    def tile_plane_position(
        self, tile: Tile, blocks: list[PlaneBlock] | None = None
    ) -> tuple[int, int, int]:
        """``(row, col, k)`` of ``tile`` on the plane ``blocks`` (default the region
        plane) in tile units; its data lies there turned ``k`` quarter turns. The full
        cube without ``blocks``: the tile's own face."""
        face, ti, tj = tile
        if blocks is None:
            if self.active_tiles is None:
                return ti, tj, 0
            blocks = self.region_plane
        for f, _, _, k in blocks:
            if f == face:
                return (*to_plane(blocks, face, ti, tj, self.tiles_per_dim), k)
        raise ValueError(f"tile {tile} is not on the plane {blocks}")

    @cached_property
    def region_box(self) -> tuple[int, int, int, int, int]:
        """``(face, i0, i1, j0, j1)``: the active region in tile units (half-open) of
        the plane based on ``face`` (``i1`` or ``j1`` beyond the face's tiles reach into
        its north or east neighbour)."""
        if self.active_tiles is None:
            raise ValueError("region_box is only defined for a regional topology")
        tiles = self.active_tiles
        blocks = self.region_plane
        pos = [self.tile_plane_position(t)[:2] for t in tiles]
        i0 = min(i for i, _ in pos)
        i1 = max(i for i, _ in pos) + 1
        j0 = min(j for _, j in pos)
        j1 = max(j for _, j in pos) + 1
        if sorted(pos) != sorted(
            itertools.product(range(i0, i1), range(j0, j1))
        ) or len(set(tiles)) != len(tiles):
            raise ValueError("active tiles must be a full rectangle without repeats")
        return blocks[0][0], i0, i1, j0, j1

    @property
    def region_spec(self) -> str:
        """The region as a ``--region`` spec, ``face,i0,i1,j0,j1`` counted from the base
        face's ``(0, 0)`` (``base_offset``)."""
        face, i0, i1, j0, j1 = self.region_box
        di, dj = base_offset(self.region_plane, self.tiles_per_dim)
        return f"{face},{i0 - di},{i1 - di},{j0 - dj},{j1 - dj}"

    @cached_property
    def ring_on_plane(self) -> bool:
        """Whether the one-tile ring around the region lies on the region plane."""
        _, i0, i1, j0, j1 = self.region_box
        rows, cols = plane_shape(self.region_plane, self.tiles_per_dim)
        return i0 >= 1 and j0 >= 1 and i1 <= rows - 1 and j1 <= cols - 1

    @cached_property
    def ring_tiles(self) -> list[Tile]:
        """The tiles that touch the region, empty for the full cube: the one-tile ring
        around it row-major on the plane, or, when that ring leaves the plane (a region
        flush with a south edge, a polar face's edge or at a cube corner), every tile of
        any face sharing an edge or corner with it (``touching_tiles``, sorted)."""
        if self.active_tiles is None:
            return []
        if not self.ring_on_plane:
            return touching_tiles(list(self.active_tiles), self.tiles_per_dim)
        _, i0, i1, j0, j1 = self.region_box
        return [
            from_plane(self.region_plane, i, j, self.tiles_per_dim)
            for i in range(i0 - 1, i1 + 1)
            for j in range(j0 - 1, j1 + 1)
            if not (i0 <= i < i1 and j0 <= j < j1)
        ]

    @cached_property
    def ring_pieces(self) -> list[RingPiece]:
        """The rectangles the anchored ring is smoothed on: the region and its ring on
        the plane, or, for a ring off the plane, on each face it reaches the smallest
        rectangle of tiles holding the region's and the ring's tiles there (it may hold
        other tiles too, ``boundary_tiles``)."""
        if self.active_tiles is None:
            return []
        ts = self.tile_size
        if self.ring_on_plane:
            _, i0, i1, j0, j1 = self.region_box
            return [
                RingPiece(
                    blocks=tuple(self.region_plane),
                    origin=((i0 - 1) * ts, (j0 - 1) * ts),
                    size=((i1 - i0 + 2) * ts, (j1 - j0 + 2) * ts),
                    tiles=tuple(self.ring_tiles),
                )
            ]
        everything = [*self.active_tiles, *self.ring_tiles]
        pieces = []
        for face in sorted({f for f, _, _ in self.ring_tiles}):
            rows = [i for f, i, _ in everything if f == face]
            cols = [j for f, _, j in everything if f == face]
            pieces.append(
                RingPiece(
                    blocks=((face, 0, 0, 0),),
                    origin=(min(rows) * ts, min(cols) * ts),
                    size=(
                        (max(rows) - min(rows) + 1) * ts,
                        (max(cols) - min(cols) + 1) * ts,
                    ),
                    tiles=tuple(t for t in self.ring_tiles if t[0] == face),
                )
            )
        return pieces

    @cached_property
    def boundary_tiles(self) -> list[Tile]:
        """``ring_tiles``, then the other tiles outside the region that a ring piece
        holds: the tiles an anchored ring needs the driver on."""
        if self.active_tiles is None or self.ring_on_plane:
            return list(self.ring_tiles)
        ts = self.tile_size
        known = {*self.active_tiles, *self.ring_tiles}
        extra = [
            (piece.blocks[0][0], i, j)
            for piece in self.ring_pieces
            for i in range(
                piece.origin[0] // ts, (piece.origin[0] + piece.size[0]) // ts
            )
            for j in range(
                piece.origin[1] // ts, (piece.origin[1] + piece.size[1]) // ts
            )
            if (piece.blocks[0][0], i, j) not in known
        ]
        return [*self.ring_tiles, *extra]

    def region_pixel_box(self) -> tuple[int, int, int, int, int]:
        """``(face, r0, r1, c0, c1)``: the region in plane pixels (half-open)."""
        face, i0, i1, j0, j1 = self.region_box
        ts = self.tile_size
        return face, i0 * ts, i1 * ts, j0 * ts, j1 * ts

    def tiles_to_box(
        self,
        x: torch.Tensor,
        tiles: list[Tile],
        origin: tuple[int, int],
        size: tuple[int, int],
        blocks: list[PlaneBlock] | None = None,
    ) -> torch.Tensor:
        """Place ``(..., len(tiles), ts, ts)`` tiles into a ``(..., *size)`` box whose
        pixel (0, 0) is pixel ``origin`` of the plane ``blocks`` (default the region
        plane); the parts of tiles outside are dropped."""
        ts = self.tile_size
        out = x.new_zeros(*x.shape[:-3], *size)
        for n, tile in enumerate(tiles):
            ti, tj, k = self.tile_plane_position(tile, blocks)
            r, c = ti * ts - origin[0], tj * ts - origin[1]
            rs, cs = max(r, 0), max(c, 0)
            re, ce = min(r + ts, size[0]), min(c + ts, size[1])
            if rs < re and cs < ce:
                data = x[..., n, :, :]
                if k:
                    data = torch.rot90(data, k, dims=(-2, -1))
                out[..., rs:re, cs:ce] = data[..., rs - r : re - r, cs - c : ce - c]
        return out

    def box_to_tiles(
        self,
        box: torch.Tensor,
        tiles: list[Tile],
        origin: tuple[int, int],
        blocks: list[PlaneBlock] | None = None,
    ) -> torch.Tensor:
        """Cut ``(..., len(tiles), ts, ts)`` tiles out of a ``(..., H, W)`` box whose
        pixel (0, 0) is pixel ``origin`` of the plane ``blocks`` (default the region
        plane); tile pixels outside the box are 0."""
        ts = self.tile_size
        H, W = box.shape[-2:]
        out = box.new_zeros(*box.shape[:-2], len(tiles), ts, ts)
        for n, tile in enumerate(tiles):
            ti, tj, k = self.tile_plane_position(tile, blocks)
            r, c = ti * ts - origin[0], tj * ts - origin[1]
            rs, cs = max(r, 0), max(c, 0)
            re, ce = min(r + ts, H), min(c + ts, W)
            if rs < re and cs < ce:
                if k:
                    # in the plane's orientation, then turned back to the tile's
                    data = box.new_zeros(*box.shape[:-2], ts, ts)
                    data[..., rs - r : re - r, cs - c : ce - c] = box[..., rs:re, cs:ce]
                    out[..., n, :, :] = torch.rot90(data, -k, dims=(-2, -1))
                else:
                    out[..., n, rs - r : re - r, cs - c : ce - c] = box[
                        ..., rs:re, cs:ce
                    ]
        return out

    @property
    def local_tiles(self) -> list[Tile]:
        start = self.rank * self.tiles_per_rank
        end = start + self.tiles_per_rank
        return self.global_tiles[start:end]

    def interior_origin(self, tile: Tile) -> tuple[int, int]:
        _, tile_i, tile_j = tile
        return tile_i * self.tile_size, tile_j * self.tile_size

    def interior_coords(
        self,
        base_coords: dict,
        *,
        time: np.ndarray | None = None,
    ) -> dict:
        coords = dict(base_coords)
        if time is not None:
            coords["time"] = time
        coords["face"] = np.arange(len(self.local_tiles))
        coords["x"] = np.arange(self.tile_size)
        coords["y"] = np.arange(self.tile_size)
        return coords

    def pad_coords(self, coords: dict) -> dict:
        padded = dict(coords)
        padded["face"] = np.arange(len(self.local_tiles))
        padded["x"] = np.arange(self.padded_tile_size)
        padded["y"] = np.arange(self.padded_tile_size)
        return padded

    def crop_coords(self, coords: dict) -> dict:
        cropped = dict(coords)
        cropped["face"] = np.arange(len(self.local_tiles))
        cropped["x"] = np.arange(self.tile_size)
        cropped["y"] = np.arange(self.tile_size)
        return cropped

    def crop(self, x: torch.Tensor) -> torch.Tensor:
        halo = self.halo_width
        if halo == 0:
            return x
        return x[..., halo:-halo, halo:-halo]

    def faces_to_local_tiles(self, x: torch.Tensor) -> torch.Tensor:
        """Slice this rank's interior tiles out of a face-major tensor.

        This is the local (no-collective) counterpart of
        :meth:`gather_tiles_to_faces`. Every rank independently indexes its own
        ``local_tiles`` out of the global face layout.

        Args:
            x: Tensor of shape ``(..., n_faces, face_size, face_size)``.

        Returns:
            Tensor of shape ``(..., tiles_per_rank, tile_size, tile_size)``
            where the new tiles axis enumerates ``self.local_tiles`` in order.
        """
        ts = self.tile_size
        tiles = [
            x[..., face, ti * ts : (ti + 1) * ts, tj * ts : (tj + 1) * ts]
            for face, ti, tj in self.local_tiles
        ]
        return torch.stack(tiles, dim=-3)

    def gather_tiles_to_faces(self, x: torch.Tensor) -> torch.Tensor:
        """All-gather local interior tiles into a global face layout.

        Args:
            x: Local interior tiles with shape ``(..., tiles_per_rank,
                tile_size, tile_size)``. Tiles on each rank are in the slice
                ``global_tiles[rank * tiles_per_rank : (rank + 1) *
                tiles_per_rank]``.

        Returns:
            Global faces with shape ``(..., n_faces, face_size, face_size)``,
            where each tile has been scattered into the position assigned by
            ``global_tiles``.
        """
        *lead, tiles_per_rank, nx, ny = x.shape
        if tiles_per_rank != self.tiles_per_rank:
            raise ValueError(
                f"Expected {self.tiles_per_rank} local tiles, got {tiles_per_rank}"
            )
        if nx != self.tile_size or ny != self.tile_size:
            raise ValueError(
                f"Expected interior tile shape ({self.tile_size}, "
                f"{self.tile_size}), got ({nx}, {ny})"
            )
        tensor_list = [torch.zeros_like(x) for _ in range(self.world_size)]
        torch.distributed.all_gather(tensor_list, x.contiguous())
        # Concatenate along the tiles axis so tile-index order matches
        # global_tiles (rank 0's tiles first, then rank 1's, ...).
        gathered = torch.cat(tensor_list, dim=-3)
        out = gathered.new_zeros(*lead, self.n_faces, self.face_size, self.face_size)
        ts = self.tile_size
        for global_index, (face, ti, tj) in enumerate(self.global_tiles):
            out[..., face, ti * ts : (ti + 1) * ts, tj * ts : (tj + 1) * ts] = gathered[
                ..., global_index, :, :
            ]
        return out


RingSource: TypeAlias = Callable[[torch.Tensor], torch.Tensor]


class DistributedTileKNNHaloPadding_AllGather(torch.nn.Module):
    """Materialize halo-padded local tiles from distributed interior tiles.

    In regional mode (``topology.is_regional``) the source points are the
    interiors of the active tiles followed by those of ``topology.ring_tiles``.
    The ring values come from ``ring_source``, called with the all-gathered
    active tiles ``(..., n_active, ts, ts)`` and returning
    ``(..., n_ring, ts, ts)``; it must be set before the first call.
    """

    def __init__(
        self,
        *,
        topology: TileTopology,
        regrid: torch.nn.Module,
        lat_deg: torch.Tensor,
        lon_deg: torch.Tensor,
    ) -> None:
        super().__init__()
        self.topology = topology
        self.regrid = regrid
        self.ring_source: RingSource | None = None
        self.register_buffer("lat_deg", lat_deg, persistent=False)
        self.register_buffer("lon_deg", lon_deg, persistent=False)

    @property
    def n_src(self) -> int:
        """Number of source points: active (or all) tiles, then ring tiles."""
        t = self.topology
        return (t.total_tiles + len(t.ring_tiles)) * t.tile_size**2

    @classmethod
    def from_padded_face_grid(
        cls,
        *,
        topology: TileTopology,
        lon: torch.Tensor,
        lat: torch.Tensor,
        pad_width_data: int,
        device: torch.device,
        k: int = 4,
        eps: float = 1e-7,
    ) -> "DistributedTileKNNHaloPadding_AllGather":
        """Build the halo padding module from padded face lon/lat grids."""
        local_lon = torch.stack(
            [
                _select_padded_tile(
                    lon,
                    tile,
                    tile_size=topology.tile_size,
                    halo_width=topology.halo_width,
                    pad_width_data=pad_width_data,
                )
                for tile in topology.local_tiles
            ]
        )
        local_lat = torch.stack(
            [
                _select_padded_tile(
                    lat,
                    tile,
                    tile_size=topology.tile_size,
                    halo_width=topology.halo_width,
                    pad_width_data=pad_width_data,
                )
                for tile in topology.local_tiles
            ]
        )

        # A regional operator is the full-cube operator with its source indices
        # remapped onto active + ring tiles, so its halo is the global halo bit
        # for bit (a KD-tree over a smaller source set could break distance
        # ties differently).
        src_tiles = (
            topology.full_cube_tiles if topology.is_regional else topology.global_tiles
        )
        src_lon = torch.cat(
            [
                _select_tile_interior(
                    lon,
                    tile,
                    tile_size=topology.tile_size,
                    pad_width_data=pad_width_data,
                ).flatten()
                for tile in src_tiles
            ]
        )
        src_lat = torch.cat(
            [
                _select_tile_interior(
                    lat,
                    tile,
                    tile_size=topology.tile_size,
                    pad_width_data=pad_width_data,
                ).flatten()
                for tile in src_tiles
            ]
        )

        regrid = KNNS2Interpolator(
            src_lon,
            src_lat,
            local_lon.flatten(),
            local_lat.flatten(),
            k=k,
            eps=eps,
        ).float()
        if topology.is_regional:
            regrid.index.copy_(_remap_to_regional_sources(regrid.index, topology))
        regrid = regrid.to(device)
        return cls(
            topology=topology,
            regrid=regrid,
            lat_deg=local_lat,
            lon_deg=local_lon,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *shape, tiles_per_rank, nx, ny = x.shape
        if tiles_per_rank != self.topology.tiles_per_rank:
            raise ValueError(
                f"Expected {self.topology.tiles_per_rank} local tiles, got {tiles_per_rank}"
            )
        if nx != self.topology.tile_size or ny != self.topology.tile_size:
            raise ValueError(
                f"Expected interior tile shape ({self.topology.tile_size}, "
                f"{self.topology.tile_size}), got ({nx}, {ny})"
            )

        tensor_list = [torch.zeros_like(x) for _ in range(self.topology.world_size)]
        torch.distributed.all_gather(tensor_list, x.contiguous())
        gathered = torch.cat(tensor_list, dim=-3)
        if self.topology.is_regional:
            if self.ring_source is None:
                raise ValueError("regional halo exchange needs a ring_source")
            ring = self.ring_source(gathered)
            n_ring = len(self.topology.ring_tiles)
            if ring.shape != (*shape, n_ring, nx, ny):
                raise ValueError(
                    f"ring_source returned {tuple(ring.shape)}, expected "
                    f"{(*shape, n_ring, nx, ny)}"
                )
            gathered = torch.cat([gathered, ring.to(gathered.dtype)], dim=-3)
        gathered = gathered.reshape(*shape, -1)
        y = self.regrid(gathered)
        return y.reshape(
            *shape,
            tiles_per_rank,
            self.topology.padded_tile_size,
            self.topology.padded_tile_size,
        )


def own_face_target_mask(
    topology: TileTopology,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Mark the padded targets that are real grid points of their own face.

    1 = the target is a real grid point of the tile's own face; 0 = it lies on
    the gnomonic extension beyond a cube-face edge. Extension targets do not
    coincide with the neighbouring face's grid, so the KNN operator has to
    interpolate them instead of copying a single point.

    Multiplying the adjoint's window by this keeps the windowed overlap-add
    wherever the halo is an exact copy (every intra-face tile seam) and drops
    it wherever the halo is interpolated (the cube-face edges and corners).

    The row/column arithmetic mirrors ``_select_padded_tile``, which defines
    the halo target grid. The padded-face offset cancels out of both bounds,
    so the predicate depends only on the tile's position within its own face.

    Returns:
        ``(tiles_per_rank, padded_tile_size, padded_tile_size)`` of 1s and 0s.
    """
    ts, halo, face = topology.tile_size, topology.halo_width, topology.face_size
    masks = []
    for _, tile_i, tile_j in topology.local_tiles:
        rows = torch.arange(tile_i * ts - halo, (tile_i + 1) * ts + halo)
        cols = torch.arange(tile_j * ts - halo, (tile_j + 1) * ts + halo)
        on_row = (rows >= 0) & (rows < face)
        on_col = (cols >= 0) & (cols < face)
        masks.append(on_row[:, None] & on_col[None, :])
    return torch.stack(masks).to(device=device, dtype=dtype)


class NormalizedHaloAdjoint(torch.nn.Module):
    """Normalized adjoint of DistributedTileKNNHaloPadding_AllGather.

    Computes D_w⁻¹ Aᵀ (w ⊙ y), where A is the halo exchange operator,
    w is an optional spatial window applied to each padded tile, and
    D_w = diag(Aᵀ w) is the per-interior-point sum of window weights.

    With a uniform window (default) this reduces to the plain Shepard
    normalization D⁻¹ Aᵀ y.  A KBD or similar window downweights halo
    contributions relative to interior-tile contributions, matching the
    windowed overlap-add used in patch-based inference.

    ``window`` may be a single ``(padded, padded)`` window shared by every
    tile, or a per-tile ``(tiles_per_rank, padded, padded)`` stack.

    ``own_face_only=True`` additionally multiplies the window by
    :func:`own_face_target_mask`, dropping the overlap-add across cube-face
    edges while keeping it at every intra-face tile seam. ``D_w`` is
    recomputed from whatever window results, so the normalization stays exact
    either way: a constant field still round-trips.

    Implementation note:

    A is equivalent to an all-gather followed by a local regridding operator and
    the transpose of an all-gather is a reduce scatter to return source-point
    contributions to the ranks that own them.  So A^T can be implemented by
    applying the transpose of the regridding followed by a reduce scatter. The
    transpose of regridding is implemented as a vector-Jacobian product through
    the forward operator using torch.func.vjp.

    Regional mode: the VJP also returns contributions to the ring sources, which
    are dropped (the ring is not predicted). ``D_w`` is built the same way, so
    points next to the region edge are normalised by the weights of the active
    tiles alone and a constant field still round-trips.
    """

    weights: torch.Tensor
    window: torch.Tensor

    def __init__(
        self,
        halo_exchange: DistributedTileKNNHaloPadding_AllGather,
        window: torch.Tensor | None = None,
        own_face_only: bool = False,
    ) -> None:
        super().__init__()
        self.halo_exchange = halo_exchange
        topology = halo_exchange.topology
        padded_size = topology.padded_tile_size
        self._local_src_chunk = topology.tiles_per_rank * topology.tile_size**2
        self._global_src_size = self._local_src_chunk * topology.world_size
        self._n_src = halo_exchange.n_src
        device = halo_exchange.lon_deg.device
        dtype = halo_exchange.lon_deg.dtype

        if window is None:
            window = torch.ones(padded_size, padded_size, device=device, dtype=dtype)
        window = window.to(device=device, dtype=dtype)
        if window.shape == (padded_size, padded_size):
            window = window.unsqueeze(0).expand(topology.tiles_per_rank, -1, -1)
        elif window.shape != (topology.tiles_per_rank, padded_size, padded_size):
            raise ValueError(
                f"window must be ({padded_size}, {padded_size}) or "
                f"({topology.tiles_per_rank}, {padded_size}, {padded_size}), "
                f"got {tuple(window.shape)}"
            )
        if own_face_only:
            window = window * own_face_target_mask(topology, device=device, dtype=dtype)
        # Flatten to (n_dst,) in tile-major order. clone() so the buffer does
        # not alias the caller's tensor.
        window_flat = window.reshape(-1).clone()
        self.register_buffer("window", window_flat)
        self._vjp = {}
        weights = self._distributed_regrid_vjp(window_flat.unsqueeze(0))
        n_bad = int((~(weights > 0)).sum())
        if (
            topology.world_size > 1
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            counter = torch.tensor([n_bad], device=weights.device, dtype=torch.int64)
            torch.distributed.all_reduce(counter, op=torch.distributed.ReduceOp.SUM)
            n_bad = int(counter.item())
        if n_bad:
            raise ValueError(
                f"{n_bad} interior points receive zero, negative or NaN total "
                "window weight, so D_w is singular. Check the window/mask."
            )
        self.register_buffer(
            "weights",
            weights.reshape(
                1, 1, topology.tiles_per_rank, topology.tile_size, topology.tile_size
            ),
        )

    def _reduce_scatter_local_shard(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self._global_src_size:
            raise ValueError(
                f"Expected global source size {self._global_src_size}, got {x.shape[-1]}"
            )
        if self.halo_exchange.topology.world_size == 1:
            return x

        *shape, n_src = x.shape
        batch_flat = x.numel() // n_src
        x_2d = x.reshape(batch_flat, n_src).contiguous()
        chunks = list(x_2d.split(self._local_src_chunk, dim=-1))
        if len(chunks) != self.halo_exchange.topology.world_size:
            raise ValueError(
                f"Expected {self.halo_exchange.topology.world_size} equal source shards, got {len(chunks)}"
            )
        out = torch.empty(
            batch_flat,
            self._local_src_chunk,
            device=x.device,
            dtype=x.dtype,
        )
        torch.distributed.reduce_scatter(out, chunks)
        return out.reshape(*shape, self._local_src_chunk)

    def _local_regrid_vjp(self, y: torch.Tensor) -> torch.Tensor:
        key = (y.shape, y.dtype, y.device)
        if key not in self._vjp:
            # vjp - vector jacobian product
            # need to disable torch inference mode or the vjp function returned by
            # torch.func.vjp will silenty return zeros when used
            # after this it is not important
            with torch.inference_mode(False):
                cotangent = y.detach().clone()
                primal = torch.zeros(
                    *cotangent.shape[:-1],
                    self._n_src,
                    device=cotangent.device,
                    dtype=cotangent.dtype,
                )
                _, self._vjp[key] = torch.func.vjp(self.halo_exchange.regrid, primal)

        return self._vjp[key](y)[0]

    def _distributed_regrid_vjp(
        self,
        y: torch.Tensor,
    ) -> torch.Tensor:
        global_adj = self._local_regrid_vjp(y)[..., : self._global_src_size]
        return self._reduce_scatter_local_shard(global_adj)

    def forward(self, padded: torch.Tensor) -> torch.Tensor:
        topology = self.halo_exchange.topology
        *shape, _, _, _ = padded.shape
        padded_flat = padded.reshape(*shape, -1) * self.window
        accumulated = self._distributed_regrid_vjp(padded_flat)
        return (
            accumulated.reshape(
                *shape, topology.tiles_per_rank, topology.tile_size, topology.tile_size
            )
            / self.weights
        )


def _select_padded_tile(
    x: torch.Tensor,
    tile: Tile,
    *,
    tile_size: int,
    halo_width: int,
    pad_width_data: int,
) -> torch.Tensor:
    face, tile_i, tile_j = tile
    return x[
        ...,
        face,
        pad_width_data
        + tile_i * tile_size
        - halo_width : pad_width_data
        + (tile_i + 1) * tile_size
        + halo_width,
        pad_width_data
        + tile_j * tile_size
        - halo_width : pad_width_data
        + (tile_j + 1) * tile_size
        + halo_width,
    ]


def _select_tile_interior(
    x: torch.Tensor,
    tile: Tile,
    *,
    tile_size: int,
    pad_width_data: int,
) -> torch.Tensor:
    face, tile_i, tile_j = tile
    return x[
        face,
        pad_width_data + tile_i * tile_size : pad_width_data + (tile_i + 1) * tile_size,
        pad_width_data + tile_j * tile_size : pad_width_data + (tile_j + 1) * tile_size,
    ]


def _remap_to_regional_sources(
    index: torch.Tensor, topology: TileTopology
) -> torch.Tensor:
    """Map full-cube KNN source indices onto the regional source vector
    (active tiles in ``global_tiles`` order, then ``ring_tiles``).

    Raises if a halo target of an active tile reads a source outside active +
    ring, i.e. if the ring is too thin for the halo.
    """
    npix = topology.tile_size**2
    slot = {
        tile: k for k, tile in enumerate(topology.global_tiles + topology.ring_tiles)
    }
    lookup = torch.tensor(
        [slot.get(tile, -1) for tile in topology.full_cube_tiles], dtype=torch.long
    )
    tile_slot = lookup[index // npix]
    if (tile_slot < 0).any():
        raise ValueError(
            f"{int((tile_slot < 0).sum())} halo sources lie outside the active "
            "tiles and their ring"
        )
    return tile_slot * npix + index % npix
