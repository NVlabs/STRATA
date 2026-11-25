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
"""Regional window rollout: invariants of the active-tile halo, the adjoint, the ring
sources, the region finder and the writer.

The toy grid is a real gnomonic equiangular cube with its halo extension, so the KNN
halo copies intra-face points exactly as on ne1024pg2, in ne1024pg2's face layout
(equatorial faces: rows north, columns east; face 5 the north pole), so the region
planes join faces where the grid does. Most tests run on a one-face region and on the
two-face planes (east and north). No test restates the code's arithmetic; the
expectations are global-run equality, constants, and closed forms.
"""

import itertools
import os
from collections import OrderedDict

import numpy as np
import pytest
import torch

from screamcast.distributed_halo import (
    FACE_AXES,
    DistributedTileKNNHaloPadding_AllGather,
    NormalizedHaloAdjoint,
    TileTopology,
    faces_to_plane,
    from_plane,
    plane_blocks,
    to_plane,
)
from screamcast.regional import (
    AnchoredBoundary,
    ReplicateRing,
    box_mask,
    edge_distance,
    faces_to_box,
    faces_to_pieces,
    find_box_region,
    find_region,
    nearest_pixel,
    parse_region,
    region_plane_of,
    smooth,
)
from screamcast.zarr_writer import FaceZarrWriteStep

FACE, TILE, HALO, PAD = 192, 32, 4, 8
REGION = "3,1,5,1,5"  # face 3, tiles 1:5 x 1:5 (128 px), ring tiles 0 and 5
# 4 x 4 tiles across a face edge (6 tiles per face side): faces 3 and 0 (cols 4-5 of
# face 3, 0-1 of face 0), and faces 3 and 5 (rows 4-5 of face 3, two rows of face 5)
# 4 x 4 tiles inside face 3, flush with its north, east and west edges: their ring
# reaches into faces 5, 0 and 2
FLUSH = {
    "flush_north": "3,2,6,1,5",
    "flush_east": "3,1,5,2,6",
    "flush_west": "3,1,5,0,4",
}
# 4 x 4 tiles across face 3's south edge (two rows on the south polar face, written with
# a negative i0)
PLANE = {"face": REGION, "east": "3,1,5,4,8", "north": "3,4,8,1,5", **FLUSH}
PLANE["south"] = "3,-2,2,1,5"
# windows whose ring leaves their plane: flush with a south edge, at a cube corner, as
# wide as a face, flush with a polar face's edge, and across the south join flush with
# the east edge (the ring reaches faces 0, 4 and, around the corner, the others)
OFF_PLANE = {
    "off_south": "3,0,4,1,5",
    "off_corner": "3,2,6,2,6",
    "off_wide": "3,1,5,0,6",
    "off_polar": "5,2,6,1,5",
    "off_south_polar": "4,1,5,2,6",
    "off_south_join": "3,-2,2,2,6",
}
REGIONS = {**PLANE, **OFF_PLANE}


@pytest.fixture(params=sorted(REGIONS))
def region(request):
    return REGIONS[request.param]


@pytest.fixture(params=sorted(PLANE))
def plane_region(request):
    """A window whose ring lies on its plane."""
    return PLANE[request.param]


def _cube_latlon(n: int, pad: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Gnomonic equiangular cube, cell centres, each face extended by ``pad`` cells."""
    a = (np.arange(-pad, n + pad) + 0.5) * (np.pi / 2 / n) - np.pi / 4
    X, Y = np.meshgrid(np.tan(a), np.tan(a), indexing="ij")
    axes = [  # (normal, row direction, column direction) per face, as ne1024pg2
        ((1, 0, 0), (0, 0, 1), (0, 1, 0)),
        ((0, 1, 0), (0, 0, 1), (-1, 0, 0)),
        ((-1, 0, 0), (0, 0, 1), (0, -1, 0)),
        ((0, -1, 0), (0, 0, 1), (1, 0, 0)),
        ((0, 0, -1), (1, 0, 0), (0, 1, 0)),
        ((0, 0, 1), (-1, 0, 0), (0, 1, 0)),
    ]
    lat, lon = [], []
    for nrm, u, v in axes:
        p = np.asarray(nrm)[:, None, None] + X * np.asarray(u)[:, None, None]
        p = p + Y * np.asarray(v)[:, None, None]
        p /= np.linalg.norm(p, axis=0)
        lat.append(np.degrees(np.arcsin(p[2])))
        lon.append(np.degrees(np.arctan2(p[1], p[0])))
    return (
        torch.tensor(np.stack(lat), dtype=torch.float32),
        torch.tensor(np.stack(lon), dtype=torch.float32),
    )


def _cube_area(n: int) -> np.ndarray:
    """Cell areas (sr) of the equiangular cube: the gnomonic Jacobian, total 4 pi."""
    a = (np.arange(n) + 0.5) * (np.pi / 2 / n) - np.pi / 4
    X, Y = np.meshgrid(np.tan(a), np.tan(a), indexing="ij")
    jac = (1 + X**2) * (1 + Y**2) / (1 + X**2 + Y**2) ** 1.5
    area = np.broadcast_to(jac, (6, n, n))
    return area * (4 * np.pi / area.sum())


@pytest.fixture(scope="module")
def grid():
    return _cube_latlon(FACE, PAD)


@pytest.fixture(autouse=True)
def _single_process(monkeypatch):
    def all_gather(tensor_list, x, group=None):
        tensor_list[0].copy_(x)

    monkeypatch.setattr(torch.distributed, "all_gather", all_gather)
    monkeypatch.setattr(torch.distributed, "barrier", lambda *a, **k: None)
    monkeypatch.setattr(torch.distributed, "broadcast", lambda *a, **k: None)


def _topology(region: str | None) -> TileTopology:
    return TileTopology(
        world_size=1,
        rank=0,
        face_size=FACE,
        tile_size=TILE,
        halo_width=HALO,
        active_tiles=parse_region(region, FACE // TILE) if region else None,
    )


def _halo(topology, grid):
    lat, lon = grid
    return DistributedTileKNNHaloPadding_AllGather.from_padded_face_grid(
        topology=topology,
        lon=lon,
        lat=lat,
        pad_width_data=PAD,
        device=torch.device("cpu"),
    )


def _kbd_window(topology) -> torch.Tensor:
    w = torch.signal.windows.kaiser(topology.padded_tile_size, beta=10.0, sym=True)
    return (w[:, None] * w[None, :]).float()


def _ring_from(field: torch.Tensor, topology: TileTopology):
    """Ring source serving the ring tiles of a full-cube field ``(C, 6, F, F)``."""

    def source(gathered):
        ring = torch.stack(
            [
                field[:, f, i * TILE : (i + 1) * TILE, j * TILE : (j + 1) * TILE]
                for f, i, j in topology.ring_tiles
            ],
            dim=1,
        )
        return ring.expand(*gathered.shape[:-3], *ring.shape[-3:])

    return source


def _tiles(field: torch.Tensor, tiles) -> torch.Tensor:
    """``(C, 6, F, F)`` -> ``(1, C, len(tiles), ts, ts)``."""
    return torch.stack(
        [
            field[:, f, i * TILE : (i + 1) * TILE, j * TILE : (j + 1) * TILE]
            for f, i, j in tiles
        ],
        dim=1,
    ).unsqueeze(0)


def _referenced_ring_pixels(halo_exchange) -> torch.Tensor:
    """Ring pixels with nonzero KNN weight in the halo, ``(n_ring, ts, ts)``."""
    topo = halo_exchange.topology
    npix = topo.tile_size**2
    n_active = topo.total_tiles * npix
    index = halo_exchange.regrid.index.reshape(-1)
    weight = halo_exchange.regrid.weight.reshape(-1)
    ring = index[(index >= n_active) & (weight > 0)] - n_active
    used = torch.zeros(len(topo.ring_tiles) * npix, dtype=torch.bool)
    used[ring] = True
    return used.reshape(len(topo.ring_tiles), topo.tile_size, topo.tile_size)


# ---------------------------------------------------------------- topology


def test_region_topology_and_ring():
    topo = _topology(REGION)
    assert topo.total_tiles == 16 and topo.tiles_per_rank == 16
    assert topo.region_box == (3, 1, 5, 1, 5)
    assert len(topo.ring_tiles) == 6 * 6 - 16
    assert not set(topo.ring_tiles) & set(topo.global_tiles)
    assert topo.ring_on_plane and topo.boundary_tiles == topo.ring_tiles
    with pytest.raises(ValueError, match="rectangle"):
        TileTopology(
            world_size=1,
            rank=0,
            face_size=FACE,
            tile_size=TILE,
            halo_width=HALO,
            active_tiles=((3, 1, 1), (3, 2, 2)),
        )


def test_plane_regions_topology_and_ring():
    east, north = _topology(REGIONS["east"]), _topology(REGIONS["north"])
    assert {f for f, _, _ in east.global_tiles} == {3, 0}
    assert {f for f, _, _ in north.global_tiles} == {3, 5}
    assert east.region_box == (3, 1, 5, 4, 8) and north.region_box == (3, 4, 8, 1, 5)
    for topo in (east, north):
        assert topo.total_tiles == 16 and len(topo.ring_tiles) == 6 * 6 - 16
        assert len(set(topo.ring_tiles)) == len(topo.ring_tiles)
        assert not set(topo.ring_tiles) & set(topo.global_tiles)
    # the ring crosses the edge with the region: face 0 col 2 east, face 5 north
    assert {(0, i, 2) for i in range(5)} <= set(east.ring_tiles)
    assert {f for f, _, _ in north.ring_tiles} == {3, 5}
    assert not _topology("3,1,5,5,12").ring_on_plane  # flush with face 0's east edge
    # inside face 0 and flush with its east edge: runs on the plane of faces 0 and 1
    assert _topology("3,1,5,8,12").region_plane == plane_blocks(0, east=True)
    with pytest.raises(ValueError, match="span faces"):
        TileTopology(
            world_size=1,
            rank=0,
            face_size=FACE,
            tile_size=TILE,
            halo_width=HALO,
            active_tiles=((0, 2, 2), (2, 2, 2)),
        )


def test_tiles_box_round_trip(region):
    topo = _topology(region)
    x = torch.randn(2, topo.total_tiles, TILE, TILE)
    _, r0, r1, c0, c1 = topo.region_pixel_box()
    box = topo.tiles_to_box(x, topo.global_tiles, (r0, c0), (r1 - r0, c1 - c0))
    assert torch.equal(topo.box_to_tiles(box, topo.global_tiles, (r0, c0)), x)


def test_the_region_box_is_the_plane_of_the_joined_faces(region):
    """Tiles placed one by one equal the plane built from whole faces."""
    topo = _topology(region)
    field = torch.randn(2, 6, FACE, FACE)
    _, r0, r1, c0, c1 = topo.region_pixel_box()
    box = topo.tiles_to_box(
        _tiles(field, topo.global_tiles)[0],
        topo.global_tiles,
        (r0, c0),
        (r1 - r0, c1 - c0),
    )
    assert torch.equal(box, faces_to_plane(field, topo.region_plane)[:, r0:r1, c0:c1])


@pytest.mark.parametrize(
    "name, plane, neighbour",
    [
        ("flush_north", plane_blocks(3, north=True), 5),
        ("flush_east", plane_blocks(3, east=True), 0),
        ("flush_west", plane_blocks(2, east=True), 2),
    ],
)
def test_a_window_flush_with_a_face_edge_takes_its_ring_from_the_neighbour(
    name, plane, neighbour
):
    spec = REGIONS[name]
    topo = _topology(spec)
    assert {f for f, _, _ in topo.global_tiles} == {3}
    assert topo.region_plane == plane
    assert region_plane_of(spec, FACE // TILE) == plane
    # the ring row or column beyond the flush edge, corners included, is the neighbour's
    assert len(topo.ring_tiles) == 6 * 6 - 16
    assert sum(f == neighbour for f, _, _ in topo.ring_tiles) == 6
    assert {f for f, _, _ in topo.ring_tiles} == {3, neighbour}


def test_a_flush_window_is_the_same_written_on_the_neighbours_plane():
    """``3,1,5,0,4`` written on the plane of face 2 joined east to face 3."""
    tiles = parse_region("2,1,5,6,10", FACE // TILE)
    assert tiles == parse_region(REGIONS["flush_west"], FACE // TILE)
    assert region_plane_of("2,1,5,6,10", FACE // TILE) == plane_blocks(2, east=True)


def _touching_by_distance(topo) -> set:
    """Tiles of any face whose centre lies within 1.5 tile widths (on the face grid's
    great circles) of a window tile centre: the tiles sharing an edge or a corner. From
    the toy grid's lat/lon, not from the cube lattice ``ring_tiles`` uses."""
    lat, lon = (torch.deg2rad(g).numpy() for g in _cube_latlon(FACE // TILE, 0))
    xyz = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    step = np.arccos(np.clip((xyz[:, 3, 2, 2] * xyz[:, 3, 2, 3]).sum(), -1, 1))
    tiles = list(itertools.product(range(6), range(FACE // TILE), range(FACE // TILE)))
    out = set()
    for t in tiles:
        if t in topo.global_tiles:
            continue
        a = xyz[:, t[0], t[1], t[2]]
        for w in topo.global_tiles:
            b = xyz[:, w[0], w[1], w[2]]
            if np.arccos(np.clip((a * b).sum(), -1, 1)) < 1.6 * step:
                out.add(t)
                break
    return out


@pytest.mark.parametrize("name", sorted(REGIONS))
def test_the_ring_is_every_tile_touching_the_window(name):
    topo = _topology(REGIONS[name])
    assert set(topo.ring_tiles) == _touching_by_distance(topo)
    assert len(set(topo.ring_tiles)) == len(topo.ring_tiles)
    assert topo.ring_on_plane == (name in PLANE)


def test_a_window_at_a_cube_corner_has_seven_neighbours_at_the_corner():
    topo = _topology("3,0,1,5,6")  # the south-east corner tile of face 3
    assert len(topo.ring_tiles) == 7
    assert {f for f, _, _ in topo.ring_tiles} == {3, 0, 4}


@pytest.mark.parametrize("name", sorted(OFF_PLANE))
def test_ring_pieces_cover_the_ring_once_on_its_faces(name):
    topo = _topology(REGIONS[name])
    covered = [t for piece in topo.ring_pieces for t in piece.tiles]
    assert sorted(covered) == sorted(topo.ring_tiles)
    ts = topo.tile_size
    for piece in topo.ring_pieces:
        ((face, *_),) = piece.blocks
        rows = range(piece.origin[0] // ts, (piece.origin[0] + piece.size[0]) // ts)
        cols = range(piece.origin[1] // ts, (piece.origin[1] + piece.size[1]) // ts)
        inside = {(face, i, j) for i in rows for j in cols}
        assert set(piece.tiles) <= inside
        # what else the piece holds is the window or the extra boundary tiles
        assert inside <= {*topo.global_tiles, *topo.boundary_tiles}


def test_replicate_ring_refuses_a_ring_off_the_plane():
    with pytest.raises(ValueError, match="anchored"):
        ReplicateRing(_topology(OFF_PLANE["off_south"]))


def test_a_south_window_is_written_with_a_negative_row():
    tiles = parse_region(PLANE["south"], FACE // TILE)
    assert {f for f, _, _ in tiles} == {3, 4}
    assert [t for t in tiles if t[0] == 3] == [
        (3, i, j) for i in range(2) for j in range(1, 5)
    ]
    topo = _topology(PLANE["south"])
    assert topo.region_plane == plane_blocks(3, south=True)
    assert topo.region_spec == PLANE["south"]
    assert topo.ring_on_plane


@pytest.mark.parametrize("side", ["east", "north", "south"])
def test_the_plane_joins_faces_where_the_grid_does(grid, side):
    """Across the seam, neighbouring plane pixels are about one grid cell apart, as
    within a face (a wrong join or turn puts them a face apart): the plane's join and
    turn match the cube. Near the cube corners the grid kinks, so "about" is 25 %."""
    lat, lon = (torch.deg2rad(g[:, PAD : PAD + FACE, PAD : PAD + FACE]) for g in grid)
    blocks = _topology(REGIONS[side]).region_plane
    la, lo = faces_to_plane(lat, blocks), faces_to_plane(lon, blocks)
    xyz = torch.stack([la.cos() * lo.cos(), la.cos() * lo.sin(), la.sin()])
    if side in ("north", "south"):
        xyz = xyz.transpose(1, 2)  # the seam between plane rows FACE - 1 and FACE
    step = (xyz[:, :, 1:] - xyz[:, :, :-1]).norm(dim=0)  # (FACE, cols - 1)
    across, inside = step[:, FACE - 1], step[:, FACE - 2]
    torch.testing.assert_close(across, inside, rtol=0.25, atol=0)


def test_the_real_grid_joins_faces_as_the_plane_does():
    """ne1024pg2 itself (skipped without data/latlon_ne1024pg2.nc)."""
    path = os.environ.get("SCREAMCAST_LATLON", "data/latlon_ne1024pg2.nc")
    if not os.path.exists(path):
        pytest.skip(f"no {path}")
    h5py = pytest.importorskip("h5py")
    from screamcast.cubesphere_transforms import reorder_cubesphere_to_2d_tensor
    from screamcast.distributed_halo import plane_blocks

    with h5py.File(path, "r") as f:
        flat = {k: torch.from_numpy(f[k][:].astype(np.float64)) for k in ("lat", "lon")}
    per = flat["lat"].numel() // 6
    faces = {
        k: torch.deg2rad(
            torch.stack(
                [
                    reorder_cubesphere_to_2d_tensor(
                        v[i * per : (i + 1) * per], ne=1024, npg=2
                    )
                    for i in range(6)
                ]
            )
        )
        for k, v in flat.items()
    }
    n = 2048
    for face in range(4):
        for side in ("east", "north", "south"):
            blocks = plane_blocks(face, **{side: True})
            la = faces_to_plane(faces["lat"], blocks)
            lo = faces_to_plane(faces["lon"], blocks)
            xyz = torch.stack([la.cos() * lo.cos(), la.cos() * lo.sin(), la.sin()])
            if side in ("north", "south"):
                xyz = xyz.transpose(1, 2)
            across = (xyz[:, :, n] - xyz[:, :, n - 1]).norm(dim=0)
            inside = (xyz[:, :, n - 1] - xyz[:, :, n - 2]).norm(dim=0)
            torch.testing.assert_close(across, inside, rtol=0.25, atol=0)
    # FACE_AXES (which ring_tiles builds the cube's corners from): each face's centre
    # lies along its normal, and its rows and columns run along its row and column axes
    xyz = torch.stack(
        [
            faces["lat"].cos() * faces["lon"].cos(),
            faces["lat"].cos() * faces["lon"].sin(),
            faces["lat"].sin(),
        ]
    )
    c = n // 2
    for face, (normal, row, col) in enumerate(FACE_AXES):
        centre = xyz[:, face, c, c]
        d_row = xyz[:, face, c + 1, c] - xyz[:, face, c - 1, c]
        d_col = xyz[:, face, c, c + 1] - xyz[:, face, c, c - 1]
        for got, want in ((centre, normal), (d_row, row), (d_col, col)):
            cosine = float(got @ torch.tensor(want, dtype=got.dtype) / got.norm())
            assert cosine > 0.999, (face, want, cosine)


# ---------------------------------------------------------------- halo + adjoint


def test_regional_halo_equals_global_halo(grid, region):
    """With the true field in the ring, every padded active tile is the global one, bit for bit."""
    g_topo, r_topo = _topology(None), _topology(region)
    g_halo, r_halo = _halo(g_topo, grid), _halo(r_topo, grid)
    field = torch.randn(3, 6, FACE, FACE)
    g_pad = g_halo(_tiles(field, g_topo.global_tiles))
    r_halo.ring_source = _ring_from(field, r_topo)
    r_pad = r_halo(_tiles(field, r_topo.global_tiles))
    index = [g_topo.global_tiles.index(t) for t in r_topo.global_tiles]
    assert torch.equal(r_pad, g_pad[:, :, index])


def test_halo_reads_only_a_thin_band_of_the_ring(grid, plane_region):
    r_topo = _topology(plane_region)
    used = _referenced_ring_pixels(_halo(r_topo, grid))
    dist = edge_distance(r_topo, r_topo.ring_tiles)
    assert used.any()
    assert int(-dist[used].min()) <= HALO + 2  # the halo plus its KNN neighbours


def test_regional_adjoint_round_trips_a_constant(grid, region):
    topo = _topology(region)
    unpad = NormalizedHaloAdjoint(_halo(topo, grid), window=_kbd_window(topo))
    p = topo.padded_tile_size
    out = unpad(torch.full((1, 2, topo.tiles_per_rank, p, p), 7.25))
    torch.testing.assert_close(out, torch.full_like(out, 7.25), rtol=1e-6, atol=0)


def test_regional_adjoint_matches_global_away_from_the_edge(grid, region):
    """Points >= halo from the region edge see only active-tile predictions."""
    g_topo, r_topo = _topology(None), _topology(region)
    w = _kbd_window(g_topo)
    g_unpad = NormalizedHaloAdjoint(_halo(g_topo, grid), window=w, own_face_only=True)
    r_unpad = NormalizedHaloAdjoint(_halo(r_topo, grid), window=w, own_face_only=True)
    p = g_topo.padded_tile_size
    y = torch.randn(1, 2, g_topo.total_tiles, p, p)
    index = [g_topo.global_tiles.index(t) for t in r_topo.global_tiles]
    g = g_unpad(y)[:, :, index]
    r = r_unpad(y[:, :, index])
    dist = edge_distance(r_topo, r_topo.global_tiles)
    deep, strip = dist >= HALO, dist < HALO
    torch.testing.assert_close(r[..., deep], g[..., deep], rtol=1e-6, atol=1e-6)
    assert not torch.allclose(r[..., strip], g[..., strip])  # the ring mattered there


# ---------------------------------------------------------------- ring sources


def test_replicate_ring_extends_the_edge_pixels(plane_region):
    topo = _topology(plane_region)
    _, r0, r1, c0, c1 = topo.region_pixel_box()
    rows = torch.arange(r0, r1, dtype=torch.float32)
    cols = torch.arange(c0, c1, dtype=torch.float32)
    box = (1000 * rows[:, None] + cols[None, :]).expand(2, -1, -1)  # f = 1000 r + c
    gathered = topo.box_to_tiles(box, topo.global_tiles, (r0, c0))
    ring = ReplicateRing(topo)(gathered)
    for k, tile in enumerate(topo.ring_tiles):
        ti, tj, turns = topo.tile_plane_position(tile)
        r = torch.arange(ti * TILE, (ti + 1) * TILE).clamp(r0, r1 - 1).float()
        c = torch.arange(tj * TILE, (tj + 1) * TILE).clamp(c0, c1 - 1).float()
        expect = torch.rot90(1000 * r[:, None] + c[None, :], -turns)  # tile's own axes
        assert torch.equal(ring[:, k], expect.expand(2, -1, -1))


class _StencilModel:
    """Deterministic local 'network': a fixed nonlinear 5-point stencil on the padded
    tile (edges replicated), so each tile's output depends on its whole halo."""

    def __call__(self, x):
        xp = torch.nn.functional.pad(
            x.reshape(-1, 1, *x.shape[-2:]), (1, 1, 1, 1), mode="replicate"
        ).reshape(*x.shape[:-2], x.shape[-2] + 2, x.shape[-1] + 2)
        c = xp[..., 1:-1, 1:-1]
        lap = (
            xp[..., :-2, 1:-1]
            + xp[..., 2:, 1:-1]
            + xp[..., 1:-1, :-2]
            + xp[..., 1:-1, 2:]
        )
        return c + 0.2 * (lap - 4 * c) + 0.05 * torch.tanh(xp[..., 2:, 2:]) - 0.01 * c


def _roll(state, pad, unpad, n, after=None):
    model = _StencilModel()
    for e in range(n):
        state = unpad(model(pad(state)))
        if after is not None:
            state = after(e, state)
    return state


# ---------------------------------------------------------------- writer


def test_regional_writer_writes_only_the_region(region):
    topo = _topology(region)
    store = {
        "U": np.full((1, 1, 2, 6, FACE, FACE), -1.0, dtype=np.float32),
        "ps": np.full((1, 1, 6, FACE, FACE), -1.0, dtype=np.float32),
    }
    grouped = OrderedDict([("U", [(18, 0), (19, 1)]), ("ps", [(None, 2)])])
    writer = FaceZarrWriteStep(
        output_steps=1, topology=topo, grouped_variables=grouped, store=store
    )
    field = torch.randn(3, 6, FACE, FACE)
    writer(0, _tiles(field, topo.global_tiles))
    writer.close()
    for f, i, j in topo.global_tiles:
        rows, cols = slice(i * TILE, (i + 1) * TILE), slice(j * TILE, (j + 1) * TILE)
        np.testing.assert_array_equal(
            store["U"][0, 0, :, f, rows, cols], field[:2, f, rows, cols]
        )
        np.testing.assert_array_equal(
            store["ps"][0, 0, f, rows, cols], field[2, f, rows, cols]
        )
        store["U"][0, 0, :, f, rows, cols] = -1.0
        store["ps"][0, 0, f, rows, cols] = -1.0
    assert (store["U"] == -1.0).all() and (store["ps"] == -1.0).all()


# ---------------------------------------------------------------- anchored ring


def _field(channels=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(channels, 6, FACE, FACE, generator=g)


def _ring_slices(field, topo):
    """The ring tiles of a face-major field ``(C, 6, F, F)``, cut from the faces:
    ``(C, n_ring, ts, ts)``."""
    return _tiles(field, topo.ring_tiles)[0]


def _anchored(region, ic, drivers, block):
    """The anchored ring with the IC and the drivers given as face-major fields."""
    topo = _topology(region)
    boundary = AnchoredBoundary(
        topology=topo,
        ic_boxes=faces_to_pieces(ic, topo),
        driver0=faces_to_pieces(drivers[0.0], topo),
        block=block,
        device=torch.device("cpu"),
    )
    for h, field in drivers.items():
        if h:
            boundary.add(h, faces_to_pieces(field, topo))
    return topo, boundary


def test_anchored_ring_starts_at_the_initial_state(region):
    ic = _field(seed=1)
    drivers = {0.0: _field(seed=2), 6.0: _field(seed=3)}
    topo, boundary = _anchored(region, ic, drivers, block=8)
    assert torch.equal(boundary.ring_at(0.0), _ring_slices(ic, topo))


def test_anchored_ring_adds_the_driver_change_unsmoothed(region):
    ic = _field(seed=1)
    d0, d6 = _field(seed=2), _field(seed=3)
    topo, boundary = _anchored(region, ic, {0.0: d0, 6.0: d6}, block=1)
    expect = _ring_slices(ic + (d6 - d0), topo)
    torch.testing.assert_close(boundary.ring_at(6.0), expect, rtol=0, atol=1e-5)


def test_anchored_ring_blends_the_bracketing_drivers(region):
    """With drivers at 0, 6 and 12 h (added out of order), the ring at t is the IC plus
    the change to the linear blend of the two drivers around t."""
    ic = _field(seed=1)
    d = {h: _field(seed=2 + k) for k, h in enumerate((0.0, 12.0, 6.0))}
    topo, boundary = _anchored(region, ic, d, block=1)
    for t, (a, b, w) in {
        3.0: (0.0, 6.0, 0.5),
        9.0: (6.0, 12.0, 0.5),
        10.5: (6.0, 12.0, 0.75),
        12.0: (6.0, 12.0, 1.0),
    }.items():
        blend = (1 - w) * d[a] + w * d[b]
        expect = _ring_slices(ic + (blend - d[0.0]), topo)
        torch.testing.assert_close(boundary.ring_at(t), expect, rtol=0, atol=1e-5)


def test_anchored_ring_is_linear_in_time():
    ic = _field(seed=1)
    drivers = {0.0: _field(seed=2), 6.0: _field(seed=3)}
    _, boundary = _anchored(REGION, ic, drivers, block=8)
    r0, r6 = boundary.ring_at(0.0), boundary.ring_at(6.0)
    torch.testing.assert_close(
        boundary.ring_at(3.0), 0.5 * (r0 + r6), rtol=0, atol=1e-6
    )
    # a quarter of the way: mostly the earlier driver
    torch.testing.assert_close(
        boundary.ring_at(1.5), 0.75 * r0 + 0.25 * r6, rtol=0, atol=1e-6
    )
    with pytest.raises(ValueError, match="no boundary driver"):
        boundary.ring_at(6.5)


def test_a_driver_that_does_not_change_keeps_the_initial_state(region):
    ic = _field(seed=1)
    d = _field(seed=2)
    topo, boundary = _anchored(region, ic, {0.0: d, 6.0: d.clone()}, block=8)
    torch.testing.assert_close(boundary.ring_at(6.0), boundary.ring_at(0.0))


def test_a_uniform_driver_change_shifts_the_ring_uniformly(region):
    """A spatially uniform change of +2.5 passes the smoothing unchanged."""
    ic = _field(seed=1)
    d0 = _field(seed=2)
    topo, boundary = _anchored(region, ic, {0.0: d0, 6.0: d0 + 2.5}, block=8)
    torch.testing.assert_close(
        boundary.ring_at(6.0) - boundary.ring_at(0.0),
        torch.full_like(boundary.ring_at(0.0), 2.5),
        rtol=0,
        atol=1e-5,
    )


@pytest.mark.parametrize("name", sorted(OFF_PLANE))
def test_a_ring_off_the_plane_is_smoothed_within_each_face(name):
    """A driver change that is a different constant on every face passes the per-face
    smoothing exactly: each ring tile moves by its own face's constant."""
    ic, d0 = _field(seed=1), _field(seed=2)
    step = torch.arange(6, dtype=torch.float32)[None, :, None, None] * 1.5
    topo, boundary = _anchored(REGIONS[name], ic, {0.0: d0, 6.0: d0 + step}, block=8)
    faces = torch.tensor([f for f, _, _ in topo.ring_tiles], dtype=torch.float32)
    torch.testing.assert_close(
        boundary.ring_at(6.0) - boundary.ring_at(0.0),
        (1.5 * faces)[None, :, None, None].expand_as(boundary.ring_at(0.0)),
        rtol=0,
        atol=1e-5,
    )


def test_the_anchored_ring_refuses_a_channel_chunk():
    ic = _field(channels=3, seed=1)
    _, boundary = _anchored(REGION, ic, {0.0: ic.clone()}, block=8)
    topo = _topology(REGION)
    gathered = torch.zeros(1, 2, topo.total_tiles, TILE, TILE)
    with pytest.raises(ValueError, match="every input channel"):
        boundary(gathered)


def test_smooth_keeps_constants_and_removes_grid_scale_noise():
    const = torch.full((2, 64, 96), 3.0)
    torch.testing.assert_close(smooth(const, 8), const)
    rows, cols = torch.meshgrid(torch.arange(64), torch.arange(96), indexing="ij")
    checker = ((rows + cols) % 2 * 2 - 1).float()  # every 8 x 8 block averages to 0
    torch.testing.assert_close(
        smooth(checker, 8), torch.zeros_like(checker), rtol=0, atol=1e-6
    )
    assert torch.equal(smooth(checker, 1), checker)


def test_faces_to_box_is_the_plane_crop(plane_region):
    topo = _topology(plane_region)
    field = torch.randn(2, 6, FACE, FACE)
    (piece,) = topo.ring_pieces
    origin, size = piece.origin, piece.size
    plane = faces_to_plane(field, topo.region_plane)
    expect = plane[:, origin[0] : origin[0] + size[0], origin[1] : origin[1] + size[1]]
    assert torch.equal(faces_to_box(field, topo, origin, size), expect)


def test_a_perfect_boundary_reproduces_the_global_run(grid, region):
    """Drivers taken from the global run itself, unsmoothed: the anchored ring is the
    global state, so the window evolves exactly as the global run does."""
    g_topo, r_topo = _topology(None), _topology(region)
    g_halo, r_halo = _halo(g_topo, grid), _halo(r_topo, grid)
    model = _StencilModel()
    field = torch.randn(2, 6, FACE, FACE)
    state = _tiles(field, g_topo.global_tiles)
    states = [state]
    for _ in range(3):
        state = g_topo.crop(model(g_halo(state)))
        states.append(state)

    def box(tiles):
        return faces_to_pieces(g_topo.gather_tiles_to_faces(tiles)[0], r_topo)

    boundary = AnchoredBoundary(
        topology=r_topo,
        ic_boxes=box(states[0]),
        driver0=box(states[0]),
        block=1,
        device=torch.device("cpu"),
    )
    for h in range(1, 4):
        boundary.add(float(h), box(states[h]))
    r_halo.ring_source = boundary
    index = [g_topo.global_tiles.index(t) for t in r_topo.global_tiles]
    x = states[0][:, :, index]
    for h in range(3):
        boundary.time_h = float(h)
        x = r_topo.crop(model(r_halo(x)))
        torch.testing.assert_close(x, states[h + 1][:, :, index], rtol=0, atol=1e-5)


# ---------------------------------------------------------------- region finder


def _faces_deg(grid):
    lat, lon = (g[:, PAD : PAD + FACE, PAD : PAD + FACE].numpy() for g in grid)
    return lat.astype(np.float64), lon.astype(np.float64)


@pytest.mark.parametrize(
    "face, row, col",
    [
        (3, 96, 96),  # face centre: a window on face 3 alone
        (3, 96, 186),  # next to the east edge of face 3
        (0, 96, 3),  # next to the west edge of face 0
        (3, 186, 96),  # next to the north edge of face 3
        (3, 3, 96),  # next to the south edge of face 3
        (5, 96, 96),  # the north pole
        (4, 96, 96),  # the south pole
    ],
)
def test_find_region_holds_the_point_centred(grid, face, row, col):
    lat, lon = _faces_deg(grid)
    point = (float(lat[face, row, col]), float(lon[face, row, col]))
    spec = find_region(lat, lon, *point, n_tiles=4, tile_size=TILE)
    topo = _topology(spec)  # validates the rectangle
    assert topo.total_tiles == 16
    assert (face, row // TILE, col // TILE) in topo.global_tiles
    assert nearest_pixel(lat, lon, *point) == (face, row, col)
    # the point's tile is one of the 2 x 2 centre tiles of the window
    i, j, _ = topo.tile_plane_position((face, row // TILE, col // TILE))
    _, i0, _, j0, _ = topo.region_box
    assert i - i0 in (1, 2) and j - j0 in (1, 2)


def test_find_region_holds_a_point_at_a_cube_corner(grid):
    """No plane centres a window on a corner tile; the window moves to hold it."""
    lat, lon = _faces_deg(grid)
    point = (float(lat[4, 3, 3]), float(lon[4, 3, 3]))
    topo = _topology(find_region(lat, lon, *point, n_tiles=4, tile_size=TILE))
    assert (4, 0, 0) in topo.global_tiles and not topo.ring_on_plane


def test_find_region_refuses_a_window_too_large_for_a_plane(grid):
    lat, lon = _faces_deg(grid)
    with pytest.raises(ValueError, match="fewer tiles"):
        find_region(lat, lon, float(lat[4, 96, 96]), float(lon[4, 96, 96]), 7, TILE)


# ------------------------------------------------------- window from a lat/lon box

# 12 tiles per face side, so a box leaves room to choose among windows
BIG = 2 * FACE
BOXES = {
    "face": (-10.0, 10.0, -10.0, 10.0),  # the centre of face 0
    "east_edge": (-10.0, 10.0, 35.0, 55.0),  # across lon 45: faces 0 and 1
    "north_edge": (38.0, 52.0, -8.0, 8.0),  # across the north edge of face 0
    "south_edge": (-52.0, -38.0, -8.0, 8.0),  # across the south edge of face 0
    "dateline": (-10.0, 10.0, 170.0, -170.0),  # the centre of face 2
}


@pytest.fixture(scope="module")
def big_grid():
    lat, lon = _cube_latlon(BIG, 0)
    return lat.numpy().astype(np.float64), lon.numpy().astype(np.float64)


def _window_holds(lat, lon, box, tiles, world_size, margin):
    """The run's own check (TileTopology), and every box cell ``margin`` tiles inside."""
    try:
        topo = TileTopology(
            world_size=world_size,
            rank=0,
            face_size=BIG,
            tile_size=TILE,
            halo_width=HALO,
            active_tiles=tiles,
        )
    except ValueError:
        return False
    _, r0, r1, c0, c1 = topo.region_pixel_box()
    m = margin * TILE
    for f, p, q in zip(*np.nonzero(box_mask(lat, lon, box))):
        try:
            P, Q = to_plane(topo.region_plane, int(f), int(p), int(q), BIG)
        except ValueError:
            return False  # a box cell off the window's plane
        if not (r0 + m <= P < r1 - m and c0 + m <= Q < c1 - m):
            return False
    return True


@pytest.mark.parametrize("world_size", [4, 6])
@pytest.mark.parametrize("name", sorted(BOXES))
def test_the_box_window_holds_the_box_and_is_locally_minimal(
    big_grid, name, world_size
):
    lat, lon = big_grid
    box = BOXES[name]
    spec, margin = find_box_region(lat, lon, box, TILE, world_size, margin_tiles=1)
    assert margin >= 1
    tiles = parse_region(spec, BIG // TILE)
    assert len(tiles) % world_size == 0
    assert _window_holds(lat, lon, box, tiles, world_size, 1)
    # without any one edge row or column of tiles the window no longer does
    topo = TileTopology(
        world_size=world_size,
        rank=0,
        face_size=BIG,
        tile_size=TILE,
        halo_width=HALO,
        active_tiles=tiles,
    )
    _, i0, i1, j0, j1 = topo.region_box
    for a0, a1, b0, b1 in (
        (i0 + 1, i1, j0, j1),
        (i0, i1 - 1, j0, j1),
        (i0, i1, j0 + 1, j1),
        (i0, i1, j0, j1 - 1),
    ):
        smaller = tuple(
            from_plane(topo.region_plane, i, j, BIG // TILE)
            for i in range(a0, a1)
            for j in range(b0, b1)
        )
        assert not _window_holds(lat, lon, box, smaller, world_size, 1)


def test_a_box_across_a_face_edge_gets_the_joined_plane(big_grid):
    lat, lon = big_grid
    for name, plane in (
        ("east_edge", plane_blocks(0, east=True)),
        ("north_edge", plane_blocks(0, north=True)),
        ("south_edge", plane_blocks(0, south=True)),
    ):
        spec, _ = find_box_region(lat, lon, BOXES[name], TILE, 4)
        assert region_plane_of(spec, BIG // TILE) == plane


def test_a_dateline_box_is_the_arc_east_of_lon0(big_grid):
    lat, lon = big_grid
    mask = box_mask(lat, lon, BOXES["dateline"])
    assert mask.any()
    assert np.all(np.abs(lon[mask]) >= 170) and np.all(np.abs(lat[mask]) <= 10)
    assert set(np.nonzero(mask)[0].tolist()) == {2}


def test_a_box_at_a_plane_edge_gets_a_window_flush_with_it(big_grid):
    """South of face 0 next to its east edge: only the south-joined plane holds the box,
    and the window reaches that plane's east edge, so its ring leaves the plane."""
    lat, lon = big_grid
    box = (-50.0, -40.0, 30.0, 40.0)  # face 0 column 10, one short of its edge
    spec, margin = find_box_region(lat, lon, box, TILE, 4, margin_tiles=1)
    tiles = parse_region(spec, BIG // TILE)
    assert margin >= 1 and _window_holds(lat, lon, box, tiles, 4, 1)
    topo = TileTopology(
        world_size=4,
        rank=0,
        face_size=BIG,
        tile_size=TILE,
        halo_width=HALO,
        active_tiles=tiles,
    )
    assert topo.region_plane == plane_blocks(0, south=True)
    assert not topo.ring_on_plane


def test_a_box_no_plane_can_hold_is_refused(big_grid):
    lat, lon = big_grid
    box = (-60.0, 60.0, -10.0, 10.0)  # from face 4 across face 0 to face 5
    with pytest.raises(ValueError, match="too large for one plane"):
        find_box_region(lat, lon, box, TILE, 4)


def test_parse_region_rejects_malformed_specs():
    with pytest.raises(ValueError, match="face,i0,i1,j0,j1"):
        parse_region("3,1,5", FACE // TILE)
    with pytest.raises(ValueError, match="empty"):
        parse_region("3,4,4,1,5", FACE // TILE)
