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

import os
from collections import OrderedDict

import numpy as np
import pytest
import torch

from screamcast.distributed_halo import (
    DistributedTileKNNHaloPadding_AllGather,
    NormalizedHaloAdjoint,
    TileTopology,
    faces_to_plane,
)
from screamcast.regional import (
    AnchoredBoundary,
    ReplicateRing,
    edge_distance,
    faces_to_box,
    find_region,
    nearest_pixel,
    parse_region,
    ring_box,
    smooth,
)
from screamcast.zarr_writer import FaceZarrWriteStep

FACE, TILE, HALO, PAD = 192, 32, 4, 8
REGION = "3,1,5,1,5"  # face 3, tiles 1:5 x 1:5 (128 px), ring tiles 0 and 5
# 4 x 4 tiles across a face edge (6 tiles per face side): faces 3 and 0 (cols 4-5 of
# face 3, 0-1 of face 0), and faces 3 and 5 (rows 4-5 of face 3, two rows of face 5)
REGIONS = {"face": REGION, "east": "3,1,5,4,8", "north": "3,4,8,1,5"}


@pytest.fixture(params=sorted(REGIONS))
def region(request):
    return REGIONS[request.param]


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
    with pytest.raises(ValueError, match="ring tile"):
        _topology("3,0,4,1,5")  # touches the face edge: no ring on this face
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
    with pytest.raises(ValueError, match="ring tile"):
        _topology("3,1,5,8,12")  # its ring would leave the plane of faces 3 and 0
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
    x = torch.randn(2, 16, TILE, TILE)
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


@pytest.mark.parametrize("side", ["east", "north"])
def test_the_plane_joins_faces_where_the_grid_does(grid, side):
    """Across the seam, neighbouring plane pixels are about one grid cell apart, as
    within a face (a wrong join or turn puts them a face apart): the plane's join and
    turn match the cube. Near the cube corners the grid kinks, so "about" is 25 %."""
    lat, lon = (torch.deg2rad(g[:, PAD : PAD + FACE, PAD : PAD + FACE]) for g in grid)
    blocks = _topology(REGIONS[side]).region_plane
    la, lo = faces_to_plane(lat, blocks), faces_to_plane(lon, blocks)
    xyz = torch.stack([la.cos() * lo.cos(), la.cos() * lo.sin(), la.sin()])
    if side == "north":
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
        for side in ("east", "north"):
            blocks = plane_blocks(face, **{side: True})
            la = faces_to_plane(faces["lat"], blocks)
            lo = faces_to_plane(faces["lon"], blocks)
            xyz = torch.stack([la.cos() * lo.cos(), la.cos() * lo.sin(), la.sin()])
            if side == "north":
                xyz = xyz.transpose(1, 2)
            across = (xyz[:, :, n] - xyz[:, :, n - 1]).norm(dim=0)
            inside = (xyz[:, :, n - 1] - xyz[:, :, n - 2]).norm(dim=0)
            torch.testing.assert_close(across, inside, rtol=0.25, atol=0)


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


def test_halo_reads_only_a_thin_band_of_the_ring(grid, region):
    r_topo = _topology(region)
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


def test_replicate_ring_extends_the_edge_pixels(region):
    topo = _topology(region)
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


def _box(region, channels=2, seed=0):
    origin, size = ring_box(_topology(region))
    g = torch.Generator().manual_seed(seed)
    return torch.randn(channels, *size, generator=g)


def _anchored(region, ic, drivers, block):
    topo = _topology(region)
    boundary = AnchoredBoundary(
        topology=topo,
        ic_box=ic,
        driver0=drivers[0.0],
        block=block,
        device=torch.device("cpu"),
    )
    for h, box in drivers.items():
        if h:
            boundary.add(h, box)
    return topo, boundary


def test_anchored_ring_starts_at_the_initial_state(region):
    ic = _box(region, seed=1)
    drivers = {0.0: _box(region, seed=2), 6.0: _box(region, seed=3)}
    topo, boundary = _anchored(region, ic, drivers, block=8)
    origin, _ = ring_box(topo)
    assert torch.equal(
        boundary.ring_at(0.0), topo.box_to_tiles(ic, topo.ring_tiles, origin)
    )


def test_anchored_ring_adds_the_driver_change_unsmoothed(region):
    ic = _box(region, seed=1)
    d0, d6 = _box(region, seed=2), _box(region, seed=3)
    topo, boundary = _anchored(region, ic, {0.0: d0, 6.0: d6}, block=1)
    origin, _ = ring_box(topo)
    expect = topo.box_to_tiles(ic + (d6 - d0), topo.ring_tiles, origin)
    torch.testing.assert_close(boundary.ring_at(6.0), expect, rtol=0, atol=1e-5)


def test_anchored_ring_is_linear_in_time():
    ic = _box(REGION, seed=1)
    drivers = {0.0: _box(REGION, seed=2), 6.0: _box(REGION, seed=3)}
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
    ic = _box(region, seed=1)
    d = _box(region, seed=2)
    topo, boundary = _anchored(region, ic, {0.0: d, 6.0: d.clone()}, block=8)
    torch.testing.assert_close(boundary.ring_at(6.0), boundary.ring_at(0.0))


def test_a_uniform_driver_change_shifts_the_ring_uniformly(region):
    """A spatially uniform change of +2.5 passes the smoothing unchanged."""
    ic = _box(region, seed=1)
    d0 = _box(region, seed=2)
    topo, boundary = _anchored(region, ic, {0.0: d0, 6.0: d0 + 2.5}, block=8)
    torch.testing.assert_close(
        boundary.ring_at(6.0) - boundary.ring_at(0.0),
        torch.full_like(boundary.ring_at(0.0), 2.5),
        rtol=0,
        atol=1e-5,
    )


def test_the_anchored_ring_refuses_a_channel_chunk():
    ic = _box(REGION, channels=3, seed=1)
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


def test_faces_to_box_is_the_plane_crop(region):
    topo = _topology(region)
    field = torch.randn(2, 6, FACE, FACE)
    origin, size = ring_box(topo)
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
    origin, size = ring_box(r_topo)

    def box(tiles):
        return faces_to_box(
            g_topo.gather_tiles_to_faces(tiles)[0], r_topo, origin, size
        )

    boundary = AnchoredBoundary(
        topology=r_topo,
        ic_box=box(states[0]),
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
        (5, 96, 96),  # the north pole
    ],
)
def test_find_region_holds_the_point_and_keeps_its_ring(grid, face, row, col):
    lat, lon = _faces_deg(grid)
    point = (float(lat[face, row, col]), float(lon[face, row, col]))
    spec = find_region(lat, lon, *point, n_tiles=4, tile_size=TILE)
    topo = _topology(spec)  # validates the rectangle and the ring
    assert topo.total_tiles == 16
    assert (face, row // TILE, col // TILE) in topo.global_tiles
    assert nearest_pixel(lat, lon, *point) == (face, row, col)


def test_find_region_refuses_a_window_too_large_for_a_plane(grid):
    lat, lon = _faces_deg(grid)
    with pytest.raises(ValueError, match="fewer tiles"):
        find_region(lat, lon, float(lat[4, 96, 96]), float(lon[4, 96, 96]), 5, TILE)


def test_parse_region_rejects_malformed_specs():
    with pytest.raises(ValueError, match="face,i0,i1,j0,j1"):
        parse_region("3,1,5", FACE // TILE)
    with pytest.raises(ValueError, match="empty"):
        parse_region("3,4,4,1,5", FACE // TILE)
