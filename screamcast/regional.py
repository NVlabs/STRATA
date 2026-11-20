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
"""Limited-area (regional window) STRATA rollouts.

A regional run predicts only the active tiles of a ``TileTopology``: a rectangle of
tiles on one cube face, or across the edge between an equatorial face and its east or
north neighbour (a region *plane*, see ``distributed_halo.plane_blocks``). Boxes and
distances below are in plane pixels. The halo of the tiles at the region edge reads a
one-tile ``ring`` around the region, whose values come from a ring source:

- ``AnchoredBoundary``: the initial state plus the change of a driver since the start,
  ``ring(t) = IC + (smooth(D(t)) - smooth(D(t0)))``. ``D`` is a sequence of states on
  the SCREAM grid at known valid times (for example ECMWF HRES forecasts built with
  ``screamcast.initial_conditions``); ``smooth`` is an ``n x n`` block mean followed
  by bilinear upsampling, so only the driver's large-scale evolution enters. At
  ``t0`` the ring is the initial state, so the window starts without a jump at its
  edge, and terrain-locked fine structure of the initial state (surface pressure
  over orography, 2 m temperature) is kept. Inside the window STRATA runs freely.
- ``ReplicateRing``: the region's own edge pixels extended outward. No outside
  information enters; useful as a baseline.

``python -m screamcast.regional --lat .. --lon .. --tiles N`` prints a ``--region``
for a window of ``N x N`` tiles around a point, ``--latlon-box LAT0,LAT1,LON0,LON1
--world-size N`` the smallest window that holds a lat/lon box (``find_box_region``).
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import torch
import torch.nn.functional as F

from screamcast.distributed_halo import (
    NORTH_POLAR_FACE,
    Tile,
    TileTopology,
    from_plane,
    plane_blocks,
    plane_parts,
    plane_shape,
    region_plane_for,
    to_plane,
)

logger = logging.getLogger(__name__)


def spec_frame(spec: str, tiles_per_dim: int) -> list:
    """The plane a ``"face,i0,i1,j0,j1"`` spec is written on: ``i1`` beyond the face's
    tiles reaches into the north polar face, ``j1`` beyond them into the next equatorial
    face to the east."""
    face, _, i1, _, j1 = (int(v) for v in spec.split(","))
    return plane_blocks(face, north=i1 > tiles_per_dim, east=j1 > tiles_per_dim)


def region_plane_of(spec: str, tiles_per_dim: int) -> list:
    """The plane a ``TileTopology`` of this region runs on (``region_plane_for``): the
    spec's frame, or for a window flush with an equatorial face's north, east or west
    edge the plane that joins the neighbour holding its ring."""
    return region_plane_for(list(parse_region(spec, tiles_per_dim)), tiles_per_dim)


def parse_region(spec: str, tiles_per_dim: int) -> tuple[Tile, ...]:
    """``"face,i0,i1,j0,j1"`` (tile units on the plane of ``face``, half-open) -> active
    tiles, row-major on the plane. E.g. with 16 tiles per face side ``3,6,14,8,24`` is two
    8 x 8 windows on faces 3 and 0 sharing their edge, ``3,8,24,5,13`` one on face 3 and
    one on the north polar face."""
    parts = spec.split(",")
    if len(parts) != 5:
        raise ValueError(f"--region needs face,i0,i1,j0,j1, got {spec!r}")
    _, i0, i1, j0, j1 = (int(v) for v in parts)
    if not (i0 < i1 and j0 < j1):
        raise ValueError(f"--region {spec!r} is empty")
    blocks = spec_frame(spec, tiles_per_dim)
    return tuple(
        from_plane(blocks, i, j, tiles_per_dim)
        for i in range(i0, i1)
        for j in range(j0, j1)
    )


def ring_box(topology: TileTopology) -> tuple[tuple[int, int], tuple[int, int]]:
    """``(origin, size)`` of the region plus its one-tile ring, in plane pixels."""
    _, r0, r1, c0, c1 = topology.region_pixel_box()
    ts = topology.tile_size
    return (r0 - ts, c0 - ts), (r1 - r0 + 2 * ts, c1 - c0 + 2 * ts)


def faces_to_box(
    x: torch.Tensor,
    topology: TileTopology,
    origin: tuple[int, int],
    size: tuple[int, int],
) -> torch.Tensor:
    """Cut the ``(..., *size)`` box at plane pixel ``origin`` out of ``(..., 6, n, n)``
    face-major fields, without building the whole plane."""
    n = topology.face_size
    r0, c0 = origin
    box = (r0, r0 + size[0], c0, c0 + size[1])
    out = x.new_zeros(*x.shape[:-3], *size)
    for face, (fr0, fr1, fc0, fc1), (pr0, pr1, pc0, pc1), k in plane_parts(
        topology.region_plane, box, n
    ):
        part = x[..., face, fr0:fr1, fc0:fc1]
        out[..., pr0 - r0 : pr1 - r0, pc0 - c0 : pc1 - c0] = torch.rot90(
            part, k, dims=(-2, -1)
        )
    return out


def smooth(box: torch.Tensor, n: int) -> torch.Tensor:
    """``n x n`` block means of ``(..., H, W)`` upsampled back bilinearly (cell-centred,
    edges clamped). ``n = 1`` returns ``box``. A constant stays constant."""
    if n == 1:
        return box
    H, W = box.shape[-2:]
    if H % n or W % n:
        raise ValueError(f"box {H} x {W} is not a multiple of the block size {n}")
    lead = box.shape[:-2]
    flat = box.reshape(-1, 1, H, W)
    coarse = F.avg_pool2d(flat.double(), n)
    fine = F.interpolate(coarse, scale_factor=n, mode="bilinear", align_corners=False)
    return fine.to(box.dtype).reshape(*lead, H, W)


def edge_distance(topology: TileTopology, tiles: list[Tile]) -> torch.Tensor:
    """Distance (px) of every pixel of ``tiles`` (in each tile's own orientation) to the
    region edge on the plane, ``(len(tiles), ts, ts)``.

    0 is the outermost row/column inside the region; pixels outside are negative.
    """
    _, r0, r1, c0, c1 = topology.region_pixel_box()
    ts = topology.tile_size
    origin = (r0 - ts, c0 - ts)
    rows = torch.arange(origin[0], r1 + ts)
    cols = torch.arange(origin[1], c1 + ts)
    dr = torch.minimum(rows - r0, r1 - 1 - rows)
    dc = torch.minimum(cols - c0, c1 - 1 - cols)
    box = torch.minimum(dr[:, None], dc[None, :])
    return topology.box_to_tiles(box, tiles, origin)


class ReplicateRing:
    """Ring source that extends the region's edge pixels outward (free boundary)."""

    def __init__(self, topology: TileTopology) -> None:
        self.topology = topology

    def __call__(self, gathered: torch.Tensor) -> torch.Tensor:
        topo = self.topology
        _, r0, r1, c0, c1 = topo.region_pixel_box()
        H, W, ts = r1 - r0, c1 - c0, topo.tile_size
        box = topo.tiles_to_box(gathered, topo.global_tiles, (r0, c0), (H, W))
        lead = box.shape[:-2]
        padded = F.pad(box.reshape(-1, 1, H, W), (ts, ts, ts, ts), mode="replicate")
        padded = padded.reshape(*lead, H + 2 * ts, W + 2 * ts)
        return topo.box_to_tiles(padded, topo.ring_tiles, (r0 - ts, c0 - ts))


class AnchoredBoundary:
    """Ring values anchored at the initial state: ``IC + (smooth(D(t)) - smooth(D(t0)))``.

    ``ic_box``, ``driver0`` and every box given to ``add`` are ``(C, H, W)`` boxes of the
    region plus its ring (``ring_box``), in the model's input channel order. ``driver0``
    is the driver at the initial time; ``add(h, box)`` adds the driver ``h`` hours later.
    ``time_h`` (set by the caller before each step) selects the linear-in-time blend of
    the two driver states around it. Ring calls need every channel at once
    (``--halo-channel-chunk 0``).
    """

    def __init__(
        self,
        *,
        topology: TileTopology,
        ic_box: torch.Tensor,
        driver0: torch.Tensor,
        block: int,
        device: torch.device,
        channel_chunk: int = 16,
    ) -> None:
        origin, size = ring_box(topology)
        if tuple(ic_box.shape[-2:]) != size or ic_box.ndim != 3:
            raise ValueError(
                f"IC box {tuple(ic_box.shape)} is not a (C, *{size}) ring box"
            )
        self.topology = topology
        self.origin = origin
        self.shape = tuple(ic_box.shape)
        self.block = int(block)
        self.device = device
        self.channel_chunk = int(channel_chunk)
        self.hours: list[float] = []
        self.ring: dict[float, torch.Tensor] = {}
        self._ring_ic = topology.box_to_tiles(ic_box, topology.ring_tiles, origin)
        self._base = self._smooth_ring(driver0)
        self.add(0.0, driver0)
        self.time_h = 0.0

    def _smooth_ring(self, box: torch.Tensor) -> torch.Tensor:
        """``smooth(box)`` on the ring tiles, ``(C, n_ring, ts, ts)``, a few channels at a
        time to bound memory."""
        if tuple(box.shape) != self.shape:
            raise ValueError(
                f"driver box {tuple(box.shape)} differs from the IC box {self.shape}"
            )
        parts = [
            self.topology.box_to_tiles(
                smooth(box[c : c + self.channel_chunk], self.block),
                self.topology.ring_tiles,
                self.origin,
            )
            for c in range(0, box.shape[0], self.channel_chunk)
        ]
        return torch.cat(parts, dim=0)

    def add(self, h: float, box: torch.Tensor) -> None:
        """Add the driver state ``h`` hours after the initial time."""
        h = float(h)
        ring = self._ring_ic + (self._smooth_ring(box) - self._base)
        if not torch.isfinite(ring).all():
            raise ValueError(f"boundary ring at +{h} h is not finite")
        self.ring[h] = ring.to(self.device)
        self.hours = sorted(self.ring)

    def ring_at(self, t: float) -> torch.Tensor:
        """The ring at ``t`` hours, linear in time between driver states."""
        if not self.hours[0] <= t <= self.hours[-1]:
            raise ValueError(
                f"no boundary driver around +{t:.2f} h (drivers at {self.hours} h)"
            )
        lo = max(h for h in self.hours if h <= t)
        hi = min(h for h in self.hours if h >= t)
        if hi == lo:
            return self.ring[lo]
        a = (t - lo) / (hi - lo)
        return (1.0 - a) * self.ring[lo] + a * self.ring[hi]

    def __call__(self, gathered: torch.Tensor) -> torch.Tensor:
        ring = self.ring_at(self.time_h)
        lead = gathered.shape[:-3]
        if lead[-1] != ring.shape[0]:
            raise ValueError(
                "the anchored ring needs every input channel in one halo call "
                "(run with --halo-channel-chunk 0)"
            )
        return ring.expand(*lead, *ring.shape[-3:])


# ---------------------------------------------------------------- region finder


def _unit_vectors(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    la, lo = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def nearest_pixel(
    lat_faces: np.ndarray, lon_faces: np.ndarray, lat: float, lon: float
) -> tuple[int, int, int]:
    """``(face, row, col)`` of the grid cell nearest to ``(lat, lon)`` on
    ``(6, n, n)`` face-major lat/lon in degrees."""
    target = _unit_vectors(np.asarray(lat), np.asarray(lon))
    best = (-2.0, (0, 0, 0))
    for face in range(lat_faces.shape[0]):
        dot = np.tensordot(target, _unit_vectors(lat_faces[face], lon_faces[face]), 1)
        k = int(np.argmax(dot))
        if dot.flat[k] > best[0]:
            best = (float(dot.flat[k]), (face, *np.unravel_index(k, dot.shape)))
    face, row, col = best[1]
    return int(face), int(row), int(col)


def _candidate_planes(face: int) -> list[tuple[int, bool, bool]]:
    """``(base face, north, east)`` planes that contain ``face``, simplest first."""
    if face < 4:
        return [
            (face, False, False),
            (face, False, True),
            ((face - 1) % 4, False, True),
            (face, True, False),
        ]
    if face == NORTH_POLAR_FACE:
        return [(face, False, False)] + [(g, True, False) for g in range(4)]
    return [(face, False, False)]


def find_region(
    lat_faces: np.ndarray,
    lon_faces: np.ndarray,
    lat: float,
    lon: float,
    n_tiles: int,
    tile_size: int,
) -> str:
    """A ``--region`` of ``n_tiles x n_tiles`` tiles whose centre tile holds the grid cell
    nearest to ``(lat, lon)``, moved as little as needed to keep its ring on one plane.

    Tries the point's own face first, then the planes joining it to a neighbour.
    """
    n = lat_faces.shape[-1]
    if n % tile_size:
        raise ValueError(
            f"face size {n} is not a multiple of the tile size {tile_size}"
        )
    tpd = n // tile_size
    face, row, col = nearest_pixel(lat_faces, lon_faces, lat, lon)
    best = None
    for base, north, east in _candidate_planes(face):
        blocks = plane_blocks(base, north=north, east=east)
        P, Q = to_plane(blocks, face, row, col, n)
        ci, cj = P // tile_size, Q // tile_size
        rows, cols = plane_shape(blocks, tpd)
        lo_i, hi_i = 1, rows - 1 - n_tiles  # keep one ring tile on the plane
        lo_j, hi_j = 1, cols - 1 - n_tiles
        if hi_i < lo_i or hi_j < lo_j:
            continue
        i0 = min(max(ci - (n_tiles - 1) // 2, lo_i), hi_i)
        j0 = min(max(cj - (n_tiles - 1) // 2, lo_j), hi_j)
        if not (i0 <= ci < i0 + n_tiles and j0 <= cj < j0 + n_tiles):
            continue  # cannot be moved far enough and still hold the point
        shift = abs(i0 - (ci - (n_tiles - 1) // 2)) + abs(
            j0 - (cj - (n_tiles - 1) // 2)
        )
        spec = f"{base},{i0},{i0 + n_tiles},{j0},{j0 + n_tiles}"
        # the spec must describe this plane (a window inside one face would not)
        if region_plane_of(spec, tpd) != blocks:
            continue
        if best is None or shift < best[0]:
            best = (shift, spec)
        if shift == 0:
            break
    if best is None:
        raise ValueError(
            f"no {n_tiles} x {n_tiles}-tile window around ({lat}, {lon}) keeps its ring "
            "on one region plane; use fewer tiles"
        )
    return best[1]


# Every region plane, simplest first: single faces, then east and north joins.
ALL_PLANES = (
    [(f, False, False) for f in range(6)]
    + [(f, False, True) for f in range(4)]
    + [(f, True, False) for f in range(4)]
)


def box_mask(
    lat_faces: np.ndarray, lon_faces: np.ndarray, box: tuple[float, float, float, float]
) -> np.ndarray:
    """Grid cells inside ``box = (lat0, lat1, lon0, lon1)`` (degrees): ``lat0 <= lat <=
    lat1`` and the longitude on the arc running east from ``lon0`` to ``lon1``, so
    ``lon0 > lon1`` (e.g. ``170, -170``) crosses the dateline."""
    lat0, lat1, lon0, lon1 = box
    if not lat0 < lat1:
        raise ValueError(f"latitude range {lat0}..{lat1} is empty")
    width = (lon1 - lon0) % 360
    if width == 0 and lon1 != lon0:
        width = 360
    in_lon = (lon_faces - lon0) % 360 <= width
    return (lat_faces >= lat0) & (lat_faces <= lat1) & in_lon


def find_box_region(
    lat_faces: np.ndarray,
    lon_faces: np.ndarray,
    box: tuple[float, float, float, float],
    tile_size: int,
    world_size: int,
    margin_tiles: int = 1,
) -> tuple[str, int]:
    """The smallest ``--region`` that holds every grid cell of ``box`` (``box_mask``)
    with at least ``margin_tiles`` whole tiles between the box's tiles and the window
    edge, keeps its ring on one region plane and has a multiple of ``world_size``
    tiles. Returns ``(spec, margin)``, ``margin`` the smallest such gap in tiles.

    Every plane that holds all the box's faces is searched for rectangles; ties in
    tile count go to the larger margin, then to the more centred window.
    """
    n = lat_faces.shape[-1]
    if n % tile_size:
        raise ValueError(
            f"face size {n} is not a multiple of the tile size {tile_size}"
        )
    if margin_tiles < 0:
        raise ValueError("margin_tiles must be >= 0")
    tpd = n // tile_size
    mask = box_mask(lat_faces, lon_faces, box)
    if not mask.any():
        raise ValueError(f"the box {box} holds no grid cell")
    hit = mask.reshape(6, tpd, tile_size, tpd, tile_size).any(axis=(2, 4))
    target = list(zip(*(v.tolist() for v in np.nonzero(hit))))
    faces = {f for f, _, _ in target}
    m = margin_tiles
    best, why = None, []
    for base, north, east in ALL_PLANES:
        blocks = plane_blocks(base, north=north, east=east)
        if not faces <= {f for f, _, _, _ in blocks}:
            continue
        pos = np.array([to_plane(blocks, f, i, j, tpd) for f, i, j in target])
        a0, b0 = pos.min(axis=0)
        a1, b1 = pos.max(axis=0) + 1
        rows, cols = plane_shape(blocks, tpd)
        if a0 - m < 1 or b0 - m < 1 or a1 + m > rows - 1 or b1 + m > cols - 1:
            why.append("margin and ring off the plane")
            continue
        # every window [i0, i1) x [j0, j1) holding the box's tiles plus the margin
        i0, i1 = np.meshgrid(np.arange(1, a0 - m + 1), np.arange(a1 + m, rows))
        j0, j1 = np.meshgrid(np.arange(1, b0 - m + 1), np.arange(b1 + m, cols))
        i0, i1, j0, j1 = (v.ravel() for v in (i0, i1, j0, j1))
        area = (i1 - i0)[:, None] * (j1 - j0)[None, :]
        margin = np.minimum(
            np.minimum(a0 - i0, i1 - a1)[:, None], np.minimum(b0 - j0, j1 - b1)[None, :]
        )
        off = np.abs((a0 - i0) - (i1 - a1))[:, None] + np.abs((b0 - j0) - (j1 - b1))
        ok = area % world_size == 0
        if not ok.any():
            why.append(f"no window with a multiple of {world_size} tiles")
            continue
        r, c = np.nonzero(ok)
        order = np.lexsort((off[r, c], -margin[r, c], area[r, c]))
        for k in order:
            key = (
                int(area[r[k], c[k]]),
                -int(margin[r[k], c[k]]),
                int(off[r[k], c[k]]),
            )
            if best is not None and key >= best[0]:
                break
            tiles = tuple(
                from_plane(blocks, i, j, tpd)
                for i in range(i0[r[k]], i1[r[k]])
                for j in range(j0[c[k]], j1[c[k]])
            )
            try:
                topo = TileTopology(
                    world_size=world_size,
                    rank=0,
                    face_size=n,
                    tile_size=tile_size,
                    halo_width=0,
                    active_tiles=tiles,
                )
            except ValueError as e:
                why.append(str(e))
                continue
            best = (key, ",".join(str(v) for v in topo.region_box))
            break
    if best is None:
        detail = "; ".join(sorted(set(why))) or (
            f"no region plane holds faces {sorted(faces)}"
        )
        raise ValueError(
            f"no window holds the box {box} with a {m}-tile margin ({detail}). A ring "
            "across an equatorial face's south edge or a cube corner, or a box too "
            "large for one plane, is not supported; try a smaller box or margin."
        )
    return best[1], -best[0][1]


def load_latlon_faces(path: str) -> tuple[np.ndarray, np.ndarray]:
    """``latlon_ne1024pg2.nc`` (flat ``ncol``, degrees) as ``(6, 2048, 2048)`` faces."""
    import netCDF4

    from screamcast.dali_ext_src import unstructured_to_6faces

    with netCDF4.Dataset(path) as ds:
        lat = np.asarray(ds["lat"][:], dtype=np.float64)
        lon = np.asarray(ds["lon"][:], dtype=np.float64)
    if np.abs(lat).max() <= np.pi / 2 + 1e-6 and np.abs(lon).max() <= 2 * np.pi + 1e-6:
        lat, lon = np.rad2deg(lat), np.rad2deg(lon)
    faces = [
        unstructured_to_6faces(torch.from_numpy(v), 1024, 2).numpy() for v in (lat, lon)
    ]
    return faces[0], faces[1]


def _corners(lat_faces, lon_faces, spec: str, tile_size: int) -> list[tuple]:
    n = lat_faces.shape[-1]
    face, i0, i1, j0, j1 = (int(v) for v in spec.split(","))
    blocks = spec_frame(spec, n // tile_size)
    out = []
    for P, Q in (
        (i0 * tile_size, j0 * tile_size),
        (i0 * tile_size, j1 * tile_size - 1),
        (i1 * tile_size - 1, j0 * tile_size),
        (i1 * tile_size - 1, j1 * tile_size - 1),
    ):
        f, p, q = from_plane(blocks, P, Q, n)
        out.append((float(lat_faces[f, p, q]), float(lon_faces[f, p, q])))
    return out


def parse_latlon_box(text: str) -> tuple[float, float, float, float]:
    """``"LAT0,LAT1,LON0,LON1"`` (degrees) -> a box for ``box_mask``."""
    parts = text.split(",")
    if len(parts) != 4:
        raise ValueError(f"need LAT0,LAT1,LON0,LON1, got {text!r}")
    lat0, lat1, lon0, lon1 = (float(v) for v in parts)
    return lat0, lat1, lon0, lon1


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Print a --region: N x N tiles around a point (--lat/--lon), or the "
            "smallest window that holds a lat/lon box (--latlon-box)."
        )
    )
    parser.add_argument("--lat", type=float)
    parser.add_argument("--lon", type=float)
    parser.add_argument("--tiles", type=int, default=8, help="Window side in tiles.")
    parser.add_argument(
        "--latlon-box",
        help=(
            "LAT0,LAT1,LON0,LON1 in degrees; the box runs east from LON0 to LON1, so "
            "LON0 > LON1 crosses the dateline."
        ),
    )
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--margin-tiles", type=int, default=1)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument(
        "--latlon",
        required=True,
        help="latlon_ne1024pg2.nc (generated by data_prep/scrip_generation).",
    )
    args = parser.parse_args(argv)
    if (args.latlon_box is None) == (args.lat is None or args.lon is None):
        parser.error("give --lat and --lon, or --latlon-box")
    lat_faces, lon_faces = load_latlon_faces(args.latlon)
    if args.latlon_box is not None:
        spec, margin = find_box_region(
            lat_faces,
            lon_faces,
            parse_latlon_box(args.latlon_box),
            args.tile_size,
            args.world_size,
            args.margin_tiles,
        )
        _, i0, i1, j0, j1 = (int(v) for v in spec.split(","))
        print(spec)
        print(
            f"# {i1 - i0} x {j1 - j0} tiles ({(i1 - i0) * (j1 - j0)}), "
            f"{margin}-tile margin, world size {args.world_size}; corners:"
        )
    else:
        spec = find_region(
            lat_faces, lon_faces, args.lat, args.lon, args.tiles, args.tile_size
        )
        print(spec)
        km = args.tiles * args.tile_size * 4.9  # ne1024pg2: about 4.9 km per cell
        print(
            f"# {args.tiles} x {args.tiles} tiles, about {km:.0f} km across; corners:"
        )
    for la, lo in _corners(lat_faces, lon_faces, spec, args.tile_size):
        print(f"#   lat {la:7.2f}  lon {lo:7.2f}")


if __name__ == "__main__":
    main()
