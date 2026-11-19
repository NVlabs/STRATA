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
"""On-the-fly replay targets (``screamcast.replay_targets``).

The build runs ``fast_ic.build_ic`` on CPU on a tiny cube (face 64, ne32pg2) from a
synthetic source on a 9 x 16 lat-lon grid and a random conservative map. The targets
the producer builds must be the block means ``Replay.add_target`` takes of the state a
driver store holds (the whole-grid build, put on the faces with the store's own index
formula), for global runs and for a window built alone. The rest checks the schedule
from the source's hours, the atomic appearance of a target, the markers and the
producer loop in each mode.
"""

import argparse
import json
import os
import time

import netCDF4
import numpy as np
import pytest
import torch
import zarr

from screamcast import replay_targets as rt
from screamcast.distributed_halo import TileTopology
from screamcast.initial_conditions.fast_ic import Era5, build_ic, load_map, load_vert
from screamcast.regional import (
    AnchoredBoundary,
    block_means,
    faces_to_box,
    parse_region,
    ring_box,
)
from screamcast.replay import Replay, ReplayStep, replay_plan
from tests.test_regional import _cube_area, _cube_latlon

FACE, TILE = 64, 16  # 2 x 2 blocks per tile
NCOL = 6 * FACE * FACE
NLAT, NLON = 9, 16
PLEV_HPA = np.array([50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000])
STEMS = ["U", "qv", "T_2m"]
REGION = "3,1,3,1,3"  # 2 x 2 tiles of face 3
EDGE_REGION = "3,3,5,1,2"  # 2 tiles, across the edge to the north polar face (turned)


# ------------------------------------------------------------------ fixtures


def _era5(seed: int) -> Era5:
    rng = np.random.default_rng(seed)
    shape = (len(PLEV_HPA), NLAT, NLON)
    sigma = (PLEV_HPA / 1000.0)[:, None, None]
    atm = {
        "T_mid": 200.0 + 90.0 * sigma + rng.normal(size=shape),
        "qv": 1e-6 + 0.015 * sigma**3 * rng.uniform(size=shape),
        "U": 10.0 * rng.normal(size=shape),
        "V": 10.0 * rng.normal(size=shape),
        "omega": 0.3 * rng.normal(size=shape),
    }
    sfc = {
        "ps": 1.0e5 + 2000.0 * rng.uniform(size=(NLAT, NLON)),
        "phis": 3000.0 * rng.uniform(size=(NLAT, NLON)),
        "T_2m": 290.0 + 5.0 * rng.normal(size=(NLAT, NLON)),
    }
    return Era5(
        atm={k: v.astype(np.float32) for k, v in atm.items()},
        sfc={k: v.astype(np.float32) for k, v in sfc.items()},
        plev_pa=PLEV_HPA.astype(np.float64) * 100.0,
    )


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    """Map, phis and L128 coefficients as files, in the formats fast_ic reads."""
    root = tmp_path_factory.mktemp("inputs")
    rng = np.random.default_rng(0)
    n_a = NLAT * NLON
    rows, cols, weights = [], [], []
    for r in range(NCOL):
        c = np.sort(rng.choice(n_a, size=4, replace=False))
        w = rng.uniform(0.1, 1.0, size=4)
        rows += [r] * 4
        cols += list(c)
        weights += list(w / w.sum())
    with netCDF4.Dataset(root / "map.nc", "w") as ds:
        ds.createDimension("n_s", len(rows))
        ds.createDimension("n_a", n_a)
        ds.createDimension("n_b", NCOL)
        for name, data, dtype in (
            ("S", weights, "f8"),
            ("row", np.array(rows) + 1, "i4"),
            ("col", np.array(cols) + 1, "i4"),
        ):
            ds.createVariable(name, dtype, ("n_s",))[:] = data
    x = np.linspace(0.0, 1.0, 129)
    hyai, hybi = 0.002 + 0.1 * x * (1.0 - x), x
    with netCDF4.Dataset(root / "vert.nc", "w") as ds:
        ds.createDimension("ilev", 129)
        ds.createDimension("lev", 128)
        for name, data, dim in (
            ("hyai", hyai, "ilev"),
            ("hybi", hybi, "ilev"),
            ("hyam", 0.5 * (hyai[1:] + hyai[:-1]), "lev"),
            ("hybm", 0.5 * (hybi[1:] + hybi[:-1]), "lev"),
        ):
            ds.createVariable(name, "f8", (dim,))[:] = data
    phis = (2500.0 * rng.uniform(size=NCOL)).astype(np.float32)
    zarr.create_array(str(root / "phis.zarr"), data=phis)
    spec = {
        "source": "era5",
        "map": str(root / "map.nc"),
        "phis": str(root / "phis.zarr"),
        "vert": str(root / "vert.nc"),
    }
    with open(root / "spec.json", "w") as f:
        json.dump(spec, f)
    return spec, str(root / "spec.json")


def _builder(spec, tiles, group_columns=1 << 21) -> rt.BlockBuilder:
    vert, p0 = load_vert(spec["vert"])
    phis = np.asarray(zarr.open_array(spec["phis"], mode="r")[:])
    return rt.BlockBuilder(
        remap=load_map(spec["map"], torch.device("cpu")),
        phis=torch.from_numpy(phis),
        vert=vert,
        p0=p0,
        tiles=tiles,
        tile_size=TILE,
        face_size=FACE,
        channels=rt.target_channels(STEMS),
        group_columns=group_columns,
    )


def _store_faces(spec, state: Era5) -> tuple[list[str], torch.Tensor]:
    """The whole-grid build as a driver store holds it (float32 on ncol), put on the
    faces as the store reader does: ``(1, C, 6, n, n)``."""
    vert, p0 = load_vert(spec["vert"])
    phis = np.asarray(zarr.open_array(spec["phis"], mode="r")[:])
    flat: dict[str, np.ndarray] = {}

    def emit(name, j, field):
        flat[name if name in rt.SURFACE else f"{name}_{j}"] = field.numpy().copy()

    remap = load_map(spec["map"], torch.device("cpu"))
    build_ic(state, remap, torch.from_numpy(phis), vert, p0, NCOL, emit)
    names = rt.target_channels(STEMS)
    # flat index within a face: (elem_i * ne + elem_j) * 4 + gp_i * 2 + gp_j
    xi, yi = np.meshgrid(np.arange(FACE), np.arange(FACE), indexing="ij")
    in_face = ((xi // 2) * (FACE // 2) + yi // 2) * 4 + (xi % 2) * 2 + yi % 2
    index = in_face[None] + (np.arange(6) * FACE * FACE)[:, None, None]
    faces = np.stack([flat[n][index] for n in names])
    return names, torch.from_numpy(faces)[None]


def _topology(region=None) -> TileTopology:
    return TileTopology(
        world_size=1,
        rank=0,
        face_size=FACE,
        tile_size=TILE,
        halo_width=2,
        active_tiles=parse_region(region, FACE // TILE) if region else None,
    )


def _replay(names, topology) -> Replay:
    lat, lon = (a.numpy() for a in _cube_latlon(FACE, 0))
    return Replay(
        in_vars=[*names, "coszr"],
        replay_vars=STEMS,
        lat_face=lat,
        lon_face=lon,
        area_face=_cube_area(FACE),
        topology=topology,
        ell_half=3.0,
        window_steps=1,
    )


# ------------------------------------------------------------------ build


@pytest.mark.parametrize("region", [None, REGION])
def test_producer_blocks_are_the_driver_store_blocks(inputs, tmp_path, region):
    """Producer path (block means of the window's columns, built alone, stored and read
    back) against the precomputed path (a driver store's state, ``add_target``)."""
    spec, _ = inputs
    state = _era5(1)
    topo = _topology(region)
    names, faces = _store_faces(spec, state)
    tiles_in = torch.cat(
        [
            topo.faces_to_local_tiles(faces),
            torch.zeros(1, 1, len(topo.local_tiles), TILE, TILE),
        ],
        dim=1,
    )
    want = _replay(names, topo)
    want.add_target(6, tiles_in)

    tiles = rt.producer_tiles(region, TILE, FACE)
    blocks = _builder(spec, tiles)(state)
    path = rt.target_path(str(tmp_path), np.datetime64("2024-09-25T18"))
    rt.write_target(
        path,
        blocks,
        channels=names,
        tiles=tiles,
        tile_size=TILE,
        face_size=FACE,
        valid=np.datetime64("2024-09-25T18"),
        history="test",
        source="synthetic",
    )
    got = _replay(names, topo)
    local = rt.read_blocks(path, got.names, topo.local_tiles, TILE)
    got.add_block_target(6, torch.from_numpy(local))
    torch.testing.assert_close(got.targets[6], want.targets[6], rtol=1e-13, atol=0)
    stored = np.asarray(zarr.open_group(path, mode="r")["blocks"][...])
    n_blocks = len(tiles) * (TILE // 8) ** 2
    assert np.isfinite(stored).sum() == len(names) * n_blocks  # NaN off the crop


@pytest.mark.parametrize("region", [REGION, EDGE_REGION])
def test_ring_blocks_are_the_driver_store_blocks(inputs, tmp_path, region):
    """The producer's window-and-ring targets, read back as the ring box, against the
    block means the ring takes of a driver store's state (``faces_to_box``), on one
    face and across a turned face edge."""
    spec, _ = inputs
    state = _era5(3)
    topo = _topology(region)
    names, faces = _store_faces(spec, state)
    want = block_means(faces_to_box(faces[0], topo, *ring_box(topo)), 8)

    tiles = rt.producer_tiles(region, TILE, FACE, ring=True)
    assert len(tiles) == len(topo.global_tiles) + len(topo.ring_tiles)
    path = str(tmp_path / "ring.zarr")
    rt.write_target(
        path,
        _builder(spec, tiles)(state),
        channels=names,
        tiles=tiles,
        tile_size=TILE,
        face_size=FACE,
        valid=np.datetime64("2024-09-25T18"),
        history="test",
        source="synthetic",
    )
    got = torch.from_numpy(rt.read_box_blocks(path, names[::-1], topo))
    torch.testing.assert_close(got, want.flip(0), rtol=1e-13, atol=0)


def test_ring_from_block_means_is_the_ring_from_boxes():
    """``AnchoredBoundary`` given block means equals the one given full driver boxes,
    and a uniform driver change c moves the ring by c."""
    topo = _topology(EDGE_REGION)
    _, (h, w) = ring_box(topo)
    rng = np.random.default_rng(0)
    ic, d0, d6 = (
        torch.from_numpy(rng.normal(size=(3, h, w))).float() for _ in range(3)
    )
    d12 = d0 + torch.tensor([2.0, 0.0, -1.0])[:, None, None]
    kw = dict(topology=topo, ic_box=ic, block=8, device=torch.device("cpu"))
    boxes = AnchoredBoundary(driver0=d0, **kw)
    blocks = AnchoredBoundary(driver0_blocks=block_means(d0, 8), **kw)
    for hour, d in ((6, d6), (12, d12)):
        boxes.add(hour, d)
        blocks.add_blocks(hour, block_means(d, 8))
    for hour in (0, 6, 12):
        torch.testing.assert_close(blocks.ring[hour], boxes.ring[hour], rtol=0, atol=0)
    ring_ic = topo.box_to_tiles(ic, topo.ring_tiles, ring_box(topo)[0])
    shift = (blocks.ring[12] - ring_ic).reshape(3, -1)
    torch.testing.assert_close(shift[:, 0], torch.tensor([2.0, 0.0, -1.0]))
    assert float((shift - shift[:, :1]).abs().max()) < 1e-5
    with pytest.raises(ValueError, match="driver blocks"):
        blocks.add_blocks(18, block_means(d0, 8)[:, :-1])
    with pytest.raises(ValueError, match="driver0 or driver0_blocks"):
        AnchoredBoundary(**kw)


def test_column_groups_give_the_same_bits(inputs, tmp_path):
    """Each column's build is independent of the others: one tile per group equals the
    whole grid in one group, bit for bit (``compare``)."""
    spec, _ = inputs
    state = _era5(2)
    tiles = rt.producer_tiles(None, TILE, FACE)
    paths = []
    for group in (TILE * TILE, 1 << 21):
        blocks = _builder(spec, tiles, group_columns=group)(state)
        paths.append(str(tmp_path / f"g{group}.zarr"))
        rt.write_target(
            paths[-1],
            blocks,
            channels=rt.target_channels(STEMS),
            tiles=tiles,
            tile_size=TILE,
            face_size=FACE,
            valid=np.datetime64("2024-09-25T18"),
            history="test",
            source="synthetic",
        )
    assert rt.compare_stores(*paths) == []


def test_tile_columns_are_the_window_pixels():
    """Each pixel's flat index, laid out on the faces with the store's formula and cut
    into the window's tiles, is the column list of those tiles."""
    xi, yi = np.meshgrid(np.arange(FACE), np.arange(FACE), indexing="ij")
    in_face = ((xi // 2) * (FACE // 2) + yi // 2) * 4 + (xi % 2) * 2 + yi % 2
    faces = in_face[None] + (np.arange(6) * FACE * FACE)[:, None, None]
    topo = _topology("3,1,3,1,3")
    want = topo.faces_to_local_tiles(torch.from_numpy(faces)).reshape(-1).numpy()
    got = rt.tile_columns(topo.local_tiles, TILE, FACE)
    np.testing.assert_array_equal(got, want)
    assert len(set(got)) == len(got) == 4 * TILE * TILE


def test_unknown_variable_is_refused():
    with pytest.raises(ValueError, match="coszr"):
        rt.target_channels(["U", "coszr"])


# ------------------------------------------------------------------ schedule


def test_hres_hours_follow_the_open_data_steps():
    src = rt.HresSource({"init": "2024-09-25T12", "cache": "unused"})
    t0 = np.datetime64("2024-09-25T12")
    assert rt.source_hours(src, t0, range(25)) == set(range(0, 25, 3))
    # a run starting an hour after the forecast
    assert rt.source_hours(src, t0 + np.timedelta64(1, "h"), range(25)) == set(
        range(2, 25, 3)
    )
    # 3-hourly to +144 h, then 6-hourly: a run starting at +138 h
    late = t0 + np.timedelta64(138, "h")
    assert rt.source_hours(src, late, range(25)) == {0, 3, 6, 12, 18, 24}
    plan = replay_plan(24, 6, rt.source_hours(src, late, range(25)))
    assert rt.target_schedule(plan) == [(6, None), (12, 6), (18, 12)]
    assert rt.ring_hours(src, late, 6, 23.83) == [0, 6, 12, 18, 24]
    with pytest.raises(ValueError, match=r"\[9, 15, 21\]"):  # leads 147, 153, 159
        rt.ring_hours(src, late, 3, 23.83)


def test_schedule_reads_ring_drivers_before_the_steps_reach_them():
    """Ring states bracket every step: the window ending at hour k needs the ring hours
    up to the first at or after k. The ring hour 12 is first read by the corrector of
    the analysis at 6 (hours 7-9), so a gated producer builds it after that analysis."""
    plan = replay_plan(24, 6, {6, 12, 18})
    assert rt.target_schedule(plan, [0, 6, 12, 18, 24]) == [
        (0, None),
        (6, None),
        (12, 6),
        (18, 12),
        (24, 18),
    ]
    # a window without replay: no analyses, nothing to gate on
    windows = [ReplayStep("window", h) for h in range(1, 25)]
    assert rt.target_schedule(windows, [0, 6, 12, 18, 24]) == [
        (h, None) for h in (0, 6, 12, 18, 24)
    ]
    # 3-hourly analyses on 6-hourly ring drivers
    plan = replay_plan(12, 2, {2, 4, 6, 8, 10})
    assert rt.target_schedule(plan, [0, 6, 12]) == [
        (0, None),
        (6, None),
        (2, None),
        (4, 2),
        (12, 6),  # first read by the corrector of the analysis at 6 (hour 7)
        (8, 6),
        (10, 8),
    ]


def test_era5_has_every_hour():
    src = rt.Era5Source({})
    t0 = np.datetime64("2017-09-14T00")
    assert rt.source_hours(src, t0, range(25)) == set(range(25))


@pytest.mark.parametrize("init", ["2024-09-25T06", "2024-03-05T00"])
def test_hres_refuses_inits_the_open_data_lacks(init):
    with pytest.raises(ValueError, match="HRES init"):
        rt.HresSource({"init": init, "cache": "unused"})


def test_hres_first_complete_cycle_is_accepted():
    rt.HresSource({"init": "2024-03-05T12", "cache": "unused"})


# ------------------------------------------------------------------ stores and markers


def test_target_appears_only_complete(tmp_path, monkeypatch):
    """The store is complete under its partial name before the rename, and the final
    name does not exist until then."""
    tiles = rt.producer_tiles(REGION, TILE, FACE)
    channels = ["U_0", "T_2m"]
    tb = TILE // 8
    blocks = np.random.default_rng(0).normal(size=(2, len(tiles), tb, tb))
    valid = np.datetime64("2024-09-25T18")
    path = rt.target_path(str(tmp_path), valid)
    real_rename = os.rename
    seen = []

    def rename(src, dst):
        assert dst == path and not os.path.exists(dst)
        assert src == f"{path}.partial{os.getpid()}"
        np.testing.assert_array_equal(
            rt.read_blocks(src, channels, tiles, TILE), blocks
        )
        seen.append(src)
        real_rename(src, dst)

    monkeypatch.setattr(rt.os, "rename", rename)
    rt.write_target(
        path,
        blocks,
        channels=channels,
        tiles=tiles,
        tile_size=TILE,
        face_size=FACE,
        valid=valid,
        history="h",
        source="s",
    )
    assert len(seen) == 1 and sorted(os.listdir(tmp_path)) == [os.path.basename(path)]
    np.testing.assert_array_equal(
        rt.read_blocks(path, channels[::-1], tiles, TILE), blocks[::-1]
    )
    assert rt.covers(path, ["T_2m"], tiles[:1], TILE)
    assert not rt.covers(path, ["U_1"], tiles, TILE)
    assert not rt.covers(path, channels, [(3, 0, 0)], TILE)


def test_wait_for_target_stops_on_failed_done_and_timeout(tmp_path):
    valid = np.datetime64("2024-09-25T18")
    with pytest.raises(TimeoutError):
        rt.wait_for_target(str(tmp_path), valid, timeout=0.2, poll=0.05)
    (tmp_path / rt.DONE).touch()
    with pytest.raises(RuntimeError, match="finished without"):
        rt.wait_for_target(str(tmp_path), valid, timeout=10, poll=0.05)
    (tmp_path / rt.FAILED).write_text("Traceback: boom")
    with pytest.raises(RuntimeError, match="boom"):
        rt.wait_for_target(str(tmp_path), valid, timeout=10, poll=0.05)
    os.makedirs(rt.target_path(str(tmp_path), valid))
    assert rt.wait_for_target(str(tmp_path), valid, timeout=10) < 1.0


# ------------------------------------------------------------------ producer


class _FakeSource:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def has(self, valid):
        return True

    def __call__(self, valid):
        self.calls.append(str(valid))
        if self.fail:
            raise OSError("source unreachable")
        return _era5(int(str(valid)[11:13]))

    def describe(self, valid):
        return "synthetic"


def _produce_args(spec_path, out_dir, mode, region=None):
    return argparse.Namespace(
        spec=spec_path,
        out_dir=str(out_dir),
        run_init="2024-09-25T12",
        hours=[6, 12],
        gates=None,
        mode=mode,
        vars=",".join(STEMS),
        region=region,
        ring=False,
        tile_size=TILE,
        face_size=FACE,
        group_columns=1 << 21,
        device="cpu",
    )


@pytest.mark.parametrize("mode", ["ahead", "gated", "serial"])
def test_producer_builds_every_target_in_each_mode(inputs, tmp_path, monkeypatch, mode):
    spec, spec_path = inputs
    source = _FakeSource()
    monkeypatch.setattr(rt, "make_source", lambda s: source)
    t0 = np.datetime64("2024-09-25T12")
    if mode == "serial":  # the run asks for each target
        for h in (6, 12):
            rt.mark(str(tmp_path), "request", t0 + np.timedelta64(h, "h"))
    if mode == "gated":  # the second target waits for the analysis at +6 h
        rt.mark(str(tmp_path), "analysis", t0 + np.timedelta64(6, "h"))
    rt.produce(_produce_args(spec_path, tmp_path, mode, REGION))
    assert (tmp_path / rt.DONE).exists() and not (tmp_path / rt.FAILED).exists()
    assert source.calls == ["2024-09-25T18:00:00", "2024-09-26T00:00:00"]

    topo = _topology(REGION)
    reader = rt.TargetReader(
        str(tmp_path),
        t0,
        lambda path: rt.read_blocks(
            path, rt.target_channels(STEMS), topo.local_tiles, TILE
        ),
        mode=mode,
        timeout=5,
    )
    reader.prefetch([6, 12])
    tiles = rt.producer_tiles(REGION, TILE, FACE)
    for h in (6, 12):
        blocks, info = reader.get(h)
        want = _builder(spec, tiles)(
            _era5(int(str(t0 + np.timedelta64(h, "h"))[11:13]))
        )
        np.testing.assert_array_equal(blocks, want)
        assert info["blocked_s"] >= 0.0
    reader.close()

    # a rerun keeps the targets that cover it and builds nothing
    rerun = _FakeSource()
    monkeypatch.setattr(rt, "make_source", lambda s: rerun)
    rt.produce(_produce_args(spec_path, tmp_path, "ahead", REGION))
    assert rerun.calls == []
    # a global run cannot use a window's targets
    with pytest.raises(ValueError, match="does not hold"):
        rt.produce(_produce_args(spec_path, tmp_path, "ahead", None))
    assert "does not hold" in (tmp_path / rt.FAILED).read_text()


@pytest.mark.parametrize("mode,built", [("gated", [6]), ("serial", [])])
def test_producer_waits_for_its_marker(inputs, tmp_path, monkeypatch, mode, built):
    """Gated: without the analysis marker the second target is not built, the first is.
    Serial: without a request no target is built."""
    _, spec_path = inputs
    monkeypatch.setattr(rt, "make_source", lambda s: _FakeSource())
    monkeypatch.setattr(rt.time, "sleep", lambda s: None)
    calls = {"n": 0}
    real_exists = os.path.exists

    def exists(p):
        # the marker loop polls os.path.exists; give up after many polls
        if os.path.basename(p).startswith(("analysis_", "request_")):
            calls["n"] += 1
            if calls["n"] > 200:
                raise TimeoutError("still waiting")
        return real_exists(p)

    monkeypatch.setattr(rt.os.path, "exists", exists)
    with pytest.raises(TimeoutError):
        rt.produce(_produce_args(spec_path, tmp_path, mode, REGION))
    t0 = np.datetime64("2024-09-25T12")
    for h in built:
        path = rt.target_path(str(tmp_path), t0 + np.timedelta64(h, "h"))
        deadline = time.time() + 60  # its write may still be on the writer thread
        while not os.path.isdir(path) and time.time() < deadline:
            pass
    for h in (6, 12):
        path = rt.target_path(str(tmp_path), t0 + np.timedelta64(h, "h"))
        assert os.path.isdir(path) == (h in built)
    assert (tmp_path / rt.FAILED).exists()


def test_reader_asks_for_a_missing_target(tmp_path):
    """A rank that finds its target missing writes the request a serial producer (or a
    gated one retrying out of memory) waits for."""
    t0 = np.datetime64("2024-09-25T12")
    reader = rt.TargetReader(
        str(tmp_path), t0, lambda path: None, mode="serial", timeout=0.1
    )
    with pytest.raises(TimeoutError):
        reader.get(6)
    assert (tmp_path / "request_2024092518").exists()


def test_block_target_of_the_wrong_shape_is_refused():
    topo = _topology()
    rep = _replay(rt.target_channels(STEMS), topo)
    tb = TILE // 8
    n, tiles = len(rep.names), len(topo.local_tiles)
    with pytest.raises(ValueError, match="block target"):
        rep.add_block_target(6, torch.zeros(n, tiles, tb, tb + 1, dtype=torch.float64))
    with pytest.raises(ValueError, match="block target"):
        rep.add_block_target(6, torch.zeros(n, tiles, tb, tb))  # float32


def test_failed_source_is_reported(inputs, tmp_path, monkeypatch):
    _, spec_path = inputs
    monkeypatch.setattr(rt, "make_source", lambda s: _FakeSource(fail=True))
    with pytest.raises(OSError):
        rt.produce(_produce_args(spec_path, tmp_path, "ahead"))
    assert "source unreachable" in (tmp_path / rt.FAILED).read_text()
    with pytest.raises(RuntimeError, match="source unreachable"):
        rt.wait_for_target(str(tmp_path), np.datetime64("2024-09-25T18"), timeout=5)


def test_producer_process_failure_reaches_the_readers(tmp_path):
    """A producer child that fails (here: no spec file) leaves FAILED, and a rank
    waiting for a target stops with its message instead of timing out."""
    (tmp_path / "request_2024092518").touch()  # an old run's marker is cleared
    proc = rt.ProducerProcess(
        spec=str(tmp_path / "missing.json"),
        out_dir=str(tmp_path),
        run_init=np.datetime64("2024-09-25T12"),
        hours=[6],
        gates=[None],
        mode="gated",
        stems=STEMS,
        tile_size=TILE,
        face_size=FACE,
        region=None,
        ring=False,
        gpu=None,
    )
    assert not (tmp_path / "request_2024092518").exists()
    assert proc.close(timeout=300) != 0
    deadline = time.time() + 30
    while not (tmp_path / rt.FAILED).exists() and time.time() < deadline:
        time.sleep(0.1)
    with pytest.raises(RuntimeError, match="missing.json"):
        rt.wait_for_target(str(tmp_path), np.datetime64("2024-09-25T18"), timeout=5)
