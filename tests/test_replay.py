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
"""Replay (incremental analysis update): the schedule, the increment and its execution.

The increment tests run on the gnomonic toy cube of ``test_regional`` and check closed
forms the code never sees: a uniform offset comes back whole, a single spherical
harmonic (evaluated with scipy) comes back scaled by the taper's gain at its degree,
the omega increment loses the degrees the omega high-pass removes, and a run split over
two ranks gets the same forcing as a single-rank run. The executor tests use an
identity model, whose replayed state must land on the driver.
"""

import threading
from collections import OrderedDict

import numpy as np
import pytest
import scipy.special
import torch

from screamcast.distributed_halo import TileTopology
from screamcast.regional import parse_region
from screamcast.replay import (
    Replay,
    ReplayStep,
    analysis_hours,
    block_cells,
    replay_channels,
    replay_plan,
    run_with_replay,
)
from screamcast.sht_lowpass import gaussian_gain, gaussian_lmax
from tests.test_regional import _cube_area, _cube_latlon

FACE, TILE = 96, 32
IN_VARS = ["U_8", "U_9", "omega_8", "T_2m", "coszr"]
ELL_HALF = 3.0


# ------------------------------------------------------------------ schedule


def test_plan_has_one_main_window_per_hour_in_order():
    plan = replay_plan(24, 6, {0, 6, 12, 18, 24})
    main = [s.hour for s in plan if s.kind == "window" and s.main]
    assert main == list(range(1, 25))
    assert analysis_hours(plan) == [6, 12, 18]  # 24 + 3 > 24


def test_each_analysis_is_predicted_then_rerun_over_its_centred_window():
    plan = replay_plan(24, 6, {6, 12, 18})
    for t in (6, 12, 18):
        i = plan.index(ReplayStep("analysis", t))
        assert plan[i + 1] == ReplayStep("restore", t - 3)
        predictor = [s.hour for s in plan[:i] if s.kind == "window" and not s.main]
        assert predictor[-3:] == [t - 2, t - 1, t]
        assert plan.index(ReplayStep("snapshot", t - 3)) < i
        forced = [s.hour for s in plan if s.kind == "window" and s.forced_by == t]
        assert forced == list(range(t - 2, t + 4))
        assert all(s.main for s in plan if s.forced_by == t)


def test_missing_driver_leaves_its_window_free():
    plan = replay_plan(24, 6, {6, 18})
    assert analysis_hours(plan) == [6, 18]
    assert not any(s.forced_by == 12 for s in plan)
    assert [s.hour for s in plan if s.kind == "window" and s.main] == list(range(1, 25))


def test_no_driver_or_short_run_is_a_plain_run():
    assert replay_plan(24, 6, set()) == [ReplayStep("window", h) for h in range(1, 25)]
    assert analysis_hours(replay_plan(8, 6, {6})) == []


def test_window_may_end_at_the_last_hour():
    plan = replay_plan(9, 6, {6})
    assert analysis_hours(plan) == [6]
    assert [s.hour for s in plan if s.kind == "window" and s.main] == list(range(1, 10))


def test_interval_must_be_even():
    with pytest.raises(ValueError, match="even"):
        replay_plan(24, 3, {3})


def test_channels_match_levels_and_surface_names():
    assert replay_channels(IN_VARS, ["U", "T_2m"]) == [0, 1, 3]
    with pytest.raises(ValueError, match="qv"):
        replay_channels(IN_VARS, ["U", "qv"])


def test_block_cells_keep_area_and_uniform_blocks():
    lat, lon = (a.numpy() for a in _cube_latlon(FACE, 0))
    area = _cube_area(FACE)
    colat, lon_c, a = block_cells(lat, lon, area, 8)
    assert a.shape == (6, FACE // 8, FACE // 8)
    np.testing.assert_allclose(a.sum(), 4 * np.pi, rtol=1e-12)
    # a block of identical points has that point as its centre
    same = np.full((1, 8, 8), 30.0)
    c, lo, _ = block_cells(same, same, np.ones((1, 8, 8)), 8)
    np.testing.assert_allclose(np.rad2deg(c), 60.0, atol=1e-9)
    np.testing.assert_allclose(np.rad2deg(lo), 30.0, atol=1e-9)


# ------------------------------------------------------------------ increment


def _topology(world_size=1, rank=0, region=None) -> TileTopology:
    return TileTopology(
        world_size=world_size,
        rank=rank,
        face_size=FACE,
        tile_size=TILE,
        halo_width=4,
        active_tiles=parse_region(region, FACE // TILE) if region else None,
    )


def _replay(topology, omega_lmax=0, window_steps=36) -> Replay:
    lat, lon = (a.numpy() for a in _cube_latlon(FACE, 0))
    return Replay(
        in_vars=IN_VARS,
        replay_vars=["U", "omega", "T_2m"],
        lat_face=lat,
        lon_face=lon,
        area_face=_cube_area(FACE),
        topology=topology,
        ell_half=ELL_HALF,
        window_steps=window_steps,
        omega_lmax=omega_lmax,
    )


def _local(topology, faces: np.ndarray) -> torch.Tensor:
    """``(C, 6, n, n)`` -> ``(1, C, tiles, ts, ts)``."""
    return topology.faces_to_local_tiles(torch.from_numpy(faces).float())[None]


def _harmonic(ell: int, m: int) -> np.ndarray:
    """Real orthonormal Y_lm on the toy cube's pixels, from scipy (Condon-Shortley
    phase removed, the module's convention)."""
    lat, lon = (np.deg2rad(a.double().numpy()) for a in _cube_latlon(FACE, 0))
    y = scipy.special.sph_harm_y(ell, abs(m), np.pi / 2 - lat, lon)
    if m == 0:
        return y.real
    return (-1) ** m * np.sqrt(2.0) * (y.real if m > 0 else y.imag)


def _state(seed=0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(len(IN_VARS), 6, FACE, FACE))


def test_same_state_gives_zero_forcing():
    topo = _topology()
    rep = _replay(topo)
    x = _local(topo, _state())
    rep.add_target(6, x)
    forcing, stats = rep.increment(x, 6)
    assert torch.count_nonzero(forcing) == 0
    assert stats["U_8"] == (0.0, 0.0)


def test_uniform_offset_is_added_whole_over_the_window():
    topo = _topology()
    rep = _replay(topo, window_steps=36)
    state = _state()
    target = state.copy()
    target[0] += 2.0  # U_8
    target[3] -= 1.0  # T_2m
    target[4] += 5.0  # coszr: not replayed
    x = _local(topo, state)
    rep.add_target(6, _local(topo, target))
    forcing, _ = rep.increment(x, 6)
    total = forcing * 36
    np.testing.assert_allclose(total[0, 0].numpy(), 2.0, atol=0.02)
    np.testing.assert_allclose(total[0, 3].numpy(), -1.0, atol=0.01)
    assert torch.count_nonzero(forcing[0, [1, 2, 4]]) == 0


@pytest.mark.parametrize("ell,m", [(1, 0), (2, -1), (3, 2), (6, 4)])
def test_single_harmonic_comes_back_scaled_by_its_gain(ell, m):
    topo = _topology()
    rep = _replay(topo, window_steps=1)
    state = _state()
    target = state.copy()
    y = _harmonic(ell, m)
    target[1] += y  # U_9
    rep.add_target(6, _local(topo, target))
    forcing, _ = rep.increment(_local(topo, state), 6)
    gain = float(gaussian_gain(gaussian_lmax(ELL_HALF), ELL_HALF)[ell * ell])
    expect = _local(topo, (gain * y)[None])[0, 0].numpy()
    got = forcing[0, 1].numpy()
    # 8 x 8 block means and cube quadrature: a few percent of the harmonic's amplitude
    assert np.abs(got - expect).max() < 0.03 * np.abs(y).max()
    assert np.abs(forcing[0, 0].numpy()).max() < 0.03 * np.abs(y).max()  # U_8 untouched


def test_omega_increment_drops_the_degrees_the_high_pass_removes():
    topo = _topology()
    rep = _replay(topo, omega_lmax=1, window_steps=1)
    state = _state()
    target = state.copy()
    target[2] += 3.0 + _harmonic(1, 1) + _harmonic(3, 0)  # omega_8
    rep.add_target(6, _local(topo, target))
    forcing, _ = rep.increment(_local(topo, state), 6)
    gain3 = float(gaussian_gain(gaussian_lmax(ELL_HALF), ELL_HALF)[9])
    y3 = _harmonic(3, 0)
    expect = _local(topo, (gain3 * y3)[None])[0, 0].numpy()
    assert np.abs(forcing[0, 2].numpy() - expect).max() < 0.03 * np.abs(y3).max()


def test_two_ranks_get_the_single_rank_forcing(monkeypatch):
    """The analysis is a sum over blocks: each rank adds its own and the all-reduce
    completes it. The two ranks run in threads; the all-reduce sums their tensors."""
    state = _state()
    target = state + 0.5 * _harmonic(2, 1)[None] + 1.0
    single = _topology()
    rep = _replay(single, window_steps=1)
    rep.add_target(6, _local(single, target))
    want, want_stats = rep.increment(_local(single, state), 6)

    barrier = threading.Barrier(2)
    slots: list = [None, None]
    who = threading.local()

    def all_reduce(t):
        slots[who.rank] = t.clone()
        barrier.wait()
        total = slots[0] + slots[1]
        barrier.wait()
        t.copy_(total)

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    out: dict = {}

    def run(rank):
        who.rank = rank
        topo = _topology(world_size=2, rank=rank)
        r = _replay(topo, window_steps=1)
        r.add_target(6, _local(topo, target))
        out[rank] = r.increment(_local(topo, state), 6)

    threads = [threading.Thread(target=run, args=(r,)) for r in (0, 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    got = torch.cat([out[0][0], out[1][0]], dim=2)  # rank 0's tiles, then rank 1's
    torch.testing.assert_close(got, want, rtol=0, atol=1e-5)
    for name in rep.names:
        for rank in (0, 1):
            np.testing.assert_allclose(out[rank][1][name], want_stats[name], rtol=1e-6)


def test_window_increment_is_zero_outside_and_bounded_by_the_offset():
    """A regional run analyses its window's blocks only: a uniform offset becomes the
    low-passed window indicator times the offset, between 0 and the offset."""
    topo = _topology(region="3,1,2,1,2")
    rep = _replay(topo, window_steps=1)
    state = _state()
    target = state.copy()
    target[0] += 4.0
    rep.add_target(6, _local(topo, target))
    forcing, _ = rep.increment(_local(topo, state), 6)
    u = forcing[0, 0].numpy()
    assert u.min() > -0.2 and u.max() < 4.2
    assert u.mean() > 0.5  # the window gets a real share of the offset


def test_non_finite_driver_is_refused():
    topo = _topology()
    rep = _replay(topo)
    target = _state()
    target[0, 3, 5, 5] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        rep.add_target(6, _local(topo, target))


# ------------------------------------------------------------------ execution


def _identity_run(n_hours, every, target_of, steps_per_hour=6, drift=0.0):
    """Scalar model: each step adds ``drift / steps_per_hour`` and the forcing."""
    written: list[tuple[int, float]] = []
    window = every * steps_per_hour

    def advance(state, hour, forcing, write):
        for _ in range(steps_per_hour):
            state = (
                state + drift / steps_per_hour + (0.0 if forcing is None else forcing)
            )
        if write:
            written.append((hour, float(state)))
        return state

    def increment(state, hour):
        return (target_of(hour) - state) / window

    plan = replay_plan(n_hours, every, set(range(0, n_hours + 1, every)))
    final = run_with_replay(
        torch.tensor(0.0, dtype=torch.float64), plan, advance, increment, torch.clone
    )
    return written, float(final)


def test_identity_model_lands_on_the_driver():
    written, final = _identity_run(24, 6, lambda h: 10.0)
    assert [h for h, _ in written] == list(range(1, 25))
    by_hour = dict(written)
    assert by_hour[3] == 0.0  # before the first window
    np.testing.assert_allclose(by_hour[6], 5.0)  # half the increment is in by t
    np.testing.assert_allclose(by_hour[9], 10.0)  # all of it at t + every/2
    np.testing.assert_allclose(final, 10.0)


def test_each_analysis_sees_the_predicted_state_not_the_corrected_one():
    """Driver ramps 6 per 6 h. The analysis at 12 is taken on the predictor run from the
    end of window 6 (state 6 at hour 9, unchanged to 12): increment 12 - 6."""
    written, final = _identity_run(24, 6, float)
    by_hour = dict(written)
    np.testing.assert_allclose(by_hour[9], 6.0)
    np.testing.assert_allclose(by_hour[15], 12.0)
    np.testing.assert_allclose(by_hour[21], 18.0)
    np.testing.assert_allclose(final, 18.0)  # hours 22-24 are free


# ------------------------------------------------------------------ runner segments


def _runner_module():
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "ace"
        / "run_screamcast_nudged.py"
    )
    spec = importlib.util.spec_from_file_location("run_screamcast_nudged", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _ToyModel:
    """Inputs (a, b); outputs (b, p, a) in another order plus a diagnostic p."""

    def output_coords(self, coords):
        out = dict(coords)
        out["variable"] = np.array(["b", "p", "a"])
        return out

    def __call__(self, x, coords):
        a, b = x[:, 0], x[:, 1]
        return torch.stack([2.0 * b, a * b, a + 1.0], dim=1), self.output_coords(coords)


class _IdentityTopology:
    def pad_coords(self, coords):
        return dict(coords)

    def crop_coords(self, coords):
        return dict(coords)


def test_runner_segments_continue_the_run_exactly():
    """Replay runs the rollout hour by hour; two segments must equal one call, also
    for a model whose outputs are not its inputs."""
    runner = _runner_module()
    x0 = torch.tensor([1.0, 0.5]).reshape(1, 2, 1, 1, 1)
    coords = OrderedDict(variable=np.array(["a", "b"]))
    args = (_ToyModel(), lambda x: x, lambda x: x, _IdentityTopology())
    whole = runner.run_scream(x0, coords, 6, *args)
    seen: list[int] = []
    written: list[int] = []
    hooks = dict(before_step=seen.append, write_step=lambda i, out: written.append(i))
    x3, c3 = runner.run_scream(x0, coords, 3, *args, return_coords=True, **hooks)
    x6, c6 = runner.run_scream(
        x3, c3, 3, *args, start_step=3, return_coords=True, **hooks
    )
    torch.testing.assert_close(x6, whole, rtol=0, atol=0)
    assert list(c6["variable"]) == ["a", "b"]
    assert seen == written == list(range(6))  # step numbers continue across segments


def test_corrector_restarts_from_the_window_start():
    """Model drifting 1 per hour, driver 2 per hour. Window of the analysis at 6 starts
    at 3 (state 3); the predictor reaches 6, the increment is 12 - 6 = 6, and the
    corrector from 3 reaches 9 + 6 = 15 at hour 9 (18 if it went on from the predictor)."""
    written, _ = _identity_run(9, 6, lambda h: 2.0 * h, drift=1.0)
    by_hour = dict(written)
    np.testing.assert_allclose(by_hour[3], 3.0)
    np.testing.assert_allclose(by_hour[6], 9.0)
    np.testing.assert_allclose(by_hour[9], 15.0)
