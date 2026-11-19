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
"""Replay: keep STRATA's large scales on a driver forecast (incremental analysis update).

Every ``every`` hours (analysis time t, where a driver state D(t) exists) the rollout
is compared with the driver on 8 x 8-pixel blocks (about 40 km):

    delta = blockmean(D(t)) - blockmean(x(t))

The increment is low-passed on the sphere (Gaussian spherical-harmonic taper, gain 1/2
at degree ``ell_half``; 160 is about 250 km wavelength) and added in equal parts after
every step of the window centred on t: the run is rewound to t - every/2 and run again
to t + every/2. Half of the increment is in by t; spreading it over the window keeps the
model from a shock. Scales below the taper are STRATA's own throughout.

``replay_plan`` is the schedule, ``Replay`` computes the increments of a distributed
run (each rank analyses its own blocks, the coefficients are all-reduced, each rank
evaluates them on its own pixels) and ``run_with_replay`` executes the schedule with a
caller-supplied stepper.

A regional run analyses the blocks of its window only. The low-pass of the window
increment ``m delta`` (``m`` the window indicator) gives a cell near the window edge only
the share of the kernel that lies inside: a uniform offset comes back at 50 % on the
edge and 84 % one kernel width (about 47 km at ``ell_half`` 160) inside. With
``normalize_edge`` the increment is divided by the low-passed indicator,
``K*(m delta) / K*(m)``, so a uniform offset comes back whole everywhere in the window;
away from the edge both equal the global operator.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from screamcast.distributed_halo import TileTopology
from screamcast.sht_lowpass import (
    gaussian_gain,
    gaussian_lmax,
    sh_analysis,
    sh_synthesis,
)

BLOCK = 8  # fine pixels per analysis block side (ne1024pg2 -> ne128pg2)
DEFAULT_VARS = ("U", "V", "PotentialTemperature", "qv", "z_mid", "omega", "T_2m")


@dataclass(frozen=True)
class ReplayStep:
    """One action of the replay schedule.

    ``window`` advances the run from hour ``hour - 1`` to ``hour``, forced by the
    increment of analysis ``forced_by`` when set; ``main`` is False for the predictor
    windows that the corrector re-runs (their output is not written). ``snapshot`` saves
    the state at ``hour``, ``analysis`` computes the increment at ``hour`` from the
    current state and ``restore`` rewinds to the snapshot of ``hour``.
    """

    kind: str
    hour: int
    forced_by: int | None = None
    main: bool = True


def replay_plan(n_hours: int, every: int, driver_hours: set[int]) -> list[ReplayStep]:
    """Schedule of an ``n_hours`` run replayed every ``every`` hours.

    Analyses fall at the multiples of ``every`` that have a driver state and whose
    centred window ends by ``n_hours``. Hours after the last window are a free forecast.
    """
    if every <= 0 or every % 2:
        raise ValueError(
            f"replay interval must be a positive even number of hours, got {every}"
        )
    half = every // 2
    analyses = [
        t
        for t in range(every, n_hours + 1, every)
        if t + half <= n_hours and t in driver_hours
    ]
    steps: list[ReplayStep] = []
    done = 0  # end hour of the written trajectory so far
    for t in analyses:
        start = t - half
        steps += [ReplayStep("window", h) for h in range(done + 1, start + 1)]
        steps.append(ReplayStep("snapshot", start))
        steps += [ReplayStep("window", h, main=False) for h in range(start + 1, t + 1)]
        steps.append(ReplayStep("analysis", t))
        steps.append(ReplayStep("restore", start))
        steps += [
            ReplayStep("window", h, forced_by=t) for h in range(start + 1, t + half + 1)
        ]
        done = t + half
    steps += [ReplayStep("window", h) for h in range(done + 1, n_hours + 1)]
    return steps


def analysis_hours(plan: list[ReplayStep]) -> list[int]:
    return [s.hour for s in plan if s.kind == "analysis"]


def replay_channels(in_vars: list[str], replay_vars: list[str]) -> list[int]:
    """Indices of the input channels a replay updates: ``<var>`` or ``<var>_<level>``."""

    def stem(name: str) -> str:
        base, sep, level = name.rpartition("_")
        return base if sep and level.isdigit() else name

    out = [
        ci
        for ci, name in enumerate(in_vars)
        if name in replay_vars or stem(name) in replay_vars
    ]
    found = {in_vars[ci] for ci in out} | {stem(in_vars[ci]) for ci in out}
    missing = sorted(set(replay_vars) - found)
    if missing:
        raise ValueError(f"replay variables {missing} are not model inputs")
    return out


def block_mean(x: torch.Tensor, k: int) -> torch.Tensor:
    """Mean over k x k blocks of the last two dimensions."""
    *lead, h, w = x.shape
    if h % k or w % k:
        raise ValueError(f"({h}, {w}) is not divisible into {k} x {k} blocks")
    return x.reshape(*lead, h // k, k, w // k, k).mean(dim=(-3, -1))


def block_cells(
    lat: np.ndarray, lon: np.ndarray, area: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Colatitude, longitude (radians) and area of the k x k blocks of a ``(..., n, n)``
    grid in degrees: centres are the normalised mean unit vector of each block, areas the
    block sums."""
    la = np.deg2rad(np.asarray(lat, np.float64))
    lo = np.deg2rad(np.asarray(lon, np.float64))
    xyz = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)
    *lead, h, w = la.shape
    mean = xyz.reshape(*lead, h // k, k, w // k, k, 3).mean(axis=(-4, -2))
    mean /= np.linalg.norm(mean, axis=-1, keepdims=True)
    colat = np.arccos(np.clip(mean[..., 2], -1.0, 1.0))
    lon_c = np.arctan2(mean[..., 1], mean[..., 0])
    a = (
        np.asarray(area, np.float64)
        .reshape(*lead, h // k, k, w // k, k)
        .sum(axis=(-3, -1))
    )
    return colat, lon_c, a


class Replay(torch.nn.Module):
    """Large-scale increments of a distributed (global or regional) STRATA run.

    ``lat_face``, ``lon_face`` (degrees) and ``area_face`` (steradians) are the
    ``(6, n, n)`` grid. Driver states are added with ``add_target`` as this rank's
    interior tiles; ``increment`` returns the forcing added after each step of the
    window centred on an analysis. With ``omega_lmax > 0`` (the run's omega high-pass)
    the omega increment keeps no degree the high-pass removes. ``normalize_edge``
    (regional runs only) divides the low-passed increment by the low-passed window
    indicator, computed once at the first analysis.
    """

    def __init__(
        self,
        *,
        in_vars: list[str],
        replay_vars: list[str],
        lat_face: np.ndarray,
        lon_face: np.ndarray,
        area_face: np.ndarray,
        topology: TileTopology,
        ell_half: float,
        window_steps: int,
        omega_lmax: int = 0,
        normalize_edge: bool = False,
        block: int = BLOCK,
    ) -> None:
        super().__init__()
        if normalize_edge and not topology.is_regional:
            raise ValueError("normalize_edge needs a regional topology (a window edge)")
        if topology.tile_size % block:
            raise ValueError(
                f"tile size {topology.tile_size} is not a multiple of {block}"
            )
        self.block = block
        self.channels = replay_channels(in_vars, replay_vars)
        self.names = [in_vars[ci] for ci in self.channels]
        self.n_in = len(in_vars)
        self.ell_half = float(ell_half)
        self.lmax = gaussian_lmax(ell_half)
        self.window_steps = int(window_steps)
        self.targets: dict[int, torch.Tensor] = {}

        def local(a: np.ndarray) -> torch.Tensor:  # (6, n, n) -> (tiles, ts, ts)
            return topology.faces_to_local_tiles(
                torch.from_numpy(np.ascontiguousarray(a))
            )

        lat_l = local(lat_face).double().numpy()
        lon_l = local(lon_face).double().numpy()
        area_l = local(area_face).double().numpy()
        self.tile_shape = lat_l.shape
        self.register_buffer(
            "colat_fine",
            torch.from_numpy(np.deg2rad(90.0 - lat_l)).reshape(-1),
            persistent=False,
        )
        self.register_buffer(
            "lon_fine",
            torch.from_numpy(np.deg2rad(lon_l)).reshape(-1),
            persistent=False,
        )
        colat_c, lon_c, area_c = block_cells(lat_l, lon_l, area_l, block)
        for name, a in (
            ("colat_block", colat_c),
            ("lon_block", lon_c),
            ("area_block", area_c),
        ):
            self.register_buffer(
                name, torch.from_numpy(a).reshape(-1), persistent=False
            )
        self.register_buffer(
            "channel_index",
            torch.tensor(self.channels, dtype=torch.long),
            persistent=False,
        )
        gain = gaussian_gain(self.lmax, self.ell_half).repeat(len(self.channels), 1)
        for row, name in enumerate(self.names):
            if omega_lmax > 0 and name.rpartition("_")[0] == "omega":
                gain[row, : (omega_lmax + 1) ** 2] = 0.0
        self.register_buffer("gain", gain, persistent=False)
        self.normalize_edge = normalize_edge
        # K*(m) on the blocks and on the fine pixels, set by the first analysis
        self.register_buffer("edge_block", None, persistent=False)
        self.register_buffer("edge_fine", None, persistent=False)

    def _blocks(self, tiles: torch.Tensor) -> torch.Tensor:
        """``(C, tiles, ts, ts)`` -> ``(C, blocks)`` block means, float64."""
        return block_mean(tiles.double(), self.block).reshape(tiles.shape[0], -1)

    def add_target(self, hour: int, local_tiles: torch.Tensor) -> None:
        """Driver state at ``+hour`` as this rank's ``(1, C_in, tiles, ts, ts)`` tiles."""
        sel = local_tiles[0, self.channel_index.to(local_tiles.device)]
        blocks = self._blocks(sel).to(self.gain.device)
        if not torch.isfinite(blocks).all():
            raise ValueError(f"replay driver at +{hour} h has non-finite values")
        self.targets[hour] = blocks

    def _all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(t)
        return t

    def _edge(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The low-passed window indicator ``K*(m)`` on the blocks and the fine pixels.
        Its coefficients depend on the grid only: computed at the first call and kept.
        Collective."""
        if self.edge_fine is None:
            ones = torch.ones_like(self.area_block)[None]
            coeffs = sh_analysis(
                ones, self.colat_block, self.lon_block, self.area_block, self.lmax
            )
            coeffs = self._all_reduce(coeffs) * gaussian_gain(
                self.lmax, self.ell_half, device=coeffs.device
            )
            self.edge_block = sh_synthesis(
                coeffs, self.colat_block, self.lon_block, self.lmax
            )[0]
            self.edge_fine = sh_synthesis(
                coeffs, self.colat_fine, self.lon_fine, self.lmax
            )[0]
        return self.edge_block, self.edge_fine

    @torch.no_grad()
    def increment(self, x: torch.Tensor, hour: int) -> tuple[torch.Tensor, dict]:
        """Forcing ``(1, C_in, tiles, ts, ts)`` added after each step of the window centred
        on ``+hour``, and ``{channel: (rms of the block increment, rms low-passed)}``.
        Collective: every rank calls it."""
        delta = self.targets[hour] - self._blocks(x[0, self.channel_index])
        coeffs = sh_analysis(
            delta, self.colat_block, self.lon_block, self.area_block, self.lmax
        )
        coeffs = self._all_reduce(coeffs) * self.gain
        low_blocks = sh_synthesis(coeffs, self.colat_block, self.lon_block, self.lmax)
        low = sh_synthesis(coeffs, self.colat_fine, self.lon_fine, self.lmax)
        if self.normalize_edge:
            edge_block, edge_fine = self._edge()
            low_blocks = low_blocks / edge_block
            low = low / edge_fine
        w = self.area_block
        sums = torch.stack(
            [
                (delta**2 * w).sum(1),
                (low_blocks.double() ** 2 * w).sum(1),
                w.sum().expand(delta.shape[0]),
            ]
        )
        sums = self._all_reduce(sums)
        stats = {
            name: (float(torch.sqrt(raw / area)), float(torch.sqrt(low / area)))
            for name, raw, low, area in zip(self.names, sums[0], sums[1], sums[2])
        }
        forcing = torch.zeros(
            1, self.n_in, *self.tile_shape, dtype=x.dtype, device=x.device
        )
        forcing[0, self.channel_index] = (
            low.reshape(len(self.channels), *self.tile_shape).to(x.dtype)
            / self.window_steps
        )
        return forcing, stats


Stepper = Callable[[object, int, torch.Tensor | None, bool], object]


def run_with_replay(
    state,
    plan: list[ReplayStep],
    advance_hour: Stepper,
    increment: Callable[[object, int], torch.Tensor],
    copy: Callable[[object], object],
):
    """Execute ``plan`` from ``state`` (the run's state at hour 0, any object).

    ``advance_hour(state, hour, forcing, write)`` runs the steps from ``hour - 1`` to
    ``hour`` (adding ``forcing`` after each step, writing output when ``write``) and
    returns the new state; ``increment(state, hour)`` is the per-step forcing of the
    analysis at ``hour``; ``copy(state)`` snapshots a state. Returns the final state.
    """
    snapshots: dict[int, object] = {}
    forcing_of: dict[int, torch.Tensor] = {}  # windows never overlap: one at a time
    for step in plan:
        if step.kind == "window":
            forcing = None if step.forced_by is None else forcing_of[step.forced_by]
            state = advance_hour(state, step.hour, forcing, step.main)
        elif step.kind == "snapshot":
            snapshots[step.hour] = copy(state)
        elif step.kind == "analysis":
            forcing_of.clear()
            forcing_of[step.hour] = increment(state, step.hour)
        elif step.kind == "restore":
            state = snapshots.pop(step.hour)
        else:
            raise ValueError(f"unknown replay step {step.kind!r}")
    return state
