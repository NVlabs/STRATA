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
"""Tests for the distributed SHT high-pass filter.

Runs in two modes from the same file:

- Plain ``pytest``: the fixture monkey-patches ``torch.distributed.all_reduce``
  to a no-op and pretends to be ``(rank=0, world_size=1)``. Exercises math
  correctness on CPU without any backend setup.
- Under ``torchrun --nproc_per_node N`` (e.g. via ``make test-distributed``):
  the fixture uses the real process group. Exercises sharded all-reduce
  correctness across N ranks. Prefers NCCL when there is one GPU per rank;
  falls back to Gloo on CPU so the tests work on dev hosts with fewer GPUs.

The grid is a Gauss-Legendre x equi-angular lat-lon grid with proper quadrature
weights, reshaped to a cubesphere-like ``[6, face_size, face_size]`` layout so
it can be sliced with ``TileTopology``. This guarantees that the SHT up to
``lmax`` is exact on the discretization, so invariants like "the projection
kills Y_{ell<lmax}" hold to float32 round-off.

The tests deliberately avoid a reference implementation that reproduces the
operator's own math — instead we assert defining invariants that pin down a
high-pass projection onto span{Y_{ell,m} : ell < lmax}:

- Constants (which are Y_{0,0}) are annihilated.
- Any Y_{ell,m} with ell < lmax is annihilated.
- A mode outside the kernel (here Y_{lmax,0}) is preserved.
- The operator is idempotent (defining property of any projection).
- The operator is linear.

These invariants also exercise the sharded all-reduce end-to-end: when the
reduction is broken across ranks, the "annihilates" tests stop reporting zero.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import scipy.special
import torch
import torch.distributed as dist

from screamcast.distributed_halo import TileTopology
from screamcast.sht_omega_filter import DistributedSHTHighpass

FACE_SIZE = 8
TILE_SIZE = 2
LMAX = 3
NSIDE_PER_FACE = FACE_SIZE * FACE_SIZE  # 64 points per face
N_TOTAL = 6 * NSIDE_PER_FACE  # 384


@pytest.fixture(scope="session")
def _session_dist() -> tuple[int, int, torch.device]:
    """Session-scoped distributed setup, shared with the ``dist_ctx`` fixture.

    Handles three scenarios:
    - Another session fixture (e.g. ``distributed_cuda_context``) already
      initialized a process group: reuse it as-is.
    - Launched under ``torchrun`` with no group yet: initialize one ourselves.
      Prefer NCCL (one GPU per rank); fall back to Gloo on CPU when there are
      fewer GPUs than ranks.
    - Plain pytest: no group; the per-test ``dist_ctx`` fixture will mock
      ``all_reduce`` to a no-op.
    """
    ws_env = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        ws = dist.get_world_size()
        if torch.cuda.is_available() and local_rank < torch.cuda.device_count():
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        yield rank, ws, device
        return

    if ws_env > 1 and "MASTER_ADDR" in os.environ:
        device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        use_nccl = torch.cuda.is_available() and device_count >= ws_env
        if use_nccl:
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
            dist.init_process_group(backend="nccl", init_method="env://")
        else:
            device = torch.device("cpu")
            dist.init_process_group(backend="gloo", init_method="env://")
        try:
            yield dist.get_rank(), dist.get_world_size(), device
        finally:
            if dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    yield 0, 1, device


@pytest.fixture
def dist_ctx(
    monkeypatch, _session_dist: tuple[int, int, torch.device]
) -> tuple[int, int, torch.device]:
    """Return ``(rank, world_size, device)``.

    Under a real distributed session the group is live. Otherwise
    ``torch.distributed.all_reduce`` is monkey-patched to a no-op so the same
    tests work with plain ``pytest``.
    """
    rank, ws, device = _session_dist
    if not (dist.is_available() and dist.is_initialized()):
        monkeypatch.setattr(dist, "all_reduce", lambda *a, **kw: None)
    return rank, ws, device


def _build_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(lat_face, lon_face, area_face)`` on a Gauss-Legendre x equiangular grid.

    Layout ``[6, face_size, face_size]``. Chosen so that SHT up to ``LMAX`` is
    exact on this discretization: the colat axis uses Gauss-Legendre weights
    (exact for polynomials of degree < 2*nlat), and the lon axis is equiangular
    (exact for Fourier modes with ``|m| < nlon/2``).
    """
    nlat = 24  # exact for ell < 24; far more than LMAX
    nlon = 16  # exact for |m| < 8; far more than LMAX
    assert nlat * nlon == N_TOTAL

    # Gauss-Legendre nodes in cos(colat) and their weights on [-1, 1].
    mu, w = scipy.special.roots_legendre(nlat)
    colat = np.arccos(mu)  # in (0, pi)
    lat = np.pi / 2.0 - colat  # radians
    lon = 2.0 * np.pi * np.arange(nlon) / nlon  # in [0, 2pi)

    lat_grid = np.broadcast_to(lat[:, None], (nlat, nlon))  # [nlat, nlon]
    lon_grid = np.broadcast_to(lon[None, :], (nlat, nlon))
    # Area: each (lat, lon) cell is w_j * (2*pi / nlon). Sum = 2 * 2pi = 4pi.
    area_grid = (w[:, None] * (2.0 * np.pi / nlon)).astype(np.float32)
    area_grid = np.broadcast_to(area_grid, (nlat, nlon)).copy()

    shape = (6, FACE_SIZE, FACE_SIZE)
    lat_face = np.rad2deg(lat_grid).astype(np.float32).reshape(shape)
    lon_face = np.rad2deg(lon_grid).astype(np.float32).reshape(shape)
    area_face = area_grid.reshape(shape)
    return lat_face, lon_face, area_face


def _make_topology(rank: int, world_size: int) -> TileTopology:
    return TileTopology(
        world_size=world_size,
        rank=rank,
        face_size=FACE_SIZE,
        tile_size=TILE_SIZE,
        halo_width=0,
    )


def _make_op(topology: TileTopology, device: torch.device) -> DistributedSHTHighpass:
    lat_face, lon_face, area_face = _build_grid()
    lat_local = topology.faces_to_local_tiles(torch.from_numpy(lat_face)).numpy()
    lon_local = topology.faces_to_local_tiles(torch.from_numpy(lon_face)).numpy()
    area_local = topology.faces_to_local_tiles(torch.from_numpy(area_face)).numpy()
    return DistributedSHTHighpass(lat_local, lon_local, area_local, LMAX).to(device)


def test_basis_shape(dist_ctx):
    rank, ws, device = dist_ctx
    topology = _make_topology(rank, ws)
    op = _make_op(topology, device)

    n_expected = topology.tiles_per_rank * TILE_SIZE * TILE_SIZE
    assert op.Y.shape == (LMAX * LMAX, n_expected)
    assert op.Y.dtype == torch.complex64
    assert op.area.shape == (n_expected,)
    assert op.n_local == n_expected
    assert op.lmax == LMAX


def _harmonic_on_grid(
    lat_face_deg: np.ndarray, lon_face_deg: np.ndarray, ell: int, m: int
) -> torch.Tensor:
    """Evaluate Re(Y_{ell, m}) pointwise on the face-major grid.

    Uses scipy directly rather than the operator's internal basis construction
    so that this helper is genuinely independent of the implementation under
    test.
    """
    colat = np.pi / 2.0 - np.deg2rad(lat_face_deg)
    phi = np.deg2rad(lon_face_deg)
    y = scipy.special.sph_harm_y(ell, m, colat, phi)
    return torch.from_numpy(np.real(y).astype(np.float32))


def _random_field(
    shape: tuple, rank: int, seed: int, device: torch.device
) -> torch.Tensor:
    """Reproducible random field, broadcast from rank 0 when ws > 1."""
    x = torch.empty(shape, device=device, dtype=torch.float32)
    if rank == 0:
        rng = np.random.default_rng(seed=seed)
        x.copy_(torch.from_numpy(rng.standard_normal(shape).astype(np.float32)))
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.broadcast(x, src=0)
    return x


def test_removes_constant_field(dist_ctx):
    """Y_{0,0} is a constant; the high-pass must annihilate it."""
    rank, ws, device = dist_ctx
    topology = _make_topology(rank, ws)
    op = _make_op(topology, device)

    x = torch.ones(
        (1, 1, topology.tiles_per_rank, TILE_SIZE, TILE_SIZE),
        device=device,
        dtype=torch.float32,
    )
    y = op(x)
    torch.testing.assert_close(y, torch.zeros_like(y), atol=1e-5, rtol=0)


@pytest.mark.parametrize(
    "ell,m",
    [(ell, m) for ell in range(LMAX) for m in range(-ell, ell + 1)],
)
def test_removes_low_order_harmonic(dist_ctx, ell, m):
    """Every Y_{ell, m} with ell < lmax lies in the kernel of the high-pass."""
    rank, ws, device = dist_ctx
    topology = _make_topology(rank, ws)
    op = _make_op(topology, device)

    lat_face, lon_face, _ = _build_grid()
    harmonic_full = _harmonic_on_grid(lat_face, lon_face, ell, m).to(device)
    x = topology.faces_to_local_tiles(harmonic_full).unsqueeze(0).unsqueeze(0)

    y = op(x)
    torch.testing.assert_close(y, torch.zeros_like(y), atol=1e-5, rtol=0)


def test_preserves_high_order_harmonic(dist_ctx):
    """A harmonic with ell >= lmax is orthogonal to the kernel, so op preserves it."""
    rank, ws, device = dist_ctx
    topology = _make_topology(rank, ws)
    op = _make_op(topology, device)

    lat_face, lon_face, _ = _build_grid()
    harmonic_full = _harmonic_on_grid(lat_face, lon_face, LMAX, 0).to(device)
    x = topology.faces_to_local_tiles(harmonic_full).unsqueeze(0).unsqueeze(0)

    y = op(x)
    torch.testing.assert_close(y, x, atol=1e-5, rtol=1e-5)


def test_idempotent(dist_ctx):
    """A projection satisfies op(op(x)) == op(x) for any input."""
    rank, ws, device = dist_ctx
    topology = _make_topology(rank, ws)
    op = _make_op(topology, device)

    x_full = _random_field(
        (1, 1, 6, FACE_SIZE, FACE_SIZE), rank=rank, seed=1, device=device
    )
    x = topology.faces_to_local_tiles(x_full)
    y = op(x)
    y2 = op(y.clone())
    torch.testing.assert_close(y2, y, atol=1e-5, rtol=1e-5)


def test_linearity(dist_ctx):
    """op(a*x1 + b*x2) == a*op(x1) + b*op(x2)."""
    rank, ws, device = dist_ctx
    topology = _make_topology(rank, ws)
    op = _make_op(topology, device)

    shape = (1, 1, 6, FACE_SIZE, FACE_SIZE)
    x1_full = _random_field(shape, rank=rank, seed=2, device=device)
    x2_full = _random_field(shape, rank=rank, seed=3, device=device)
    a, b = 1.7, -0.3

    x1 = topology.faces_to_local_tiles(x1_full)
    x2 = topology.faces_to_local_tiles(x2_full)

    y_combined = op((a * x1 + b * x2).clone())
    y_separate = a * op(x1.clone()) + b * op(x2.clone())
    torch.testing.assert_close(y_combined, y_separate, atol=1e-5, rtol=1e-5)
