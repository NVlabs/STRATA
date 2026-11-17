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
"""Spherical-harmonic low-pass on scattered points, for large-scale increments.

Real orthonormal spherical harmonics up to degree ``lmax`` are evaluated on the fly in
chunks of points (float64 Legendre recursion, float32 basis), so the transforms work on
any point set -- the ne128pg2 cells of a coarse increment, or one rank's ne1024pg2
pixels -- without holding a points x modes matrix in memory.

Modes are degree-major: mode ``l*l + l + m`` for ``m = -l..l``. ``m > 0`` is
``sqrt(2) P_lm(cos colat) cos(m lon)``, ``m < 0`` is ``sqrt(2) P_l|m| sin(|m| lon)`` and
``m = 0`` is ``P_l0``, with ``P_lm`` normalised so every mode has unit L2 norm on the
unit sphere and no Condon-Shortley phase. All modes of degree ``< L`` are therefore the
first ``L**2`` entries.
"""

from __future__ import annotations

import math

import torch


def n_modes(lmax: int) -> int:
    return (lmax + 1) ** 2


def mode_degree(lmax: int, device=None) -> torch.Tensor:
    """Degree of every mode in degree-major order, shape ``[(lmax + 1)**2]``."""
    ell = torch.arange(lmax + 1, device=device)
    return torch.repeat_interleave(ell, 2 * ell + 1)


def real_sh_basis(colat: torch.Tensor, lon: torch.Tensor, lmax: int) -> torch.Tensor:
    """Orthonormal real spherical harmonics, shape ``[points, (lmax + 1)**2]`` float32.

    ``colat`` and ``lon`` are radians, shape ``[points]``.
    """
    x = torch.cos(colat.double())
    s = torch.sin(colat.double())
    n = x.shape[0]
    dev = x.device
    out = torch.empty(n, n_modes(lmax), dtype=torch.float32, device=dev)
    m = torch.arange(lmax + 1, device=dev, dtype=torch.float64)
    cos_m = torch.cos(lon.double()[:, None] * m)
    sin_m = torch.sin(lon.double()[:, None] * m)

    # sectoral P_mm: P_00 = 1/sqrt(4 pi), P_mm = sqrt((2m+1)/(2m)) sin(colat) P_m-1,m-1
    diag = torch.empty(n, lmax + 1, dtype=torch.float64, device=dev)
    diag[:, 0] = 1.0 / math.sqrt(4.0 * math.pi)
    for mm in range(1, lmax + 1):
        diag[:, mm] = math.sqrt((2 * mm + 1) / (2 * mm)) * s * diag[:, mm - 1]

    prev2 = prev = None
    for ell in range(lmax + 1):
        cur = torch.zeros(n, lmax + 1, dtype=torch.float64, device=dev)
        cur[:, ell] = diag[:, ell]
        if ell >= 1:
            cur[:, ell - 1] = math.sqrt(2 * ell + 1) * x * diag[:, ell - 1]
        if ell >= 2:
            mm = torch.arange(ell - 1, device=dev, dtype=torch.float64)
            a = torch.sqrt((4.0 * ell * ell - 1.0) / (ell * ell - mm * mm))
            b = torch.sqrt(
                ((ell - 1.0) ** 2 - mm * mm) / (4.0 * (ell - 1.0) ** 2 - 1.0)
            )
            cur[:, : ell - 1] = a * (
                x[:, None] * prev[:, : ell - 1] - b * prev2[:, : ell - 1]
            )
        base = ell * ell + ell
        out[:, base] = cur[:, 0].float()
        if ell >= 1:
            pos = torch.arange(1, ell + 1, device=dev)
            scaled = math.sqrt(2.0) * cur[:, 1 : ell + 1]
            out[:, base + pos] = (scaled * cos_m[:, 1 : ell + 1]).float()
            out[:, base - pos] = (scaled * sin_m[:, 1 : ell + 1]).float()
        prev2, prev = prev, cur
    return out


def _chunk(lmax: int, max_basis_elems: int) -> int:
    return max(256, max_basis_elems // n_modes(lmax))


def sh_analysis(
    field: torch.Tensor,
    colat: torch.Tensor,
    lon: torch.Tensor,
    weight: torch.Tensor,
    lmax: int,
    max_basis_elems: int = 1 << 27,
) -> torch.Tensor:
    """Quadrature coefficients ``sum_i weight_i field_ci Y_k(i)``, shape ``[C, modes]``.

    ``field`` is ``[C, points]``; ``weight`` holds the cell areas in steradians.
    """
    coeffs = torch.zeros(
        field.shape[0], n_modes(lmax), dtype=torch.float64, device=field.device
    )
    step = _chunk(lmax, max_basis_elems)
    for i in range(0, field.shape[1], step):
        basis = real_sh_basis(colat[i : i + step], lon[i : i + step], lmax)
        weighted = (field[:, i : i + step] * weight[i : i + step]).float()
        coeffs += (weighted @ basis).double()
    return coeffs


def sh_synthesis(
    coeffs: torch.Tensor,
    colat: torch.Tensor,
    lon: torch.Tensor,
    lmax: int,
    max_basis_elems: int = 1 << 27,
) -> torch.Tensor:
    """Evaluate ``[C, modes]`` coefficients at the points, shape ``[C, points]`` float32."""
    out = torch.empty(
        coeffs.shape[0], colat.shape[0], dtype=torch.float32, device=coeffs.device
    )
    c32 = coeffs.float()
    step = _chunk(lmax, max_basis_elems)
    for i in range(0, colat.shape[0], step):
        basis = real_sh_basis(colat[i : i + step], lon[i : i + step], lmax)
        out[:, i : i + step] = c32 @ basis.T
    return out


def gaussian_gain(lmax: int, ell_half: float, device=None) -> torch.Tensor:
    """Heat-kernel gain ``2**(-l(l+1) / (lh(lh+1)))`` per mode: 1 at l = 0, 1/2 at l = lh."""
    ell = mode_degree(lmax, device=device).double()
    return torch.exp(-math.log(2.0) * ell * (ell + 1.0) / (ell_half * (ell_half + 1.0)))


def gaussian_lmax(ell_half: float, floor: float = 0.01) -> int:
    """Smallest degree whose Gaussian gain is below ``floor`` (the truncation degree)."""
    target = ell_half * (ell_half + 1.0) * math.log2(1.0 / floor)
    return int(math.ceil((-1.0 + math.sqrt(1.0 + 4.0 * target)) / 2.0))
