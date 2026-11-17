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
import numpy as np
import pytest
import scipy.special
import torch

from screamcast.sht_lowpass import (
    gaussian_gain,
    gaussian_lmax,
    mode_degree,
    n_modes,
    real_sh_basis,
    sh_analysis,
    sh_synthesis,
)


def gauss_grid(lmax: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gauss-Legendre x uniform-longitude grid: exact quadrature up to degree 2*lmax."""
    nlat, nlon = lmax + 1, 2 * lmax + 2
    x, w = np.polynomial.legendre.leggauss(nlat)
    colat = np.repeat(np.arccos(x), nlon)
    lon = np.tile(2.0 * np.pi * np.arange(nlon) / nlon, nlat)
    weight = np.repeat(w, nlon) * (2.0 * np.pi / nlon)
    return torch.from_numpy(colat), torch.from_numpy(lon), torch.from_numpy(weight)


def scipy_real_harmonic(
    ell: int, m: int, colat: np.ndarray, lon: np.ndarray
) -> np.ndarray:
    """Real harmonic from scipy's complex one; scipy carries the Condon-Shortley phase
    (-1)^m, the module's basis does not."""
    y = scipy.special.sph_harm_y(ell, abs(m), colat, lon)
    if m == 0:
        return y.real
    return (-1) ** m * np.sqrt(2.0) * (y.real if m > 0 else y.imag)


def test_basis_matches_scipy_spherical_harmonics():
    rng = np.random.default_rng(0)
    colat = rng.uniform(0.0, np.pi, 64)
    lon = rng.uniform(-np.pi, np.pi, 64)
    lmax = 7
    basis = real_sh_basis(torch.from_numpy(colat), torch.from_numpy(lon), lmax).numpy()
    for ell in range(lmax + 1):
        for m in range(-ell, ell + 1):
            np.testing.assert_allclose(
                basis[:, ell * ell + ell + m],
                scipy_real_harmonic(ell, m, colat, lon),
                atol=2e-6,
                err_msg=f"degree {ell} order {m}",
            )


def test_basis_is_orthonormal_under_exact_quadrature():
    lmax = 12
    colat, lon, weight = gauss_grid(lmax)
    basis = real_sh_basis(colat, lon, lmax).double()
    gram = basis.T @ (basis * weight[:, None])
    np.testing.assert_allclose(gram.numpy(), np.eye(n_modes(lmax)), atol=1e-5)


def test_analysis_inverts_synthesis_in_any_chunking():
    lmax = 10
    colat, lon, weight = gauss_grid(lmax)
    gen = torch.Generator().manual_seed(1)
    coeffs = torch.randn(3, n_modes(lmax), dtype=torch.float64, generator=gen)
    field = sh_synthesis(coeffs, colat, lon, lmax)
    whole = sh_analysis(field, colat, lon, weight, lmax)
    chunked = sh_analysis(
        field, colat, lon, weight, lmax, max_basis_elems=300 * n_modes(lmax)
    )
    np.testing.assert_allclose(whole.numpy(), coeffs.numpy(), atol=2e-5)
    np.testing.assert_allclose(chunked.numpy(), whole.numpy(), atol=1e-9)


def test_lowpass_scales_each_harmonic_by_its_gain():
    """The taper is diagonal in degree: a pure degree-l field comes back times
    gain(l), and a constant is untouched."""
    lmax = 16
    ell_half = 5.0
    colat, lon, weight = gauss_grid(lmax)
    gain = gaussian_gain(lmax, ell_half)
    for ell, m in ((0, 0), (3, -2), (9, 4), (16, 0)):
        field = torch.from_numpy(
            scipy_real_harmonic(ell, m, colat.numpy(), lon.numpy())
        )
        coeffs = sh_analysis(field[None], colat, lon, weight, lmax) * gain
        out = sh_synthesis(coeffs, colat, lon, lmax).double()[0]
        np.testing.assert_allclose(
            out.numpy(), float(gain[ell * ell]) * field.numpy(), atol=2e-5
        )


def test_gaussian_gain_is_one_at_zero_and_half_at_ell_half():
    lmax = gaussian_lmax(40.0)
    gain = gaussian_gain(lmax, 40.0)
    assert float(gain[0]) == 1.0
    np.testing.assert_allclose(gain[mode_degree(lmax) == 40].numpy(), 0.5, rtol=1e-12)


@pytest.mark.parametrize("ell_half", [20.0, 40.0, 80.0])
def test_gaussian_lmax_is_the_first_degree_below_floor(ell_half):
    lmax = gaussian_lmax(ell_half, floor=0.01)
    gain = gaussian_gain(lmax, ell_half)
    degree = mode_degree(lmax)
    assert float(gain[degree == lmax][0]) < 0.01
    assert float(gain[degree == lmax - 1][0]) >= 0.01
