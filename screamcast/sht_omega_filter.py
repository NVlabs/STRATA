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
"""Distributed spherical-harmonic high-pass filter.

Adapted from the global cubesphere rollout pipeline.
The eval-script implementation applied the filter on rank 0 to the global
unstructured field after a stitch step. For the distributed tiled rollout in
``scripts/ace/run_screamcast_nudged.py`` we avoid gathering the full field and
instead all-reduce the (tiny) spherical-harmonic coefficients:

    c_local  = (a_local * f_local) @ Y_local^H    # [B, lmax**2] on each rank
    c_global = all_reduce_sum(c_local)            # identical on every rank
    lowpass  = Re(c_global @ Y_local)             # [B, N_local] on each rank
    highpass = f_local - lowpass

Relative to the eval script, this uses the proper quadrature-weighted analysis
``c = sum_j a_j f_j Y_{ell,m}^*(j)`` and unweighted synthesis
``f_lp = sum_{ell,m} c Y_{ell,m}`` instead of the eval script's symmetric
``W = sqrt(a) Y`` form. The two are equivalent only when the grid is uniform
(area weights constant); otherwise only the quadrature-weighted form is a
genuine orthogonal projection in the ``L^2(dΩ)`` inner product and can exactly
remove low-order modes like the constant.

Correctness of the all-reduce does not depend on how points are partitioned
across ranks as long as every global grid point appears on exactly one rank
and its lat/lon/area triple match.
"""
from __future__ import annotations

import numpy as np
import scipy.special
import torch


def _build_sht_basis_local(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    lmax: int,
) -> np.ndarray:
    """Return complex64 spherical-harmonic basis matrix.

    Shape ``[lmax**2, lat_deg.size]``. Rows are ``(ell, m)`` in the canonical
    order ``ell=0..lmax-1, m=-ell..ell``. Columns are the input points in
    C-order flatten. No area weighting — the caller is expected to handle
    quadrature weights during the forward transform. Module-private; callers
    should use :class:`DistributedSHTHighpass` rather than invoking this
    directly.
    """
    if lmax <= 0:
        raise ValueError(f"lmax must be > 0, got {lmax}")
    if lat_deg.shape != lon_deg.shape:
        raise ValueError(
            f"lat_deg and lon_deg must share the same shape; got "
            f"{lat_deg.shape} and {lon_deg.shape}"
        )
    phi = np.deg2rad(lon_deg).ravel()
    colat = np.pi / 2.0 - np.deg2rad(lat_deg).ravel()
    bands = [
        scipy.special.sph_harm_y(ell, m, colat, phi)
        for ell in range(lmax)
        for m in range(-ell, ell + 1)
    ]
    Y = np.stack(bands, axis=0)
    return Y.astype(np.complex64)


class DistributedSHTHighpass(torch.nn.Module):
    """Remove spherical-harmonic modes ``ell < lmax`` via an all-reduce on coefficients.

    Construction takes the local grid directly; the basis matrix and area
    weights are registered as non-persistent buffers so standard ``nn.Module``
    semantics move them with ``.to(device)`` / ``.cuda()`` etc. The same
    convention is used by ``DistributedTileKNNHaloPadding_AllGather`` in this
    codebase.

    ``forward`` is layout- and channel-agnostic: the input ``x`` may have any
    shape whose total element count is a multiple of
    ``N_local = lat_deg.size``. The trailing ``N_local`` elements (in C-order)
    are treated as the spatial axis; everything above is flattened into an
    effective batch axis and the output shape equals the input shape. The
    high-pass is applied to every channel of ``x``; the caller is responsible
    for selecting which channels (e.g. omega) to filter.
    """

    def __init__(
        self,
        lat_deg: np.ndarray,
        lon_deg: np.ndarray,
        area: np.ndarray,
        lmax: int,
    ) -> None:
        super().__init__()
        if area.shape != lat_deg.shape:
            raise ValueError(
                "area must share the shape of lat_deg; got "
                f"{area.shape} vs {lat_deg.shape}"
            )
        Y_np = _build_sht_basis_local(lat_deg, lon_deg, lmax)
        self.register_buffer("Y", torch.from_numpy(Y_np), persistent=False)
        self.register_buffer(
            "area",
            torch.from_numpy(area.ravel().astype(np.float32)),
            persistent=False,
        )
        self.n_local = int(self.Y.shape[1])
        self.lmax = int(lmax)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() % self.n_local != 0:
            raise ValueError(
                f"Input has {x.numel()} elements; not divisible by "
                f"N_local={self.n_local}."
            )
        batch = x.numel() // self.n_local
        flat = x.reshape(batch, self.n_local)
        weighted = (flat * self.area).to(torch.complex64)
        c_local = weighted @ self.Y.conj().transpose(0, 1)
        torch.distributed.all_reduce(c_local, op=torch.distributed.ReduceOp.SUM)
        lowpass = torch.real(c_local @ self.Y).to(x.dtype)
        return x - lowpass.reshape(x.shape)
