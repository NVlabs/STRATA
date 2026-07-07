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
"""Spherical tile geometry: index -> lat/lon lookup and tile centers.

Isolates the earth2grid / xarray grid handling from the model classes. The
Strata architecture itself (physicsnemo) takes lat/lon (``pos``) as a forward
input; this module derives it from the SCREAM tile ``index`` for both HEALPix
and CubeSphere grids, exactly as the pre-migration ``DiT`` did internally.
"""

import math
from typing import Mapping

import earth2grid
import numpy as np
import torch
import torch.nn as nn
import xarray as xr


def mean_longitude(
    lon: torch.Tensor,
    reduce_dims=(-2, -1),
    return_0_2pi: bool = True,
) -> torch.Tensor:
    """
    Compute circular mean of longitude, batched, handling 0/2π or -π/π wrapping.

    Args:
        lon: longitude in radians ([..., H, W])
        reduce_dims: which dimensions to average over (default: last two -> H, W)
        return_0_2pi:
            - True  -> output in [0, 2π)
            - False -> output in [-π, π)

    Returns:
        lon_mean: mean longitude ([..., 1, 1])
    """
    # Circular mean: average sin and cos
    sin_mean = lon.sin().mean(dim=reduce_dims, keepdim=True)
    cos_mean = lon.cos().mean(dim=reduce_dims, keepdim=True)

    # atan2 gives angle in [-π, π)
    lon_mean = torch.atan2(sin_mean, cos_mean)

    if return_0_2pi:
        lon_mean = lon_mean % (2 * torch.pi)

    return lon_mean


class TileGeometry(nn.Module):
    """Grid geometry for one tiled spherical grid (HEALPix or CubeSphere).

    Owns the global lat/lon lookup tables (non-persistent buffers, so they
    never appear in checkpoints) and converts a tile ``index`` — either global
    column ids or explicit ``{"lat", "lon"}`` tensors — into per-pixel lat/lon
    in radians, plus the mean-based tile center used by the wind rotation.
    """

    def __init__(
        self,
        grid_type: str = "healpix",
        nside: int = 1024,
        cubesphere_latlon_path: str = "data/latlon_ne1024pg2.nc",
        index_is_latlon: bool = False,
    ):
        super().__init__()
        self.grid_type = (grid_type or "healpix").lower()
        if self.grid_type not in {"healpix", "cubesphere"}:
            raise ValueError(
                f"Unsupported grid_type='{grid_type}'. Expected 'healpix' or 'cubesphere'."
            )
        self.nside = nside
        self.index_is_latlon = index_is_latlon

        # Precompute lat/lon (radians) over global column ids so callers can do
        #   lat = self._lat_radians[index.long()]
        # for both HEALPix and CubeSphere (matches the pre-migration DiT).
        if self.grid_type == "healpix":
            input_grid = earth2grid.healpix.Grid(
                earth2grid.healpix.nside2level(self.nside),
                pixel_order=earth2grid.healpix.PixelOrder.NEST,
            )
            lat_radians = input_grid.lat * np.pi / 180.0
            lon_radians = input_grid.lon * np.pi / 180.0
            self.ncol_total = int(12 * self.nside * self.nside)
        else:
            ds_ll = xr.open_dataset(cubesphere_latlon_path)
            lat_deg = ds_ll["lat"].values
            lon_deg = ds_ll["lon"].values
            self.ncol_total = int(lat_deg.shape[0])
            lat_radians = lat_deg * (np.pi / 180.0)
            lon_radians = lon_deg * (np.pi / 180.0)
        self.register_buffer(
            "_lat_radians", torch.from_numpy(lat_radians).float(), persistent=False
        )
        self.register_buffer(
            "_lon_radians", torch.from_numpy(lon_radians).float(), persistent=False
        )

    def lat_lon_from_index(
        self, index: torch.Tensor | Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-pixel lat/lon in radians, shape ``[B, H, W]``."""
        if self.index_is_latlon:
            try:
                lat = index["lat"]
                lon = index["lon"]
            except Exception as exc:  # pragma: no cover - defensive guard
                raise ValueError(
                    "index must provide 'lat' and 'lon' tensors when index_is_latlon is True"
                ) from exc

            if lat.ndim == 2:
                lat = lat.unsqueeze(0)
            if lon.ndim == 2:
                lon = lon.unsqueeze(0)
            if lat.ndim != 3 or lon.ndim != 3:
                raise ValueError(
                    "lat/lon tensors must be 2D or 3D with shape [B, H, W]"
                )
            if lat.shape != lon.shape:
                raise ValueError("lat/lon tensors must have matching shapes")
            return lat, lon

        return self._lat_radians[index.long()], self._lon_radians[index.long()]

    @staticmethod
    def tile_center(
        lat: torch.Tensor, lon: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean-based tile center ``(lat0, lon0)``, each ``[B, 1, 1]``.

        Arithmetic mean latitude + circular mean longitude — the historical
        SCREAMCast center used by the wind rotation. (The stereographic RoPE
        inside physicsnemo Strata uses the pole/seam-robust spherical centroid
        instead; the wind rotation keeps this center so its numerics are
        unchanged by the migration.)
        """
        lat0 = lat.mean(dim=(1, 2), keepdim=True)
        lon0 = mean_longitude(lon)
        return lat0, lon0

    def rope_length_scale(
        self, patch_size_horiz: int, use_hpx_pe_scaling: bool = True
    ) -> float:
        """Coordinate normalization for the stereographic RoPE.

        Each token covers ``patch_size_horiz**2`` fine pixels; the length scale
        is the sqrt of the approximate token area so coordinates are comparable
        across tiles and grids. ``use_hpx_pe_scaling=True`` (the production
        setting) pins the reference to the HEALPix nside=1024 pixel area.
        """
        if use_hpx_pe_scaling:
            return math.sqrt(math.pi * patch_size_horiz**2 / (3.0 * 1024**2))
        return math.sqrt(4.0 * math.pi * patch_size_horiz**2 / float(self.ncol_total))
