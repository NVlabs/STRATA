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
import earth2grid
import numpy as np
import torch
from earth2grid import KNNS2Interpolator


def normalize_latlon_grid(target_lat, target_lon):
    target_lat = np.asarray(target_lat, dtype=np.float32)
    target_lon = np.asarray(target_lon, dtype=np.float32)
    if target_lat.ndim == 1 and target_lon.ndim == 1:
        lon_2d, lat_2d = np.meshgrid(target_lon, target_lat, indexing="xy")
        return target_lat, target_lon, lat_2d.ravel(), lon_2d.ravel(), lat_2d.shape
    if target_lat.shape != target_lon.shape:
        raise ValueError(
            f"Expected lat/lon with matching shapes, got {target_lat.shape} vs {target_lon.shape}"
        )
    if target_lat.ndim != 2:
        raise ValueError(
            f"Expected 1D or 2D lat/lon arrays, got {target_lat.ndim}D inputs"
        )
    target_lat_1d = target_lat[:, 0]
    target_lon_1d = target_lon[0, :]
    lon_2d, lat_2d = np.meshgrid(target_lon_1d, target_lat_1d, indexing="xy")
    if not (np.allclose(target_lat, lat_2d) and np.allclose(target_lon, lon_2d)):
        raise ValueError("Expected a regular tensor-product lat/lon grid")
    return (
        target_lat_1d,
        target_lon_1d,
        target_lat.ravel(),
        target_lon.ravel(),
        target_lat.shape,
    )


class UnstructuredToLatLonRegridder(torch.nn.Module):
    def __init__(
        self,
        source_lon,
        source_lat,
        target_lat,
        target_lon,
        *,
        source_hpx_level: int = 10,
        target_hpx_level: int = 6,
    ):
        """Build a regridder from unstructured source points to a lat/lon grid.

        Args:
            source_lon: Source longitudes with shape `(n_source,)` or any
                source-grid shape.
            source_lat: Source latitudes with the same shape as `source_lon`.
            target_lat: Target latitudes as either shape `(n_lat,)` or `(n_lat, n_lon)`.
            target_lon: Target longitudes as either shape `(n_lon,)` or `(n_lat, n_lon)`.
            source_hpx_level: Healpix level used for the nearest-neighbor projection
                from source points onto an intermediate healpix grid.
            target_hpx_level: Healpix level used before bilinear interpolation to the
                target lat/lon grid.

        The target grid must be a regular tensor-product lat/lon grid. When
        `target_lat` and `target_lon` are 2D, they must have identical shape and
        match `np.meshgrid(target_lon[0, :], target_lat[:, 0], indexing="xy")`.
        Inputs passed to `forward` must either already be flattened with shape
        `(..., n_source)` or end in the same source-grid shape used here.
        """
        source_lon = np.asarray(source_lon, dtype=np.float32)
        source_lat = np.asarray(source_lat, dtype=np.float32)
        _, _, target_lat_flat, target_lon_flat, target_shape = normalize_latlon_grid(
            target_lat, target_lon
        )
        if source_lon.shape != source_lat.shape:
            raise ValueError(
                f"Expected matching source lon/lat shapes, got {source_lon.shape} vs {source_lat.shape}"
            )
        if source_hpx_level < target_hpx_level:
            raise ValueError(
                f"source_hpx_level must be >= target_hpx_level, got {source_hpx_level} < {target_hpx_level}"
            )
        super().__init__()
        self._source_shape = tuple(source_lon.shape)
        self._source_size = int(np.prod(self._source_shape))
        self._target_shape = tuple(target_shape)
        self._source_hpx_level = source_hpx_level
        self._target_hpx_level = target_hpx_level
        self.source_hpx_grid = earth2grid.healpix.Grid(
            level=source_hpx_level, pixel_order=earth2grid.healpix.PixelOrder.NEST
        )
        self.target_hpx_grid = earth2grid.healpix.Grid(
            level=target_hpx_level, pixel_order=earth2grid.healpix.PixelOrder.NEST
        )
        self.source_to_hpx = KNNS2Interpolator(
            torch.as_tensor(source_lon.ravel()),
            torch.as_tensor(source_lat.ravel()),
            torch.as_tensor(self.source_hpx_grid.lon).float(),
            torch.as_tensor(self.source_hpx_grid.lat).float(),
            k=1,
        )
        if hasattr(self.target_hpx_grid, "bilinear_regridder_to"):
            self.hpx_to_target = self.target_hpx_grid.bilinear_regridder_to(
                target_lat_flat, target_lon_flat
            )
        elif hasattr(self.target_hpx_grid, "get_bilinear_regridder_to"):
            self.hpx_to_target = self.target_hpx_grid.get_bilinear_regridder_to(
                target_lat_flat, target_lon_flat
            )
        else:
            raise AttributeError(
                f"No bilinear regridder method available on {type(self.target_hpx_grid).__name__}"
            )
        if hasattr(self.hpx_to_target, "float"):
            self.hpx_to_target = self.hpx_to_target.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == self._source_size:
            x_flat = x
        elif x.shape[-len(self._source_shape) :] == self._source_shape:
            x_flat = x.reshape(*x.shape[: -len(self._source_shape)], self._source_size)
        else:
            raise ValueError(
                f"Expected input with trailing shape {self._source_shape} "
                f"or flattened trailing dimension {self._source_size}, got {tuple(x.shape)}"
            )
        y_source_hpx = self.source_to_hpx(x_flat)
        y_target_hpx = y_source_hpx.reshape(
            *y_source_hpx.shape[:-1],
            -1,
            4 ** (self._source_hpx_level - self._target_hpx_level),
        ).mean(-1)
        y_target = self.hpx_to_target(y_target_hpx)
        return y_target.reshape(*y_target.shape[:-1], *self._target_shape)


class LatLonToPointGridRegridder(torch.nn.Module):
    def __init__(self, target_lon, target_lat, source_lat, source_lon):
        """Build a regridder from a regular lat/lon grid to point-grid targets.

        Args:
            target_lon: Target longitudes with any shape.
            target_lat: Target latitudes with the same shape as `target_lon`.
            source_lat: Source latitudes with shape `(n_lat,)` or `(n_lat, n_lon)`.
            source_lon: Source longitudes with shape `(n_lon,)` or `(n_lat, n_lon)`.

        The source grid must be a regular tensor-product lat/lon grid. Inputs
        passed to `forward` must either have shape `(..., n_lat, n_lon)` or
        flattened shape `(..., n_lat * n_lon)`.
        """
        source_lat_1d, source_lon_1d, _, _, source_shape = normalize_latlon_grid(
            source_lat, source_lon
        )
        target_lon = np.asarray(target_lon, dtype=np.float32)
        target_lat = np.asarray(target_lat, dtype=np.float32)

        # Work around earth2grid rejecting exact upper-boundary queries during
        # bilinear interpolation: https://github.com/NVlabs/earth2grid/issues/66
        source_lat_max = np.nextafter(
            np.float32(np.max(source_lat_1d)), np.float32(-np.inf)
        )
        target_lat = np.clip(target_lat, np.min(source_lat_1d), source_lat_max)

        if target_lon.shape != target_lat.shape:
            raise ValueError(
                f"Expected matching target lon/lat shapes, got {target_lon.shape} vs {target_lat.shape}"
            )
        super().__init__()
        self._source_shape = tuple(source_shape)
        self._source_size = int(np.prod(self._source_shape))
        self._target_shape = tuple(target_lat.shape)
        self.source_grid = earth2grid.latlon.LatLonGrid(
            lat=list(np.asarray(source_lat_1d, dtype=np.float64)),
            lon=list(np.asarray(source_lon_1d, dtype=np.float64)),
        )
        if hasattr(self.source_grid, "bilinear_regridder_to"):
            self.source_to_target = self.source_grid.bilinear_regridder_to(
                target_lat.ravel(), target_lon.ravel()
            )
        elif hasattr(self.source_grid, "get_bilinear_regridder_to"):
            self.source_to_target = self.source_grid.get_bilinear_regridder_to(
                target_lat.ravel(), target_lon.ravel()
            )
        else:
            raise AttributeError(
                f"No bilinear regridder method available on {type(self.source_grid).__name__}"
            )
        if hasattr(self.source_to_target, "float"):
            self.source_to_target = self.source_to_target.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == self._source_shape:
            x_source = x
        elif x.shape[-1] == self._source_size:
            x_source = x.reshape(*x.shape[:-1], *self._source_shape)
        else:
            raise ValueError(
                f"Expected input with trailing shape {self._source_shape} "
                f"or flattened trailing dimension {self._source_size}, got {tuple(x.shape)}"
            )
        y_target = self.source_to_target(x_source)
        return y_target.reshape(*y_target.shape[:-1], *self._target_shape)
