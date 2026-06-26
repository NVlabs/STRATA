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
"""Distributed cubed-sphere halo exchange based on all-gather.

Due to the expense of creating the distributed plan,
./distributed_cubesphere_transforms.py was not feasible for the distributed
tiled rollout at full resolution, so this all gather implementation is used
instead, and seems to have minimal overead up to at least 32 GPUS.

Example:
    >>> topology = TileTopology(
    ...     world_size=world_size,
    ...     rank=rank,
    ...     face_size=2048,
    ...     tile_size=128,
    ...     halo_width=32,
    ... )
    >>> print(lon_faces_padded.shape)
    (6, 2112, 2112)
    >>> print(lat_faces_padded.shape)
    (6, 2112, 2112)
    >>> halo_exchange = DistributedTileKNNHaloPadding_AllGather.from_padded_face_grid(
    ...     topology=topology,
    ...     lon=lon_faces_padded,
    ...     lat=lat_faces_padded,
    ...     pad_width_data=pad_width_data,
    ...     device=device,
    ... )
    >>> padded_tiles = halo_exchange(local_tiles)
    >>> interior_tiles = topology.crop(padded_tiles)
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import torch
from earth2grid import KNNS2Interpolator

Tile: TypeAlias = tuple[int, int, int]


@dataclass(frozen=True)
class TileTopology:
    """Describe the distributed cubed-sphere tile decomposition."""

    world_size: int
    rank: int
    face_size: int
    tile_size: int
    halo_width: int
    n_faces: int = 6

    def __post_init__(self) -> None:
        if self.face_size % self.tile_size != 0:
            raise ValueError(
                f"face_size={self.face_size} must be divisible by tile_size={self.tile_size}"
            )
        if self.total_tiles % self.world_size != 0:
            raise ValueError(
                f"Number of tiles ({self.total_tiles}) must be divisible by "
                f"world_size ({self.world_size})"
            )

    @property
    def tiles_per_dim(self) -> int:
        return self.face_size // self.tile_size

    @property
    def total_tiles(self) -> int:
        return self.n_faces * self.tiles_per_dim * self.tiles_per_dim

    @property
    def tiles_per_rank(self) -> int:
        return self.total_tiles // self.world_size

    @property
    def padded_tile_size(self) -> int:
        return self.tile_size + 2 * self.halo_width

    @property
    def global_tiles(self) -> list[Tile]:
        return list(
            itertools.product(
                range(self.n_faces),
                range(self.tiles_per_dim),
                range(self.tiles_per_dim),
            )
        )

    @property
    def local_tiles(self) -> list[Tile]:
        start = self.rank * self.tiles_per_rank
        end = start + self.tiles_per_rank
        return self.global_tiles[start:end]

    def interior_origin(self, tile: Tile) -> tuple[int, int]:
        _, tile_i, tile_j = tile
        return tile_i * self.tile_size, tile_j * self.tile_size

    def interior_coords(
        self,
        base_coords: dict,
        *,
        time: np.ndarray | None = None,
    ) -> dict:
        coords = dict(base_coords)
        if time is not None:
            coords["time"] = time
        coords["face"] = np.arange(len(self.local_tiles))
        coords["x"] = np.arange(self.tile_size)
        coords["y"] = np.arange(self.tile_size)
        return coords

    def pad_coords(self, coords: dict) -> dict:
        padded = dict(coords)
        padded["face"] = np.arange(len(self.local_tiles))
        padded["x"] = np.arange(self.padded_tile_size)
        padded["y"] = np.arange(self.padded_tile_size)
        return padded

    def crop_coords(self, coords: dict) -> dict:
        cropped = dict(coords)
        cropped["face"] = np.arange(len(self.local_tiles))
        cropped["x"] = np.arange(self.tile_size)
        cropped["y"] = np.arange(self.tile_size)
        return cropped

    def crop(self, x: torch.Tensor) -> torch.Tensor:
        halo = self.halo_width
        if halo == 0:
            return x
        return x[..., halo:-halo, halo:-halo]

    def faces_to_local_tiles(self, x: torch.Tensor) -> torch.Tensor:
        """Slice this rank's interior tiles out of a face-major tensor.

        This is the local (no-collective) counterpart of
        :meth:`gather_tiles_to_faces`. Every rank independently indexes its own
        ``local_tiles`` out of the global face layout.

        Args:
            x: Tensor of shape ``(..., n_faces, face_size, face_size)``.

        Returns:
            Tensor of shape ``(..., tiles_per_rank, tile_size, tile_size)``
            where the new tiles axis enumerates ``self.local_tiles`` in order.
        """
        ts = self.tile_size
        tiles = [
            x[..., face, ti * ts : (ti + 1) * ts, tj * ts : (tj + 1) * ts]
            for face, ti, tj in self.local_tiles
        ]
        return torch.stack(tiles, dim=-3)

    def gather_tiles_to_faces(self, x: torch.Tensor) -> torch.Tensor:
        """All-gather local interior tiles into a global face layout.

        Args:
            x: Local interior tiles with shape ``(..., tiles_per_rank,
                tile_size, tile_size)``. Tiles on each rank are in the slice
                ``global_tiles[rank * tiles_per_rank : (rank + 1) *
                tiles_per_rank]``.

        Returns:
            Global faces with shape ``(..., n_faces, face_size, face_size)``,
            where each tile has been scattered into the position assigned by
            ``global_tiles``.
        """
        *lead, tiles_per_rank, nx, ny = x.shape
        if tiles_per_rank != self.tiles_per_rank:
            raise ValueError(
                f"Expected {self.tiles_per_rank} local tiles, got {tiles_per_rank}"
            )
        if nx != self.tile_size or ny != self.tile_size:
            raise ValueError(
                f"Expected interior tile shape ({self.tile_size}, "
                f"{self.tile_size}), got ({nx}, {ny})"
            )
        tensor_list = [torch.zeros_like(x) for _ in range(self.world_size)]
        torch.distributed.all_gather(tensor_list, x.contiguous())
        # Concatenate along the tiles axis so tile-index order matches
        # global_tiles (rank 0's tiles first, then rank 1's, ...).
        gathered = torch.cat(tensor_list, dim=-3)
        out = gathered.new_zeros(*lead, self.n_faces, self.face_size, self.face_size)
        ts = self.tile_size
        for global_index, (face, ti, tj) in enumerate(self.global_tiles):
            out[..., face, ti * ts : (ti + 1) * ts, tj * ts : (tj + 1) * ts] = gathered[
                ..., global_index, :, :
            ]
        return out


class DistributedTileKNNHaloPadding_AllGather(torch.nn.Module):
    """Materialize halo-padded local tiles from distributed interior tiles."""

    def __init__(
        self,
        *,
        topology: TileTopology,
        regrid: torch.nn.Module,
        lat_deg: torch.Tensor,
        lon_deg: torch.Tensor,
    ) -> None:
        super().__init__()
        self.topology = topology
        self.regrid = regrid
        self.register_buffer("lat_deg", lat_deg, persistent=False)
        self.register_buffer("lon_deg", lon_deg, persistent=False)

    @classmethod
    def from_padded_face_grid(
        cls,
        *,
        topology: TileTopology,
        lon: torch.Tensor,
        lat: torch.Tensor,
        pad_width_data: int,
        device: torch.device,
        k: int = 4,
        eps: float = 1e-7,
    ) -> "DistributedTileKNNHaloPadding_AllGather":
        """Build the halo padding module from padded face lon/lat grids."""
        local_lon = torch.stack(
            [
                _select_padded_tile(
                    lon,
                    tile,
                    tile_size=topology.tile_size,
                    halo_width=topology.halo_width,
                    pad_width_data=pad_width_data,
                )
                for tile in topology.local_tiles
            ]
        )
        local_lat = torch.stack(
            [
                _select_padded_tile(
                    lat,
                    tile,
                    tile_size=topology.tile_size,
                    halo_width=topology.halo_width,
                    pad_width_data=pad_width_data,
                )
                for tile in topology.local_tiles
            ]
        )

        src_lon = torch.cat(
            [
                _select_tile_interior(
                    lon,
                    tile,
                    tile_size=topology.tile_size,
                    pad_width_data=pad_width_data,
                ).flatten()
                for tile in topology.global_tiles
            ]
        )
        src_lat = torch.cat(
            [
                _select_tile_interior(
                    lat,
                    tile,
                    tile_size=topology.tile_size,
                    pad_width_data=pad_width_data,
                ).flatten()
                for tile in topology.global_tiles
            ]
        )

        regrid = (
            KNNS2Interpolator(
                src_lon,
                src_lat,
                local_lon.flatten(),
                local_lat.flatten(),
                k=k,
                eps=eps,
            )
            .float()
            .to(device)
        )
        return cls(
            topology=topology,
            regrid=regrid,
            lat_deg=local_lat,
            lon_deg=local_lon,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *shape, tiles_per_rank, nx, ny = x.shape
        if tiles_per_rank != self.topology.tiles_per_rank:
            raise ValueError(
                f"Expected {self.topology.tiles_per_rank} local tiles, got {tiles_per_rank}"
            )
        if nx != self.topology.tile_size or ny != self.topology.tile_size:
            raise ValueError(
                f"Expected interior tile shape ({self.topology.tile_size}, "
                f"{self.topology.tile_size}), got ({nx}, {ny})"
            )

        tensor_list = [torch.zeros_like(x) for _ in range(self.topology.world_size)]
        torch.distributed.all_gather(tensor_list, x.contiguous())
        gathered = torch.cat(tensor_list, dim=-3)
        gathered = gathered.reshape(*shape, -1)
        y = self.regrid(gathered)
        return y.reshape(
            *shape,
            tiles_per_rank,
            self.topology.padded_tile_size,
            self.topology.padded_tile_size,
        )


class NormalizedHaloAdjoint(torch.nn.Module):
    """Normalized adjoint of DistributedTileKNNHaloPadding_AllGather.

    Computes D_w⁻¹ Aᵀ (w ⊙ y), where A is the halo exchange operator,
    w is an optional spatial window applied to each padded tile, and
    D_w = diag(Aᵀ w) is the per-interior-point sum of window weights.

    With a uniform window (default) this reduces to the plain Shepard
    normalization D⁻¹ Aᵀ y.  A KBD or similar window downweights halo
    contributions relative to interior-tile contributions, matching the
    windowed overlap-add used in patch-based inference.

    Implementation note:

    A is equivalent to an all-gather followed by a local regridding operator and
    the transpose of an all-gather is a reduce scatter to return source-point
    contributions to the ranks that own them.  So A^T can be implemented by
    applying the transpose of the regridding followed by a reduce scatter. The
    transpose of regridding is implemented as a vector-Jacobian product through
    the forward operator using torch.func.vjp.
    """

    weights: torch.Tensor
    window: torch.Tensor

    def __init__(
        self,
        halo_exchange: DistributedTileKNNHaloPadding_AllGather,
        window: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.halo_exchange = halo_exchange
        topology = halo_exchange.topology
        padded_size = topology.padded_tile_size
        self._local_src_chunk = topology.tiles_per_rank * topology.tile_size**2
        self._global_src_size = self._local_src_chunk * topology.world_size
        device = halo_exchange.lon_deg.device
        dtype = halo_exchange.lon_deg.dtype

        if window is None:
            window = torch.ones(padded_size, padded_size, device=device, dtype=dtype)
        # Flatten and tile window across all local tiles: (n_dst,)
        window_flat = (
            window.to(device=device, dtype=dtype)
            .flatten()
            .repeat(topology.tiles_per_rank)
        )
        self.register_buffer("window", window_flat)
        self._vjp = {}
        weights = self._distributed_regrid_vjp(window_flat.unsqueeze(0))
        self.register_buffer(
            "weights",
            weights.reshape(
                1, 1, topology.tiles_per_rank, topology.tile_size, topology.tile_size
            ),
        )

    def _reduce_scatter_local_shard(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self._global_src_size:
            raise ValueError(
                f"Expected global source size {self._global_src_size}, got {x.shape[-1]}"
            )
        if self.halo_exchange.topology.world_size == 1:
            return x

        *shape, n_src = x.shape
        batch_flat = x.numel() // n_src
        x_2d = x.reshape(batch_flat, n_src).contiguous()
        chunks = list(x_2d.split(self._local_src_chunk, dim=-1))
        if len(chunks) != self.halo_exchange.topology.world_size:
            raise ValueError(
                f"Expected {self.halo_exchange.topology.world_size} equal source shards, got {len(chunks)}"
            )
        out = torch.empty(
            batch_flat,
            self._local_src_chunk,
            device=x.device,
            dtype=x.dtype,
        )
        torch.distributed.reduce_scatter(out, chunks)
        return out.reshape(*shape, self._local_src_chunk)

    def _local_regrid_vjp(self, y: torch.Tensor) -> torch.Tensor:
        key = (y.shape, y.dtype, y.device)
        if key not in self._vjp:
            # vjp - vector jacobian product
            # need to disable torch inference mode or the vjp function returned by
            # torch.func.vjp will silenty return zeros when used
            # after this it is not important
            with torch.inference_mode(False):
                cotangent = y.detach().clone()
                primal = torch.zeros(
                    *cotangent.shape[:-1],
                    self._global_src_size,
                    device=cotangent.device,
                    dtype=cotangent.dtype,
                )
                _, self._vjp[key] = torch.func.vjp(self.halo_exchange.regrid, primal)

        return self._vjp[key](y)[0]

    def _distributed_regrid_vjp(
        self,
        y: torch.Tensor,
    ) -> torch.Tensor:
        global_adj = self._local_regrid_vjp(y)
        return self._reduce_scatter_local_shard(global_adj)

    def forward(self, padded: torch.Tensor) -> torch.Tensor:
        topology = self.halo_exchange.topology
        *shape, _, _, _ = padded.shape
        padded_flat = padded.reshape(*shape, -1) * self.window
        accumulated = self._distributed_regrid_vjp(padded_flat)
        return (
            accumulated.reshape(
                *shape, topology.tiles_per_rank, topology.tile_size, topology.tile_size
            )
            / self.weights
        )


def _select_padded_tile(
    x: torch.Tensor,
    tile: Tile,
    *,
    tile_size: int,
    halo_width: int,
    pad_width_data: int,
) -> torch.Tensor:
    face, tile_i, tile_j = tile
    return x[
        ...,
        face,
        pad_width_data
        + tile_i * tile_size
        - halo_width : pad_width_data
        + (tile_i + 1) * tile_size
        + halo_width,
        pad_width_data
        + tile_j * tile_size
        - halo_width : pad_width_data
        + (tile_j + 1) * tile_size
        + halo_width,
    ]


def _select_tile_interior(
    x: torch.Tensor,
    tile: Tile,
    *,
    tile_size: int,
    pad_width_data: int,
) -> torch.Tensor:
    face, tile_i, tile_j = tile
    return x[
        face,
        pad_width_data + tile_i * tile_size : pad_width_data + (tile_i + 1) * tile_size,
        pad_width_data + tile_j * tile_size : pad_width_data + (tile_j + 1) * tile_size,
    ]
