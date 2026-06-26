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
"""Regional averages for cubed-sphere SCREAMcast rollouts.

Area-weighted averages over a fixed set of regions (``global``, ``land``,
``ocean``, ``tropics``, ``extratropics``) computed per rollout step for every
model output channel. Designed for the distributed tiled rollout in
``scripts/ace/run_screamcast_nudged.py``.

Regions are defined as weights ``w_r(i) = area(i) * f_r(i)``:

- ``global``:        ``f = 1``
- ``land``:          ``f = landfrac``
- ``ocean``:         ``f = 1 - landfrac``
- ``tropics``:       ``f = 1{|lat| < cutoff}``
- ``extratropics``:  ``f = 1{|lat| >= cutoff}``

The averager holds pre-computed local weights for each rank's interior tiles
and a cached global normalizer per region. Each step performs a weighted sum
reduction over local tiles, a single ``all_reduce(SUM)`` across ranks, and a
divide by the cached normalizer. On rank 0 the per-step regional averages are
written to a pre-allocated netCDF file, grouped per base variable exactly like
the raw zarr output (via ``_split_variable_levels``).

No unit conversions are applied: values are returned and written in the model
output's native units. The caller is responsible for conversions such as
``kg/m**2/s`` -> ``mm/day`` for precipitation.

The output netCDF file is opened in HDF5 SWMR (single-writer-multiple-reader)
mode, so it can safely be opened for reading by another process while the
rollout is still running. ``Dataset.sync()`` is called after every step write
so that readers see fresh data promptly.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from typing import Any

import h5netcdf
import h5py
import numpy as np
import torch
import xarray as xr

from screamcast.cubesphere_transforms import reorder_cubesphere_to_2d_tensor
from screamcast.distributed_halo import TileTopology

logger = logging.getLogger(__name__)

REGIONS: tuple[str, ...] = ("global", "land", "ocean", "tropics", "extratropics")


def _unstructured_to_faces_np(
    flat_1d: np.ndarray, *, ne: int, npg: int
) -> torch.Tensor:
    """Reshape a 1-D unstructured cubed-sphere field to ``[6, nside, nside]``.

    Mirrors the per-face reshape used by
    ``screamcast.earth2studio_wrappers.ScreamcastModel.from_checkpoint`` for
    static forcing fields, so regional weights align with the model's own
    static inputs.
    """
    ncol_per_face = ne * ne * npg * npg
    expected = 6 * ncol_per_face
    if flat_1d.shape != (expected,):
        raise ValueError(
            f"Expected 1-D array of length {expected} (6 * ne^2 * npg^2), "
            f"got shape {flat_1d.shape}"
        )
    flat_t = torch.as_tensor(flat_1d, dtype=torch.float32)
    return torch.stack(
        [
            reorder_cubesphere_to_2d_tensor(
                flat_t[i * ncol_per_face : (i + 1) * ncol_per_face], ne=ne, npg=npg
            )
            for i in range(6)
        ]
    )


def _tiles_from_face(face_field: torch.Tensor, topology: TileTopology) -> torch.Tensor:
    """Slice local interior tiles from a full-face ``[6, nside, nside]`` field.

    Returns ``[tiles_per_rank, tile_size, tile_size]``.
    """
    ts = topology.tile_size
    return torch.stack(
        [
            face_field[face, ti * ts : (ti + 1) * ts, tj * ts : (tj + 1) * ts]
            for face, ti, tj in topology.local_tiles
        ]
    )


def _build_region_weights(
    *,
    area: torch.Tensor,
    lat_deg: torch.Tensor,
    landfrac: torch.Tensor,
    tropics_lat_cutoff_deg: float,
) -> "OrderedDict[str, torch.Tensor]":
    """Build per-region weight tensors matching the shape of ``area``.

    ``area``, ``lat_deg``, and ``landfrac`` must be on the same device and have
    the same shape (typically ``[tiles_per_rank, tile_size, tile_size]``).
    """
    if not (area.shape == lat_deg.shape == landfrac.shape):
        raise ValueError(
            "area, lat_deg, and landfrac must share a shape; got "
            f"{tuple(area.shape)}, {tuple(lat_deg.shape)}, {tuple(landfrac.shape)}"
        )
    abs_lat = lat_deg.abs()
    tropics = (abs_lat < tropics_lat_cutoff_deg).to(area.dtype)
    weights: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    weights["global"] = area.clone()
    weights["land"] = area * landfrac
    weights["ocean"] = area * (1.0 - landfrac)
    weights["tropics"] = area * tropics
    weights["extratropics"] = area * (1.0 - tropics)
    return weights


def _coerce_attr(value: Any) -> Any:
    """Return a value netCDF4 can store as an attribute."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (str, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


class RegionalAverager:
    """Per-step area-weighted regional averages with netCDF output.

    The averager is constructed once before the rollout (typically via
    :py:meth:`from_topology`), then called with ``(step_index, out_tensor)``
    once per step. ``out_tensor`` is expected to have shape
    ``[1, n_channels, tiles_per_rank, tile_size, tile_size]`` — the same
    layout the rollout loop already produces after ``unpad``.

    Only rank 0 writes to the netCDF file; all ranks participate in the
    cross-rank ``all_reduce`` used to aggregate local weighted sums.
    """

    def __init__(
        self,
        *,
        weights: "OrderedDict[str, torch.Tensor]",
        normalizers: "OrderedDict[str, float]",
        variable_names: list[str],
        grouped_variables: "OrderedDict[str, list[tuple[int | None, int]]]",
        output_path: str | None,
        rank: int,
        device: torch.device,
        t0: np.datetime64,
        dt: np.timedelta64,
        n_steps: int,
        hyam: np.ndarray | None = None,
        hybm: np.ndarray | None = None,
        attrs: dict | None = None,
    ) -> None:
        if list(weights.keys()) != list(normalizers.keys()):
            raise ValueError("weights and normalizers must share the same keys")
        self.regions: tuple[str, ...] = tuple(weights.keys())
        # Stack per-region weights into [n_regions, *tile_shape] for einsum.
        self._weight_stack = torch.stack(
            [weights[r].to(device=device, dtype=torch.float32) for r in self.regions],
            dim=0,
        )
        self._normalizers = torch.tensor(
            [normalizers[r] for r in self.regions],
            device=device,
            dtype=torch.float32,
        )
        self.variable_names = list(variable_names)
        self.grouped_variables = grouped_variables
        self.rank = rank
        self.device = device
        self.n_steps = n_steps
        self.output_path = output_path
        self._ds: h5netcdf.File | None = None
        if rank == 0 and output_path is not None:
            self._ds = self._create_netcdf(
                output_path=output_path,
                t0=t0,
                dt=dt,
                n_steps=n_steps,
                hyam=hyam,
                hybm=hybm,
                attrs=attrs or {},
            )

    def _create_netcdf(
        self,
        *,
        output_path: str,
        t0: np.datetime64,
        dt: np.timedelta64,
        n_steps: int,
        hyam: np.ndarray | None,
        hybm: np.ndarray | None,
        attrs: dict,
    ) -> h5netcdf.File:
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        # libver="latest" is required to enable HDF5 SWMR below.
        ds = h5netcdf.File(output_path, "w", libver="latest")

        ds.dimensions["time"] = 1
        ds.dimensions["step"] = n_steps
        ds.dimensions["region"] = len(self.regions)

        time_var = ds.create_variable("time", ("time",), dtype="f8")
        t0_seconds = np.asarray(t0, dtype="datetime64[s]").astype("int64")
        time_var[:] = np.array([t0_seconds], dtype="f8")
        time_var.attrs["units"] = "seconds since 1970-01-01T00:00:00"
        time_var.attrs["calendar"] = "proleptic_gregorian"

        step_var = ds.create_variable("step", ("step",), dtype="f8")
        step_deltas = (
            (np.arange(1, n_steps + 1) * np.asarray(dt))
            .astype("timedelta64[s]")
            .astype("int64")
        )
        step_var[:] = step_deltas.astype("f8")
        step_var.attrs["units"] = "seconds"
        step_var.attrs["long_name"] = "forecast lead time"

        str_dtype = h5py.string_dtype(encoding="utf-8")
        region_var = ds.create_variable("region", ("region",), dtype=str_dtype)
        region_var[:] = np.array(list(self.regions), dtype=object)

        has_level = False
        if hyam is not None and hybm is not None:
            first_3d = next(
                (
                    entries
                    for entries in self.grouped_variables.values()
                    if entries[0][0] is not None
                ),
                None,
            )
            if first_3d is not None:
                levels = np.asarray([lvl for lvl, _ in first_3d], dtype=np.int32)
                ds.dimensions["level"] = len(levels)
                lvl_var = ds.create_variable("level", ("level",), dtype="i4")
                lvl_var[:] = levels
                hyam_var = ds.create_variable("hyam", ("level",), dtype="f4")
                hyam_var[:] = hyam[levels].astype(np.float32)
                hybm_var = ds.create_variable("hybm", ("level",), dtype="f4")
                hybm_var[:] = hybm[levels].astype(np.float32)
                has_level = True

        for base_name, entries in self.grouped_variables.items():
            if entries[0][0] is None:
                ds.create_variable(
                    base_name,
                    ("time", "step", "region"),
                    dtype="f4",
                    fillvalue=np.float32(np.nan),
                )
            else:
                if not has_level:
                    raise ValueError(
                        f"3-D variable {base_name!r} requires hyam/hybm to build "
                        "the netCDF 'level' dimension"
                    )
                expected_levels = len(ds.dimensions["level"])
                if len(entries) != expected_levels:
                    raise ValueError(
                        f"Variable {base_name!r} has {len(entries)} levels but "
                        f"netCDF 'level' dim has {expected_levels}"
                    )
                ds.create_variable(
                    base_name,
                    ("time", "step", "region", "level"),
                    dtype="f4",
                    fillvalue=np.float32(np.nan),
                )

        for k, v in attrs.items():
            ds.attrs[k] = _coerce_attr(v)

        # Enable HDF5 Single-Writer-Multiple-Reader mode so the file can be
        # opened by other processes (e.g. a live monitoring script) while the
        # rollout is still running. Must be set after ALL dims, variables, and
        # attributes are created; once enabled, no new objects can be added.
        ds.flush()
        ds._h5file.swmr_mode = True
        return ds

    def __call__(
        self, step_index: int, out_tensor: torch.Tensor
    ) -> "OrderedDict[str, dict[str, float]]":
        """Compute regional averages and write a step slice to netCDF.

        Args:
            step_index: Zero-based step index; used as the ``step`` dim index
                when writing to the netCDF file.
            out_tensor: Local rollout output with shape
                ``[1, n_channels, tiles_per_rank, tile_size, tile_size]``.

        Returns:
            An ordered mapping ``{region: {channel: value}}`` of area-weighted
            averages in the input units. The caller is responsible for any
            unit conversions (e.g. precipitation ``kg/m**2/s`` -> ``mm/day``).
        """
        if out_tensor.ndim != 5 or out_tensor.shape[0] != 1:
            raise ValueError(
                f"Expected shape [1, C, T, H, W], got {tuple(out_tensor.shape)}"
            )
        n_channels = out_tensor.shape[1]
        if n_channels != len(self.variable_names):
            raise ValueError(
                f"n_channels={n_channels} does not match variable_names "
                f"({len(self.variable_names)})"
            )
        if out_tensor.shape[2:] != self._weight_stack.shape[1:]:
            raise ValueError(
                "out_tensor tile shape "
                f"{tuple(out_tensor.shape[2:])} does not match weight shape "
                f"{tuple(self._weight_stack.shape[1:])}"
            )

        x = out_tensor[0].to(device=self._weight_stack.device, dtype=torch.float32)
        # local_sums[r, c] = sum_{t, h, w} w[r, t, h, w] * x[c, t, h, w]
        local_sums = torch.einsum("rthw,cthw->rc", self._weight_stack, x)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(local_sums, op=torch.distributed.ReduceOp.SUM)

        means = local_sums / self._normalizers.unsqueeze(-1)
        means_cpu = means.detach().cpu().numpy()

        result: "OrderedDict[str, dict[str, float]]" = OrderedDict()
        for ri, region_name in enumerate(self.regions):
            result[region_name] = {
                self.variable_names[ci]: float(means_cpu[ri, ci])
                for ci in range(n_channels)
            }

        if self._ds is not None:
            self._write_step(step_index, means_cpu)
        return result

    def _write_step(self, step_index: int, means: np.ndarray) -> None:
        if not 0 <= step_index < self.n_steps:
            raise IndexError(
                f"step_index {step_index} out of range [0, {self.n_steps})"
            )
        for base_name, entries in self.grouped_variables.items():
            var = self._ds[base_name]
            if entries[0][0] is None:
                _, ch_idx = entries[0]
                var[0, step_index, :] = means[:, ch_idx].astype(np.float32)
            else:
                channel_indices = np.array([ci for _, ci in entries], dtype=np.int64)
                var[0, step_index, :, :] = means[:, channel_indices].astype(np.float32)
        self._ds.flush()

    def close(self) -> None:
        if self._ds is not None:
            self._ds.close()
            self._ds = None

    @classmethod
    def from_topology(
        cls,
        *,
        topology: TileTopology,
        lat_face_deg: torch.Tensor,
        landfrac_face: torch.Tensor,
        area_face: torch.Tensor,
        variable_names: list[str],
        grouped_variables: "OrderedDict[str, list[tuple[int | None, int]]]",
        output_path: str | None,
        t0: np.datetime64,
        dt: np.timedelta64,
        n_steps: int,
        device: torch.device,
        tropics_lat_cutoff_deg: float = 30.0,
        hyam: np.ndarray | None = None,
        hybm: np.ndarray | None = None,
        attrs: dict | None = None,
    ) -> "RegionalAverager":
        """Build an averager from full-face inputs and a tile topology.

        ``lat_face_deg``, ``landfrac_face``, and ``area_face`` are each
        ``[6, nside, nside]`` tensors aligned with the interior tiles described
        by ``topology``.
        """
        for name, t in (
            ("lat_face_deg", lat_face_deg),
            ("landfrac_face", landfrac_face),
            ("area_face", area_face),
        ):
            if t.shape != (
                topology.n_faces,
                topology.face_size,
                topology.face_size,
            ):
                raise ValueError(
                    f"{name} must have shape "
                    f"(6, {topology.face_size}, {topology.face_size}); got "
                    f"{tuple(t.shape)}"
                )

        lat_local = _tiles_from_face(lat_face_deg, topology).to(
            device=device, dtype=torch.float32
        )
        landfrac_local = _tiles_from_face(landfrac_face, topology).to(
            device=device, dtype=torch.float32
        )
        area_local = _tiles_from_face(area_face, topology).to(
            device=device, dtype=torch.float32
        )

        weights = _build_region_weights(
            area=area_local,
            lat_deg=lat_local,
            landfrac=landfrac_local,
            tropics_lat_cutoff_deg=tropics_lat_cutoff_deg,
        )

        local_totals = torch.stack([w.sum() for w in weights.values()])
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                local_totals, op=torch.distributed.ReduceOp.SUM
            )
        totals_cpu = local_totals.detach().cpu().numpy()
        normalizers: "OrderedDict[str, float]" = OrderedDict()
        for (name, _), total in zip(weights.items(), totals_cpu):
            if not np.isfinite(total) or total <= 0.0:
                raise ValueError(
                    f"Region {name!r} has non-positive total weight {total}. "
                    "Check area/landfrac inputs."
                )
            normalizers[name] = float(total)

        combined_attrs = dict(attrs or {})
        combined_attrs.setdefault(
            "tropics_lat_cutoff_deg", float(tropics_lat_cutoff_deg)
        )
        combined_attrs.setdefault(
            "region_definitions",
            "global=all; land=landfrac; ocean=1-landfrac; "
            f"tropics=|lat|<{tropics_lat_cutoff_deg}; "
            f"extratropics=|lat|>={tropics_lat_cutoff_deg}",
        )
        combined_attrs.setdefault("land_weight_kind", "landfrac_continuous")

        return cls(
            weights=weights,
            normalizers=normalizers,
            variable_names=variable_names,
            grouped_variables=grouped_variables,
            output_path=output_path,
            rank=topology.rank,
            device=device,
            t0=t0,
            dt=dt,
            n_steps=n_steps,
            hyam=hyam,
            hybm=hybm,
            attrs=combined_attrs,
        )


def load_grid_area_face(
    scrip_path: str, *, ne: int = 1024, npg: int = 2
) -> torch.Tensor:
    """Load ``grid_area`` from a SCRIP file into a ``[6, nside, nside]`` tensor."""
    with xr.open_dataset(scrip_path) as ds:
        area = np.asarray(ds["grid_area"].values, dtype=np.float32)
    return _unstructured_to_faces_np(area, ne=ne, npg=npg)


def load_landfrac_face(scream_data, *, ne: int = 1024, npg: int = 2) -> torch.Tensor:
    """Load ``landfrac`` from a ``ScreamData`` catalog into ``[6, nside, nside]``.

    Uses the same reshape path that
    ``screamcast.earth2studio_wrappers.ScreamcastModel.from_checkpoint`` uses
    to build its static forcing buffer, so the result aligns with the model's
    interior tiles.
    """
    group = scream_data.to_xarray(chunks=None)
    flat = np.asarray(group["landfrac"][:].values, dtype=np.float32)
    return _unstructured_to_faces_np(flat, ne=ne, npg=npg)
