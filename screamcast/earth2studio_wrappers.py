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
"""
Earth2Studio wrappers for SCREAM data sources and prognostic model.
"""

from __future__ import annotations

import dataclasses
import os
from collections import OrderedDict
from collections.abc import Generator, Iterator
from datetime import datetime

import numpy as np
import torch
import xarray as xr
from earth2grid import KNNS2Interpolator

import screamcast.catalog_core
import train as _train_module
from screamcast.ace import ACE2ForecastResidualModel
from screamcast.astronomy import calculate_cosine_zenith_direct
from screamcast.config import TrainConfig
from screamcast.cubesphere_transforms import reorder_cubesphere_to_2d_tensor
from screamcast.dali_ext_src import ScreamV2
from screamcast.datetime import as_py_datetime
from screamcast.model_registry import MixedPredictionAsymmetric_init
from screamcast.strata_wrappers import StrataBackboneModel, StrataModel


class ScreamDataSource:
    """
    Earth2Studio DataSource wrapper that loads a spatial tile of SCREAM zarr
    data efficiently, reading only the requested cells.
    """

    NE = 1024
    NPG = 2
    MODEL_DT_MINUTES = 10.0

    def __init__(
        self,
        var_group: dict,
        reference_datetime=datetime(2020, 10, 1, 0, 0, 0, tzinfo=None),
    ):
        self.ne = self.NE
        self.npg = self.NPG
        self.nside = self.NE * self.NPG
        self._reference_datetime = reference_datetime
        self._var_dims = ScreamV2.get_default_variables_dimensions()
        self._var_group = var_group

    def _parse_variable(self, var_str: str) -> tuple:
        """Parse a channel name into (zarr_var_name, level_or_None).

        If var_str is a known key in var_dims it is a 2D/1D variable.
        Otherwise split on the last '_' to get (zarr_name, int_level).
        """
        if var_str in self._var_dims:
            return var_str, None
        idx = var_str.rfind("_")
        if idx == -1:
            raise KeyError(f"Unknown variable: {var_str!r}")
        return var_str[:idx], int(var_str[idx + 1 :])

    def _compute_flat_indices(
        self, face_idx: int, x_range: np.ndarray, y_range: np.ndarray
    ) -> np.ndarray:
        """
        Compute flat zarr ncol indices for a 2D tile within a face.

        The zarr stores each face as ne*ne*npg*npg contiguous cells in
        (ne, ne, npg, npg) layout (element-row, element-col, gp-row, gp-col).

        The reordered 2D grid (as produced by reorder_cubesphere_to_2d_tensor)
        has position (xi, yi) corresponding to:
            elem_i = xi // npg,  gp_i = xi % npg
            elem_j = yi // npg,  gp_j = yi % npg

        In the flat (ne, ne, npg, npg) storage the index is:
            flat_in_face = (elem_i * ne + elem_j) * npg*npg + gp_i * npg + gp_j

        Returns a 2D array of shape (len(x_range), len(y_range)) with global flat indices.
        """
        ne = self.ne
        npg = self.npg
        cells_per_face = ne * ne * npg * npg
        face_offset = face_idx * cells_per_face

        xi = np.asarray(x_range, dtype=np.int64)
        yi = np.asarray(y_range, dtype=np.int64)

        # Meshgrid: rows = x_range, cols = y_range
        XI, YI = np.meshgrid(xi, yi, indexing="ij")  # shape (len(x), len(y))

        elem_i = XI // npg
        gp_i = XI % npg
        elem_j = YI // npg
        gp_j = YI % npg

        flat_in_face = (elem_i * ne + elem_j) * (npg * npg) + gp_i * npg + gp_j
        flat_global = face_offset + flat_in_face  # shape (len(x), len(y))

        return flat_global

    def _datetime_to_tidx(self, t) -> int:
        """Convert a datetime or np.datetime64 to a zarr time index."""
        t = as_py_datetime(t)
        delta_seconds = (t - self._reference_datetime).total_seconds()
        return round(delta_seconds / (self.MODEL_DT_MINUTES * 60))

    def _is_global_request(
        self, face_range: np.ndarray, x_range: np.ndarray, y_range: np.ndarray
    ) -> bool:
        expected_faces = np.arange(6, dtype=np.int64)
        expected_xy = np.arange(self.nside, dtype=np.int64)
        return (
            face_range.shape == expected_faces.shape
            and x_range.shape == expected_xy.shape
            and y_range.shape == expected_xy.shape
            and np.array_equal(face_range, expected_faces)
            and np.array_equal(x_range, expected_xy)
            and np.array_equal(y_range, expected_xy)
        )

    def _reshape_global_face_data(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw)
        return (
            raw.reshape(6, self.ne, self.ne, self.npg, self.npg)
            .transpose(0, 1, 3, 2, 4)
            .reshape(6, self.nside, self.nside)
        )

    def __call__(self, coords: dict) -> xr.DataArray:
        """
        Load a spatial tile of SCREAM data for the requested coordinates.

        Parameters
        ----------
        coords : dict
            Keys: "time", "variable", "face", "x", "y"
            - "time": array-like of datetime or np.datetime64
            - "variable": list of variable name strings (as built by __init__)
            - "face": array-like of face indices
            - "x": array of x indices within face (along first spatial dim)
            - "y": array of y indices within face (along second spatial dim)

        Returns
        -------
        xr.DataArray with dims ["time", "variable", "face", "x", "y"]
        """
        times = list(coords["time"])
        variables = list(coords["variable"])
        face_range = np.asarray(coords["face"], dtype=np.int64)
        x_range = np.asarray(coords["x"], dtype=np.int64)
        y_range = np.asarray(coords["y"], dtype=np.int64)

        zarr_keys = [self._parse_variable(v) for v in variables]

        nx = len(x_range)
        ny = len(y_range)
        nf = len(face_range)
        nt = len(times)
        nv = len(variables)
        is_global_request = self._is_global_request(face_range, x_range, y_range)

        flat_indices_by_face = None
        if not is_global_request:
            flat_indices_by_face = [
                self._compute_flat_indices(int(face_idx), x_range, y_range).ravel()
                for face_idx in face_range
            ]
        data = np.empty((nt, nv, nf, nx, ny), dtype=np.float32)

        for t_i, t in enumerate(times):
            t_idx = self._datetime_to_tidx(t)

            for v_i, (zarr_name, zarr_level) in enumerate(zarr_keys):
                ndim = self._var_dims[zarr_name]
                group = self._var_group[zarr_name]
                zarr_array = group[zarr_name]
                # much faster
                if is_global_request:
                    if ndim == "1D":
                        raw = zarr_array[:]
                    elif ndim == "2D":
                        raw = zarr_array[t_idx, :]
                    elif ndim == "3D":
                        raw = zarr_array[t_idx, int(zarr_level), :]
                    else:
                        raise ValueError(
                            f"Unknown dimensionality {ndim!r} for variable {zarr_name!r}"
                        )
                    data[t_i, v_i] = self._reshape_global_face_data(raw)
                    continue

                for f_i, flat_1d in enumerate(flat_indices_by_face):
                    if ndim == "1D":
                        raw = zarr_array.vindex[flat_1d]
                    elif ndim == "2D":
                        raw = zarr_array.vindex[t_idx, flat_1d]
                    elif ndim == "3D":
                        raw = zarr_array.vindex[t_idx, int(zarr_level), flat_1d]
                    else:
                        raise ValueError(
                            f"Unknown dimensionality {ndim!r} for variable {zarr_name!r}"
                        )
                    data[t_i, v_i, f_i] = raw.reshape(nx, ny)

        return xr.DataArray(
            data,
            dims=["time", "variable", "face", "x", "y"],
            coords={
                "time": times,
                "variable": variables,
                "face": face_range,
                "x": x_range,
                "y": y_range,
            },
        )


# ---------------------------------------------------------------------------
# Helper for building flat variable-name lists
# ---------------------------------------------------------------------------


def _build_variable_names(
    variables: tuple,
    levels: np.ndarray,
    var_dims: dict,
) -> list[str]:
    """Expand variable names into flat channel strings.

    3D variables get per-level suffixes ``"{name}_{level_idx}"``;
    2D/1D variables are returned as-is.
    """
    names: list[str] = []
    for v in variables:
        dim = var_dims.get(v, "2D")
        if dim == "3D":
            for lvl_idx in levels:
                names.append(f"{v}_{int(lvl_idx)}")  # noqa
        else:
            names.append(v)
    return names


def _tile_latlon(
    latlon: torch.Tensor | None, coords: OrderedDict, device: torch.device
) -> torch.Tensor:
    if latlon is None:
        raise ValueError("latlon is required to compute coszr.")

    face_ids = np.asarray(coords["face"])
    xs = np.asarray(coords["x"], dtype=np.int64)
    ys = np.asarray(coords["y"], dtype=np.int64)
    tile_ll = latlon[face_ids][:, xs][:, :, ys]
    return tile_ll.to(device)


# Forcing computed analytically from time/coords — excluded from required inputs.
_SCREAM_AUTONOMOUS_FORCING: frozenset[str] = frozenset({"coszr"})

# Time-invariant forcing — stored as a model buffer, excluded from required inputs.
_SCREAM_STATIC_FORCING: frozenset[str] = frozenset({"phis", "sgh30", "landfrac"})


# ---------------------------------------------------------------------------
# ScreamcastModel — earth2studio PrognosticModel wrapper
# ---------------------------------------------------------------------------


class ScreamcastModel(torch.nn.Module):
    """earth2studio ``PrognosticModel``-compatible wrapper around a screamcast pipeline.

    Parameters
    ----------
    pipeline:
        A ``MixedPredictionAsymmetric`` (or compatible) pipeline that has
        already been loaded and placed into eval mode.
    tile_size:
        Spatial extent of each cubesphere tile (pixels per side).
    levels:
        1-D array of vertical-level indices used by the model,
        e.g. ``np.arange(3, 128, 4)``.
    variables_prognostic:
        Tuple of prognostic variable names.
    variables_forcing:
        Tuple of forcing variable names.
    variables_diagnostic:
        Tuple of diagnostic variable names.
    dt:
        Model time-step as a :class:`numpy.timedelta64`, default 600 s.
    """

    def __init__(
        self,
        pipeline,
        tile_size: int,
        levels: np.ndarray,
        variables_prognostic: tuple,
        variables_forcing: tuple,
        variables_diagnostic: tuple,
        dt: np.timedelta64 = np.timedelta64(600, "s"),
        latlon: torch.Tensor | None = None,
        static_forcing: torch.Tensor | None = None,
        persistence: bool = False,
        inference_batch_size: int | None = None,
    ) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.tile_size = tile_size
        # latlon: [f, nside, nside, 2] in degrees (lat, lon), or None
        self._n_faces: int = latlon.shape[0] if latlon is not None else 6
        self.register_buffer("latlon", latlon, persistent=False)
        # _src_latlon/static_forcing: original grid at construction time.
        # set_latlon() always re-grids from these, so it is safe to call
        # repeatedly.  These share tensor storage with latlon/static_forcing
        # initially (no extra memory) and diverge after the first set_latlon
        # call.  Invariant: never mutate these in-place.
        self.register_buffer("_src_latlon", latlon, persistent=False)
        self.levels = levels
        self.variables_prognostic = variables_prognostic
        self.variables_forcing = variables_forcing
        self.variables_diagnostic = variables_diagnostic
        self.dt = dt

        self._static_forcing_names = [
            v for v in variables_forcing if v in _SCREAM_STATIC_FORCING
        ]
        if static_forcing is not None:
            sf_f32 = static_forcing.to(torch.float32)
            self.register_buffer("static_forcing", sf_f32, persistent=False)
            # Source copy for re-gridding; same tensor object, no extra memory.
            self.register_buffer("_src_static_forcing", sf_f32, persistent=False)
        else:
            self.register_buffer("static_forcing", None, persistent=False)
            self.register_buffer("_src_static_forcing", None, persistent=False)

        var_dims = ScreamV2.get_default_variables_dimensions()

        self._input_variable_names = np.array(
            _build_variable_names(
                tuple(variables_prognostic),
                levels,
                var_dims,
            )
        )
        self._output_variable_names = np.array(
            _build_variable_names(
                tuple(variables_prognostic) + tuple(variables_diagnostic),
                levels,
                var_dims,
            )
        )
        self._n_prog_channels = len(
            _build_variable_names(tuple(variables_prognostic), levels, var_dims)
        )

        self.persistence = persistence
        self.inference_batch_size = inference_batch_size

        # Mixin hooks (no-ops by default; callers may replace them)
        self.front_hook = lambda x, c: (x, c)
        self.rear_hook = lambda x, c: (x, c)

    # ------------------------------------------------------------------
    # PrognosticModel protocol
    # ------------------------------------------------------------------

    def input_coords(self) -> OrderedDict:
        """Return the input coordinate system.

        The ``face``, ``x``, and ``y`` entries reflect the full grid stored in
        the model (all faces, full tile size).  Callers performing sub-tile or
        single-face inference must override these entries in the returned dict
        before passing it to the model.
        """
        return OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0, dtype="datetime64[ns]"),
                "lead_time": np.array([np.timedelta64(0, "ns")]),
                "variable": self._input_variable_names,
                "face": np.arange(self._n_faces),
                "x": np.arange(self.tile_size),
                "y": np.arange(self.tile_size),
            }
        )

    def output_coords(self, input_coords: OrderedDict) -> OrderedDict:
        """Return the output coordinate system derived from *input_coords*."""
        out = OrderedDict(input_coords)
        out["lead_time"] = input_coords["lead_time"] + self.dt
        out["variable"] = self._output_variable_names
        return out

    def compile(self):
        self.pipeline.network = torch.compile(self.pipeline.network)

    def _build_full_input(self, x: torch.Tensor, coords: OrderedDict) -> torch.Tensor:
        """Append forcing channels to the prognostic state tensor.

        ``x`` has shape ``[batch, n_prog, faces, h, w]``.  Returns a tensor
        with all forcing channels appended along dim 1, ready for the pipeline.
        """
        b, _, f, h, w = x.shape
        forcing_channels: list[torch.Tensor] = []

        for name in self.variables_forcing:
            if name in _SCREAM_AUTONOMOUS_FORCING:
                if "time" not in coords or "lead_time" not in coords:
                    raise ValueError(
                        f"coords must contain 'time' and 'lead_time' to compute {name!r}"
                    )
                if len(coords["time"]) == 0:
                    raise ValueError(
                        f"coords['time'] must contain at least one timestamp to compute {name!r}"
                    )
                if len(coords["lead_time"]) == 0:
                    raise ValueError(
                        f"coords['lead_time'] must contain at least one offset to compute {name!r}"
                    )
                tile_ll = _tile_latlon(self.latlon, coords, x.device)
                tile_lat_deg = tile_ll[..., 0].detach().cpu().numpy()
                tile_lon_deg = tile_ll[..., 1].detach().cpu().numpy()
                valid_time = coords["time"][0] + coords["lead_time"][0]
                coszr = calculate_cosine_zenith_direct(
                    valid_time, tile_lat_deg, tile_lon_deg
                )
                coszr_t = torch.from_numpy(coszr.astype(np.float32)).to(
                    device=x.device, dtype=x.dtype
                )
                forcing_channels.append(coszr_t[None, None].expand(b, 1, f, h, w))
            elif name in _SCREAM_STATIC_FORCING:
                if self.static_forcing is None:
                    raise ValueError(
                        f"Static forcing {name!r} was not provided at construction."
                    )
                i = self._static_forcing_names.index(name)
                face_ids = np.asarray(coords["face"])
                xs = np.asarray(coords["x"], dtype=np.int64)
                ys = np.asarray(coords["y"], dtype=np.int64)
                tile = self.static_forcing[i][face_ids][:, xs][:, :, ys]  # [f, h, w]
                forcing_channels.append(
                    tile[None, None].expand(b, 1, f, h, w).to(dtype=x.dtype)
                )
            else:
                raise ValueError(
                    f"Forcing variable {name!r} is not in _SCREAM_AUTONOMOUS_FORCING "
                    "or _SCREAM_STATIC_FORCING."
                )

        if not forcing_channels:
            return x
        return torch.cat([x, *forcing_channels], dim=1)

    def __call__(
        self,
        x: torch.Tensor,
        coords: OrderedDict,
    ) -> tuple[torch.Tensor, OrderedDict]:
        """Run the model for one time step.

        Parameters
        ----------
        x:
            Input tensor of shape ``[batch, in_channels, f, tile_size, tile_size]``
            where ``f`` is the number of faces.
        coords:
            Input coordinate dictionary.

        Returns
        -------
        output:
            Tensor of shape ``[batch, out_channels, f, tile_size, tile_size]``.
        output_coords:
            Updated coordinate dictionary.
        """
        x_in = self._build_full_input(x, coords)
        b, c, f, H, W = x_in.shape
        # Flatten faces into the batch dimension: [b*f, c, H, W]
        x_flat = x_in.permute(0, 2, 1, 3, 4).reshape(b * f, c, H, W)

        if self.latlon is not None:
            tile_ll = _tile_latlon(self.latlon, coords, x.device)
            # repeat for batch: [b*f, h, w]; pipeline expects radians
            lat = torch.deg2rad(tile_ll[..., 0]).repeat(b, 1, 1)
            lon = torch.deg2rad(tile_ll[..., 1]).repeat(b, 1, 1)
            index = {"lat": lat, "lon": lon}
        else:
            index = torch.zeros(b * f, dtype=torch.long, device=x_in.device)

        if self.persistence:
            prog = x_flat[:, : self._n_prog_channels]
            n_out = len(self._output_variable_names)
            n_diag = n_out - self._n_prog_channels
            if n_diag > 0:
                diag = torch.zeros(
                    prog.shape[0],
                    n_diag,
                    *prog.shape[2:],
                    device=prog.device,
                    dtype=prog.dtype,
                )
                output_full = torch.cat([prog, diag], dim=1)
            else:
                output_full = prog
        else:
            with torch.inference_mode():
                if self.inference_batch_size is None:
                    output_full, _ = self.pipeline.step(x_flat, index)
                else:
                    chunks = []
                    for start in range(0, x_flat.shape[0], self.inference_batch_size):
                        end = start + self.inference_batch_size
                        x_chunk = x_flat[start:end]
                        if isinstance(index, dict):
                            index_chunk = {k: v[start:end] for k, v in index.items()}
                        else:
                            index_chunk = index[start:end]
                        out_chunk, _ = self.pipeline.step(x_chunk, index_chunk)
                        chunks.append(out_chunk)
                    output_full = torch.cat(chunks, dim=0)

        out_c = output_full.shape[1]
        output = output_full.reshape(b, f, out_c, H, W).permute(0, 2, 1, 3, 4)
        return output, self.output_coords(coords)

    def create_iterator(
        self,
        x: torch.Tensor,
        coords: OrderedDict,
    ) -> Iterator[tuple[torch.Tensor, OrderedDict]]:
        """Create a time-integration iterator.

        Yields the initial condition first (lead time 0), then repeatedly
        calls :meth:`__call__` to advance the state.

        Parameters
        ----------
        x:
            Initial condition tensor of shape
            ``[batch, in_channels, f, tile_size, tile_size]``.
        coords:
            Initial coordinate dictionary (must include ``"lead_time"``).

        Yields
        ------
        (output_tensor, output_coords)
        """
        yield from self._default_generator(x, coords)

    def _default_generator(
        self,
        x: torch.Tensor,
        coords: OrderedDict,
    ) -> Generator[tuple[torch.Tensor, OrderedDict], None, None]:
        # Shallow-copy so we don't mutate the caller's dict when we update
        # lead_time below.
        coords = OrderedDict(coords)
        # Yield initial condition with prognostic values from the input state
        # and NaNs for diagnostic channels, which do not exist on the input.
        coords_out = OrderedDict(coords)
        coords_out["lead_time"] = coords["lead_time"][-1:]
        coords_out["variable"] = self._output_variable_names

        n_diag_channels = len(self._output_variable_names) - self._n_prog_channels
        if n_diag_channels > 0:
            diag_fill = torch.full(
                (x.shape[0], n_diag_channels, *x.shape[2:]),
                torch.nan,
                device=x.device,
                dtype=x.dtype,
            )
            x_init = torch.cat([x[:, : self._n_prog_channels], diag_fill], dim=1)
        else:
            x_init = x[:, : self._n_prog_channels]
        yield x_init, coords_out

        while True:
            x, coords = self.front_hook(x, coords)

            x_out, coords_out = self(x, coords)

            x_out, coords_out = self.rear_hook(x_out, coords_out)

            # Advance lead_time for the next input state while keeping the
            # reference time unchanged.
            coords["lead_time"] = coords_out["lead_time"]
            x = x_out[:, : self._n_prog_channels]

            yield x_out, coords_out

    def set_tile_size(self, tile_size: int) -> None:
        """Override the tile size used for inference.

        Updates ``self.tile_size`` and recomputes the static RoPE buffers in
        the underlying DiT network so the model accepts ``tile_size × tile_size``
        inputs instead of the size it was trained on.
        """
        self.tile_size = tile_size
        # Both wrapper classes expose set_tile_size(height, width); DiT_Pixel's
        # implementation delegates to the semantic stage and also refreshes its
        # own pixel-pathway RoPE buffers when applicable.
        self.pipeline.network.set_tile_size(tile_size, tile_size)

    def to_persistence(self) -> "ScreamcastModel":
        self.persistence = True
        return self

    def set_latlon(
        self,
        lat_deg: torch.Tensor,
        lon_deg: torch.Tensor,
        *,
        regrid_method: str = "nearest",
    ) -> None:
        """Set the lat/lon grid and re-grid static forcing to match.

        Re-gridding always uses the original source grid stored at construction,
        so this method may be called multiple times safely.  Also calls
        :meth:`set_tile_size` internally; callers do not need a separate
        ``set_tile_size`` call for grids set via this method.

        .. note::
            If :meth:`compile` has been called, ``set_latlon`` will still
            invoke ``set_tile_size``, which mutates the network's RoPE buffers.
            This is safe only when the new tile size equals the compiled size;
            call :meth:`compile` after all grid configuration is finalised.

        Parameters
        ----------
        lat_deg, lon_deg:
            Tensors of shape ``[f, n, n]`` in degrees. Any number of faces ``f``
            is accepted.
        regrid_method:
            Interpolation method for static forcing. Only ``"nearest"`` is
            currently supported.
        """
        if regrid_method != "nearest":
            raise NotImplementedError(f"regrid_method={regrid_method!r}")

        lat_deg = lat_deg.float()
        lon_deg = lon_deg.float()
        if lat_deg.shape[-2] != lat_deg.shape[-1]:
            raise ValueError(
                f"set_latlon requires a square spatial grid; "
                f"got shape {tuple(lat_deg.shape)} (h={lat_deg.shape[-2]}, w={lat_deg.shape[-1]})"
            )

        if self._static_forcing_names and self._src_static_forcing is None:
            raise ValueError(
                f"set_latlon cannot re-grid static forcing {self._static_forcing_names} "
                "because no static_forcing was provided at construction."
            )

        if self._src_static_forcing is not None:
            if self._src_latlon is None:
                raise ValueError(
                    "Cannot regrid static forcing without a source latlon on this model."
                )
            src_lat = self._src_latlon[..., 0].reshape(-1).cpu()
            src_lon = self._src_latlon[..., 1].reshape(-1).cpu()
            tgt_lat = lat_deg.reshape(-1).cpu()
            tgt_lon = lon_deg.reshape(-1).cpu()
            interp = (
                KNNS2Interpolator(src_lon, src_lat, tgt_lon, tgt_lat, k=1)
                .to(self._src_static_forcing.device)
                .float()
            )
            tgt_shape = tuple(lat_deg.shape)
            channels = [
                interp(
                    self._src_static_forcing[i]
                    .reshape(-1)
                    .to(device=self._src_static_forcing.device, dtype=torch.float32)
                ).reshape(tgt_shape)
                for i in range(self._src_static_forcing.shape[0])
            ]
            self.register_buffer(
                "static_forcing", torch.stack(channels), persistent=False
            )  # [n_static, f, n, n]

        self.register_buffer(
            "latlon", torch.stack([lat_deg, lon_deg], dim=-1), persistent=False
        )  # [f, n, n, 2] degrees
        self._n_faces = lat_deg.shape[0]
        self.set_tile_size(lat_deg.shape[-1])

    def reset_latlon(self) -> None:
        """Restore the original lat/lon grid and static forcing from construction.

        Reverts any changes made by :meth:`set_latlon`, returning the model to
        its as-constructed grid topology.  Raises if no source latlon was
        available at construction (i.e. ``latlon=None`` was passed).
        """
        if self._src_latlon is None:
            raise ValueError(
                "reset_latlon: no source latlon was provided at construction."
            )
        self.register_buffer("latlon", self._src_latlon, persistent=False)
        self.register_buffer(
            "static_forcing", self._src_static_forcing, persistent=False
        )
        self._n_faces = self._src_latlon.shape[0]
        self.set_tile_size(self._src_latlon.shape[-2])

    def disable_activation_checkpointing(self) -> bool:
        """Disable runtime activation checkpointing on supported networks.

        Returns
        -------
        bool
            True if a supported checkpointing knob was found and changed.
        """
        network = self.pipeline.network
        if isinstance(network, (StrataModel, StrataBackboneModel)):
            network.disable_activation_checkpointing()
            return True
        return False

    # ------------------------------------------------------------------
    # Class method: load from checkpoint
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        scream_data: screamcast.catalog_core.ScreamData | None = None,
        bf16: bool = False,
        disable_activation_checkpointing: bool = True,
    ) -> "ScreamcastModel":
        """Construct a :class:`ScreamcastModel` from a saved checkpoint.

        Parameters
        ---------
        checkpoint_path:
            Path to the ``.pth`` checkpoint file saved by ``train.py``.
        bf16:
            If True, enable bfloat16 mixed-precision for inference via the
            model's internal autocast guards.
        disable_activation_checkpointing:
            If True, disable runtime activation checkpointing on supported DiT
            backbones after loading the checkpoint. This is enabled by default
            for inference-oriented wrapper usage.

        Returns
        -------
        ScreamcastModel
            Model loaded and placed into eval mode.
        """

        ckpt_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        train_cfg = TrainConfig.from_dict(ckpt_data["train_config"])

        # Fix stale latlon path baked into old checkpoints from other clusters
        aux_data_root = os.environ.get("AUX_DATA_ROOT", "data")
        latlon_path = os.path.join(
            aux_data_root, os.path.basename(train_cfg.data.latlon_path)
        )
        train_cfg = dataclasses.replace(
            train_cfg,
            data=dataclasses.replace(train_cfg.data, latlon_path=latlon_path),
        )

        data_cfg = train_cfg.data

        levels = np.arange(data_cfg.level_start, data_cfg.level_end, data_cfg.plevel)

        nside = data_cfg.cubesphere_ne * data_cfg.cubesphere_npg
        in_channels_3d = len(data_cfg.variables_prognostic) + len(
            data_cfg.variables_forcing
        )
        out_channels_3d = len(data_cfg.variables_prognostic) + len(
            data_cfg.variables_diagnostic
        )
        num_depth_levels = len(levels)

        model_type = train_cfg.experiment.model_type

        if train_cfg.dit.do_rotate_wind:
            wind_channel_indices = (
                data_cfg.variables_prognostic.index("U"),
                data_cfg.variables_prognostic.index("V"),
            )
        else:
            wind_channel_indices = None

        def _network_factory():
            if model_type == "pixeldit":
                return _train_module.build_strata(
                    in_channels=in_channels_3d,
                    out_channels=out_channels_3d,
                    nside=nside,
                    tile_size=data_cfg.tile_size,
                    dit_cfg=train_cfg.dit,
                    pixel_cfg=train_cfg.pixel_dit,
                    do_bf16_mixed=bf16,
                    depth_levels=num_depth_levels,
                    wind_channel_indices=wind_channel_indices,
                    grid_type="cubesphere",
                    cubesphere_latlon_path=data_cfg.latlon_path,
                )
            else:
                return _train_module.build_backbone(
                    in_channels=in_channels_3d,
                    out_channels=out_channels_3d,
                    nside=nside,
                    tile_size=data_cfg.tile_size,
                    dit_cfg=train_cfg.dit,
                    do_bf16_mixed=bf16,
                    depth_levels=num_depth_levels,
                    wind_channel_indices=wind_channel_indices,
                    grid_type="cubesphere",
                    cubesphere_latlon_path=data_cfg.latlon_path,
                )

        pipeline_factory = MixedPredictionAsymmetric_init(
            _network_factory,
            torch.nn.MSELoss,
            "",  # experiment_name unused when pretrained=False
            plevel=data_cfg.plevel,
            level_start=data_cfg.level_start,
            level_end=data_cfg.level_end,
            variables_prognostic=data_cfg.variables_prognostic,
            variables_forcing=data_cfg.variables_forcing,
            variables_diagnostic=data_cfg.variables_diagnostic,
            variables_prognostic_state=data_cfg.variables_prognostic_state,
            enable_3d_adapter=True,
            do_qv_softplus=train_cfg.pipeline.do_qv_softplus,
            do_precip_relu=train_cfg.pipeline.do_precip_relu,
            variables_input_zeroed=train_cfg.pipeline.variables_input_zeroed,
        )

        pipeline = pipeline_factory(pretrained=False)

        # Legacy checkpoints (pre-Strata module names, torch.compile
        # _orig_mod. prefixes) are translated inside the network wrapper's
        # load_state_dict pre-hook (screamcast.checkpoint_compat), so the raw
        # state dict loads as-is under strict=True.
        pipeline.load_checkpoint(ckpt_data)
        pipeline.eval()

        # Load lat/lon for index_is_latlon models: [6, nside, nside, 2] in degrees
        latlon = None
        if train_cfg.dit.index_is_latlon and data_cfg.latlon_path:
            with xr.open_dataset(data_cfg.latlon_path) as ds_ll:
                lat = torch.from_numpy(ds_ll["lat"].values.astype(np.float32))
                lon = torch.from_numpy(ds_ll["lon"].values.astype(np.float32))
            ne, npg = data_cfg.cubesphere_ne, data_cfg.cubesphere_npg
            ncol_per_face = ne * ne * npg * npg
            # reshape flat [6*ncol_per_face] → [6, nside, nside]
            lat_2d = torch.stack(
                [
                    reorder_cubesphere_to_2d_tensor(
                        lat[i * ncol_per_face : (i + 1) * ncol_per_face], ne=ne, npg=npg
                    )
                    for i in range(6)
                ]
            )  # [6, nside, nside]
            lon_2d = torch.stack(
                [
                    reorder_cubesphere_to_2d_tensor(
                        lon[i * ncol_per_face : (i + 1) * ncol_per_face], ne=ne, npg=npg
                    )
                    for i in range(6)
                ]
            )
            latlon = torch.stack(
                [lat_2d, lon_2d], dim=-1
            )  # [6, nside, nside, 2] degrees

        # Load static forcing fields (phis, sgh30, landfrac …) from the aux zarr
        # at full [n_static, 6, nside, nside] resolution; tiled to the inference
        # window at runtime inside _build_full_input.
        static_forcing = None
        static_vars = [
            v for v in data_cfg.variables_forcing if v in _SCREAM_STATIC_FORCING
        ]
        if static_vars:
            if scream_data is None:
                import data_catalog

                scream_data = data_catalog.scream_sdecadal

            group = scream_data.to_xarray(chunks=None)
            grid_type = str(group.attrs.get("grid"))

            if grid_type != screamcast.catalog_core.Grids.ne1024pg2:
                raise NotImplementedError(f"{grid_type}")

            ne, npg = screamcast.catalog_core.GRID_INFO[grid_type]
            ncol_per_face = ne * ne * npg * npg
            channels = []
            for var in static_vars:
                flat = torch.from_numpy(group[var][:].values.astype(np.float32))
                channels.append(
                    torch.stack(
                        [
                            reorder_cubesphere_to_2d_tensor(
                                flat[i * ncol_per_face : (i + 1) * ncol_per_face],
                                ne=ne,
                                npg=npg,
                            )
                            for i in range(6)
                        ]
                    )
                )  # [6, nside, nside]
            static_forcing = torch.stack(channels)  # [n_static, 6, nside, nside]

        model = cls(
            pipeline=pipeline,
            tile_size=data_cfg.tile_size,
            levels=levels,
            variables_prognostic=data_cfg.variables_prognostic,
            variables_forcing=data_cfg.variables_forcing,
            variables_diagnostic=data_cfg.variables_diagnostic,
            latlon=latlon,
            static_forcing=static_forcing,
        )
        if disable_activation_checkpointing:
            model.disable_activation_checkpointing()
        return model


__all__ = ["ScreamcastModel", "ACE2ForecastResidualModel"]
