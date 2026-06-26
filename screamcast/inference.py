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
import dataclasses
import datetime
import logging
import os
import tempfile
from typing import List

import cartopy.crs
import einops
import matplotlib.animation
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torchmetrics
import tqdm
import xarray as xr
from earth2grid import healpix_bare
from matplotlib.gridspec import GridSpec

import screamcast.cached_dataset
from screamcast.dali_ext_src import (
    ScreamV2,
    reorder_morton_to_hpx_pad,
)


def _load_cubesphere_latlon_from_index(
    *,
    latlon_path: str,
    index: torch.Tensor,
):
    """
    Load CubeSphere lon/lat for an arbitrary selection of points described by `index`,
    returning arrays with the same shape as `index`.

    Expects a dataset with coords 'lat' and 'lon' of shape (ncol,),
    where ncol matches the dataset's total number of columns.
    """
    ds = xr.open_dataset(latlon_path)
    if "lat" not in ds.coords or "lon" not in ds.coords:
        raise ValueError(
            f"{latlon_path} must contain 'lat' and 'lon' coords, got coords={list(ds.coords)}"
        )
    if "ncol" not in ds.dims:
        raise ValueError(
            f"{latlon_path} must have dim 'ncol', got dims={list(ds.dims)}"
        )
    ncol = int(ds.dims["ncol"])

    # Ensure indices are on CPU for numpy conversion
    idx_np = index.detach().to("cpu").numpy().astype("int64", copy=False)
    if idx_np.size == 0:
        raise ValueError("index_flat is empty")
    if idx_np.min() < 0 or idx_np.max() >= ncol:
        raise ValueError(
            f"index_2d contains out-of-range values for {latlon_path}: "
            f"min={int(idx_np.min())}, max={int(idx_np.max())}, ncol={ncol}"
        )

    lat = ds["lat"].values[idx_np]
    lon = ds["lon"].values[idx_np]
    return lon, lat


@dataclasses.dataclass(frozen=True)
class PlotField:
    loc: tuple[str, int | None]
    units: str
    cmap: str = "viridis"
    scale: float = 1.0
    norm: matplotlib.colors.Normalize = dataclasses.field(
        default_factory=matplotlib.colors.Normalize
    )


# Version-specific default channels and fields
default_channels = (
    ("qv", 127, None),
    ("qv", 79, None),
    ("z_mid", 79, None),
    ("omega", 79, None),
    ("PotentialTemperature", 79, 1),
)

default_channels_cubesphere = (
    ("qv", 31, None),
    ("qv", 19, None),
    ("z_mid", 19, None),
    ("omega", 19, None),
    ("PotentialTemperature", 19, None),
)

# for video inference output
Fields = [
    PlotField(loc=("T_2m", None), units="deg K"),
    PlotField(
        loc=("qv", 79),
        scale=1000.0,
        units="g/kg",
    ),
    PlotField(
        loc=("precip_liq_surf_mass_flux", None),
        scale=1000 * 86400,
        norm=matplotlib.colors.LogNorm(vmin=0.1, clip=True, vmax=90),
        units="mm/day",
    ),
    PlotField(
        loc=("omega", 79),
        norm=matplotlib.colors.Normalize(vmin=-3, vmax=3),
        units="Pa/s",
    ),
    PlotField(loc=("U", 79), units="m/s"),
]

REGIONS = {
    "india": (80, 15),
    "s india": (80, 4),
    "s china sea": (112, 10),
    "fl": (-81, 27),
    "n atlantic sea": (-56, 55),
    "labrador sea": (-43, 56),
}


def _get_nearest(index: pd.MultiIndex, loc):
    name, level = loc
    levels_idx = index[index.get_loc(name)].get_level_values(1)
    if len(levels_idx) == 1:
        nearest_level = levels_idx[0]
    else:
        nearest_level = levels_idx[levels_idx.get_indexer([level], "nearest")[0]]
    return (name, nearest_level)


class EvalIndiaOceantile:
    def __init__(
        self,
        device,
        grid_type: str = "healpix",
        plevel=4,
        tile_size: int = 256,
        level_start: int = 3,
        level_end: int = 128,
        variables_prognostic: tuple = (
            "PotentialTemperature",
            "U",
            "V",
            "geopotential_mid",
            "omega",
            "qv",
            "T_2m",
        ),
        variables_forcing: tuple = ("coszr", "sst", "phis"),
        variables_diagnostic: tuple = (
            "precip_ice_surf_mass_flux",
            "precip_liq_surf_mass_flux",
        ),
        inference_start_index: int = 200,
        cubesphere_latlon_path: str = "../data/latlon_ne1024pg2.nc",
    ):
        if grid_type == "cubesphere":
            face = 1  # indian ocean face
        elif grid_type == "healpix":
            face = 5  # indian ocean face
        self.tile_size = tile_size
        self.device = device
        self.plevel = plevel
        self.level_start = level_start
        self.level_end = level_end
        self.variables_prognostic = variables_prognostic
        self.variables_forcing = variables_forcing
        self.variables_diagnostic = variables_diagnostic
        # the global index of the timestep to be used for inference plots
        self.inference_start_index = inference_start_index
        self.cubesphere_latlon_path = cubesphere_latlon_path

        ds = ScreamV2(
            batch_size=1,
            split="test",
            grid_type=grid_type,
            plevel=plevel,
            level_start=level_start,
            level_end=level_end,
            variables_prognostic=variables_prognostic,
            variables_forcing=variables_forcing,
            variables_diagnostic=variables_diagnostic,
        )

        self.ds = ds

        data = ds.load_patch_input(
            t=inference_start_index, patch=face
        )  # Input data (all variables)
        data = ds._post_process(data)

        data1 = ds.load_patch_output(
            t=inference_start_index + 1, patch=face
        )  # Output data (prognostic variables only)
        data1 = ds._post_process(data1)

        start = face * ds.patch_size
        index = torch.arange(start, start + ds.patch_size)
        index = ds._post_process(index)
        idx = ds.channel_index_output()  # Use output channel index for evaluation

        self.index = index
        self.data = data
        self.data1 = data1
        self.channel_index = idx
        # Side length for a single face (HEALPix: 1024; CubeSphere defaults to 2048)
        self.nside = int(ds.nside)
        self.grid_type = grid_type
        self._cubesphere_lonlat = None

    def __call__(
        self,
        get_output,
        output_dir: str = "",
        channels=None,
        plot_vertical: bool = True,
        index_is_latlon: bool | None = None,
        lat_radians: np.ndarray | torch.Tensor | None = None,
        lon_radians: np.ndarray | torch.Tensor | None = None,
    ):
        """Save figures of state increment for some fields to ``output_dir``

        get_output: (input, index) -> output_full in physical units

        If ``output_dir`` is not provided than the figure will not be closed. This
        is useful for jupyter notebooks.

        plot_vertical: if True, also create vertical cross-section plots

        """
        # Use version-specific defaults if no channels provided
        if channels is None:
            if self.grid_type == "cubesphere":
                channels = default_channels_cubesphere
            else:
                channels = default_channels

        logging.getLogger("screamcast.inference.EvalIndiaOceantile").info(
            f"Creating {os.path.abspath(output_dir)}"
        )
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        if self.grid_type == "healpix":
            lon, lat = healpix_bare.pix2ang(
                self.nside, self.index.flatten(), nest=True, lonlat=True
            )
            lon = lon.reshape((self.nside, self.nside))
            lat = lat.reshape((self.nside, self.nside))
        else:
            if self._cubesphere_lonlat is None:
                lon2d, lat2d = _load_cubesphere_latlon_from_index(
                    latlon_path=self.cubesphere_latlon_path,
                    index=self.index,
                )
                self._cubesphere_lonlat = (lon2d, lat2d)
            lon, lat = self._cubesphere_lonlat
        tiles = einops.rearrange(
            self.data,
            "c (h x) (h1 y) -> (h h1) c x y",
            x=self.tile_size,
            y=self.tile_size,
        )
        index = einops.rearrange(
            self.index,
            "(h x) (h1 y) -> (h h1)  x y",
            x=self.tile_size,
            y=self.tile_size,
        )
        if index_is_latlon is None:
            index_is_latlon = False
        if index_is_latlon and (lat_radians is None or lon_radians is None):
            raise ValueError(
                "lat_radians/lon_radians are required when index_is_latlon=True"
            )
        if index_is_latlon:
            lat_base = (
                lat_radians
                if torch.is_tensor(lat_radians)
                else torch.as_tensor(lat_radians)
            )
            lon_base = (
                lon_radians
                if torch.is_tensor(lon_radians)
                else torch.as_tensor(lon_radians)
            )

        # chunked evaluation of the data to save memory
        outputs = []
        with torch.no_grad():
            for i in range(tiles.size(0)):
                if index_is_latlon:
                    idx_tile = index[i : i + 1].to("cpu").long()
                    lat_tile = torch.as_tensor(lat_base[idx_tile], device=self.device)
                    lon_tile = torch.as_tensor(lon_base[idx_tile], device=self.device)
                    index_tile = {"lat": lat_tile, "lon": lon_tile}
                else:
                    index_tile = index[i : i + 1].to(self.device)
                output = get_output(tiles[i : i + 1].to(self.device), index_tile)
                outputs.append(output.cpu())

        output = torch.cat(outputs, 0)

        prognostic_tiles = self._extract_prognostic_from_input(tiles)
        output_prognostic = self._extract_prognostic_from_output(output)
        diff = output_prognostic - prognostic_tiles

        diff = einops.rearrange(
            diff, "(h h1) c x y -> c (h x) (h1 y)", h=self.nside // self.tile_size
        )

        output = einops.rearrange(
            output, "(h h1) c x y -> c (h x) (h1 y)", h=self.nside // self.tile_size
        )

        def plot(x, pos=111, ax=None, **kwargs):
            if not ax:
                ax = plt.subplot(pos, projection=cartopy.crs.PlateCarree())
            ret = ax.pcolormesh(
                lon, lat, x, **kwargs, transform=cartopy.crs.PlateCarree()
            )
            ax.coastlines()
            return ret

        def plot_field(field, level, m=None):
            try:
                loc = _get_nearest(self.channel_index, (field, level))
                i = self.channel_index.get_loc(loc)
            except (KeyError, IndexError):
                logging.warning(f"Channel ({field}, {level}) not found, skipping")
                return

            prognostic_data = self._extract_prognostic_from_input(
                self.data.unsqueeze(0)
            )[0]
            truth = self.data1[i] - prognostic_data[i]

            prediction = diff[i].cpu()
            proj = cartopy.crs.Orthographic(central_longitude=90, central_latitude=0)
            # Use GridSpec for uniform panel sizes with dedicated colorbar space
            fig = plt.figure(figsize=(16, 12))
            # 2 rows x 4 cols: [plot, cbar, plot, cbar] layout
            gs = GridSpec(
                2, 4, width_ratios=[1, 0.04, 1, 0.04], wspace=0.08, hspace=0.25
            )
            a = fig.add_subplot(gs[0, 0], projection=proj)
            cax_a = fig.add_subplot(gs[0, 1])
            b = fig.add_subplot(gs[0, 2], projection=proj)
            cax_b = fig.add_subplot(gs[0, 3])
            c = fig.add_subplot(gs[1, 0], projection=proj)
            cax_c = fig.add_subplot(gs[1, 1])
            d = fig.add_subplot(gs[1, 2], projection=proj)
            cax_d = fig.add_subplot(gs[1, 3])

            if m is None:
                m = max(-truth.quantile(0.005), truth.quantile(0.995))

            def p(x, ax, m, cmap=None):
                return plot(x, ax=ax, vmin=-m, vmax=m, cmap=cmap)

            # Row 1: prediction and truth (same colorscale)
            im_a = p(prediction.cpu(), ax=a, m=m)
            a.set_title(f"{field} difference @ {level=}\n (prediction)")
            fig.colorbar(im_a, cax=cax_a)

            im_b = p(truth, ax=b, m=m)
            b.set_title(f"{field} difference @ {level=}\n (truth)")
            fig.colorbar(im_b, cax=cax_b)

            # Row 2, Col 1: prediction - truth with symmetric colorscale and RdBu_r
            error = prediction.cpu() - truth.cpu()
            error_m = max(-error.quantile(0.005).item(), error.quantile(0.995).item())
            im_err = p(error, ax=c, m=error_m, cmap="RdBu_r")
            c.set_title(f"{field} @ {level=}\n (prediction - truth)")
            fig.colorbar(im_err, cax=cax_c)

            # Row 2, Col 2: 64x64 coarsened difference
            coarse_size = 64
            # Coarsen error, lon, lat using non-overlapping 64x64 tiles
            error_np = error.numpy()
            lon_np = lon if isinstance(lon, np.ndarray) else lon.cpu().numpy()
            lat_np = lat if isinstance(lat, np.ndarray) else lat.cpu().numpy()
            h, w = error_np.shape
            n_tiles_h = h // coarse_size
            n_tiles_w = w // coarse_size
            # Trim to exact multiple of coarse_size
            error_trimmed = error_np[
                : n_tiles_h * coarse_size, : n_tiles_w * coarse_size
            ]
            lon_trimmed = lon_np[: n_tiles_h * coarse_size, : n_tiles_w * coarse_size]
            lat_trimmed = lat_np[: n_tiles_h * coarse_size, : n_tiles_w * coarse_size]
            # Reshape and average over tiles
            error_coarse = error_trimmed.reshape(
                n_tiles_h, coarse_size, n_tiles_w, coarse_size
            ).mean(axis=(1, 3))
            lon_coarse = lon_trimmed.reshape(
                n_tiles_h, coarse_size, n_tiles_w, coarse_size
            ).mean(axis=(1, 3))
            lat_coarse = lat_trimmed.reshape(
                n_tiles_h, coarse_size, n_tiles_w, coarse_size
            ).mean(axis=(1, 3))
            # Symmetric colorscale for coarsened error
            coarse_m = max(np.abs(error_coarse).max(), 1e-10)
            im_coarse = d.pcolormesh(
                lon_coarse,
                lat_coarse,
                error_coarse,
                vmin=-coarse_m,
                vmax=coarse_m,
                cmap="RdBu_r",
                transform=cartopy.crs.PlateCarree(),
            )
            d.coastlines()
            d.set_title(
                f"{field} @ {level=}\n (coarsened {coarse_size}x{coarse_size} error)"
            )
            fig.colorbar(im_coarse, cax=cax_d)
            if output_dir:
                path = os.path.join(output_dir, f"{field}-{level}.png")
                path = os.path.abspath(path)
                logging.getLogger("screamcast.inference.EvalIndiaOceantile").info(
                    f"Saving figure to {path}."
                )
                fig.savefig(path, bbox_inches="tight")
                plt.close(fig)

        for channel, level, m in channels:
            plot_field(channel, level, m=m)

        # Plot vertical cross-sections if requested
        if plot_vertical:
            self._plot_vertical_cross_sections(diff, output, output_dir)

    def _plot_vertical_cross_sections(self, diff, output, output_dir):
        """Create vertical cross-section plots (x-z slices) for all 3D variables"""

        # Get middle slice in y direction
        middle_y = self.nside // 2
        prognostic_data = self._extract_prognostic_from_input(self.data.unsqueeze(0))[0]

        # Identify 3D variables by finding variables with multiple pressure levels
        variables_3d = {}
        for var_name, level in self.channel_index:
            if level is not None and pd.notna(
                level
            ):  # This is a 3D variable with valid pressure level
                if var_name not in variables_3d:
                    variables_3d[var_name] = []
                variables_3d[var_name].append(
                    (level, self.channel_index.get_loc((var_name, level)))
                )

        # Sort each variable's levels
        for var_name in variables_3d:
            variables_3d[var_name].sort(key=lambda x: x[0])

        logging.getLogger("screamcast.inference.EvalIndiaOceantile").info(
            f"Creating vertical cross-sections for 3D variables: {list(variables_3d.keys())}"
        )

        for var_name, level_info in variables_3d.items():
            self._plot_variable_vertical_cross_section(
                var_name,
                level_info,
                diff,
                output,
                middle_y,
                output_dir,
                prognostic_data,
            )

    def _plot_variable_vertical_cross_section(
        self, var_name, level_info, diff, output, middle_y, output_dir, prognostic_data
    ):
        """Plot vertical cross-section for a single 3D variable"""

        # Extract data for all pressure levels of this variable at middle y slice
        levels = [level for level, _ in level_info]
        channel_indices = [idx for _, idx in level_info]

        is_diagnostic = var_name in self.variables_diagnostic

        if is_diagnostic:
            # Diagnostic variable: plot predicted state vs truth state
            truth_slices = []
            pred_slices = []
            for idx in channel_indices:
                truth_slices.append(self.data1[idx][middle_y, :])
                pred_slices.append(output[idx][middle_y, :])
            truth_2d = torch.stack(truth_slices, dim=0)
            pred_2d = torch.stack(pred_slices, dim=0)
        else:
            # Prognostic variable: plot predicted difference vs truth difference
            truth_slices = []
            pred_slices = []
            for idx in channel_indices:
                truth_slices.append(
                    (self.data1[idx] - prognostic_data[idx])[middle_y, :]
                )
                pred_slices.append(diff[idx][middle_y, :])
            truth_2d = torch.stack(truth_slices, dim=0)
            pred_2d = torch.stack(pred_slices, dim=0)

        # Create the vertical cross-section plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

        # Create coordinate arrays for plotting
        x_coords = np.arange(self.nside)
        pressure_coords = np.array(levels, dtype=float)

        # Slice to show only middle 256 pixels in x direction
        middle_pixels = 256
        start_x = (self.nside - middle_pixels) // 2
        end_x = start_x + middle_pixels

        # Slice coordinates and data
        x_coords = x_coords[start_x:end_x]
        truth_2d = truth_2d[:, start_x:end_x]  # [n_levels, middle_pixels]
        pred_2d = pred_2d[:, start_x:end_x]  # [n_levels, middle_pixels]

        # Validate coordinates
        if not np.all(np.isfinite(x_coords)):
            logging.warning(f"Non-finite values in x_coords for {var_name}")
            return
        if not np.all(np.isfinite(pressure_coords)):
            logging.warning(
                f"Non-finite values in pressure_coords for {var_name}: {pressure_coords}"
            )
            return

        # Determine color scale
        combined_data = torch.cat([truth_2d.flatten(), pred_2d.flatten()])

        # Remove any remaining non-finite values for quantile calculation
        finite_mask = torch.isfinite(combined_data)
        if not finite_mask.any():
            logging.warning(
                f"No finite values in data for {var_name}, skipping vertical plot"
            )
            plt.close(fig)
            return

        finite_data = combined_data[finite_mask]

        if is_diagnostic:
            if var_name in ["omega", "U", "V"]:
                cmap = "RdBu_r"
                vmin_q, vmax_q = finite_data.quantile(0.005), finite_data.quantile(
                    0.995
                )
                vmax = max(abs(vmin_q.item()), abs(vmax_q.item()))
                vmin = -vmax
            else:
                vmin = finite_data.quantile(0.005).item()
                vmax = finite_data.quantile(0.995).item()
                cmap = "viridis"
            pred_title = f"{var_name} state (prediction)"
            truth_title = f"{var_name} state (truth)"
            cbar_label = f"{var_name} state"
        else:
            vmin_q, vmax_q = finite_data.quantile(0.005), finite_data.quantile(0.995)
            max_abs = max(abs(vmin_q.item()), abs(vmax_q.item()))
            vmin, vmax = -max_abs, max_abs
            cmap = "RdBu_r"
            pred_title = f"{var_name} difference (prediction)"
            truth_title = f"{var_name} difference (truth)"
            cbar_label = f"{var_name} difference"

        # Plot prediction and truth panels
        _ = ax1.pcolormesh(
            x_coords,
            pressure_coords,
            pred_2d.cpu().numpy(),
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        ax1.set_title(
            f"{pred_title}\nVertical cross-section at y={middle_y} (x={start_x}-{end_x})"
        )
        ax1.set_xlabel("X coordinate")
        ax1.set_ylabel("Vertical level")
        ax1.invert_yaxis()

        im2 = ax2.pcolormesh(
            x_coords,
            pressure_coords,
            truth_2d.cpu().numpy(),
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        ax2.set_title(
            f"{truth_title}\nVertical cross-section at y={middle_y} (x={start_x}-{end_x})"
        )
        ax2.set_xlabel("X coordinate")
        ax2.set_ylabel("Vertical level")
        ax2.invert_yaxis()
        plt.colorbar(im2, ax=[ax1, ax2], shrink=0.8, label=cbar_label)
        if output_dir:
            path = os.path.join(output_dir, f"{var_name}-vertical.png")
            path = os.path.abspath(path)
            logging.getLogger("screamcast.inference.EvalIndiaOceantile").info(
                f"Saving vertical cross-section to {path}."
            )
            fig.savefig(path, bbox_inches="tight", dpi=150)
            plt.close(fig)

    def _extract_prognostic_from_input(self, input_tensor):
        """Extract prognostic variables from input tensor"""
        if not self.ds.variables_forcing:
            return input_tensor
        ranges_input = ScreamV2.ranges_input(
            variables_prognostic=self.variables_prognostic,
            variables_forcing=self.variables_forcing,
            plevel=self.plevel,
            level_start=self.level_start,
            level_end=self.level_end,
        )

        prognostic_parts = []
        for var in self.variables_prognostic:
            input_slice = ranges_input[var]
            prognostic_parts.append(input_tensor[:, input_slice])

        return torch.cat(prognostic_parts, dim=1)

    def _extract_prognostic_from_output(self, output_tensor):
        """Extract prognostic variables from input tensor"""
        if not self.ds.variables_diagnostic:
            return output_tensor
        ranges_output = ScreamV2.ranges_output(
            variables_prognostic=self.variables_prognostic,
            variables_diagnostic=self.variables_diagnostic,
            plevel=self.plevel,
            level_start=self.level_start,
            level_end=self.level_end,
        )

        prognostic_parts = []
        for var in self.variables_prognostic:
            output_slice = ranges_output[var]
            prognostic_parts.append(output_tensor[:, output_slice])

        return torch.cat(prognostic_parts, dim=1)


class VideoInferenceLoader(torch.utils.data.Dataset):
    """Loads patches for inference"""

    def __init__(self, ds, patch, local_start, size, tile_size):
        self.ds = ds
        self.patch = patch
        self.local_start = local_start
        self.size = size
        self.patch_size = tile_size

    def __getitem__(self, t):
        return _load_data_input(
            self.ds, t, self.patch, self.local_start, self.patch_size
        ), _load_data_output(self.ds, t, self.patch, self.local_start, self.patch_size)


def _load_data_input(ds, t, patch, local_start, patch_size):
    if ds.grid_type == "cubesphere":
        raise NotImplementedError("CubeSphere not supported for VideoInferenceLoader")
    array = ds.load_patch_input(t=t, patch=patch)
    array = array[:, local_start : local_start + patch_size**2]
    array = reorder_morton_to_hpx_pad(
        torch.as_tensor(array), shape=(patch_size, patch_size)
    )
    return array


def _load_data_output(ds, t, patch, local_start, patch_size):
    if ds.grid_type == "cubesphere":
        raise NotImplementedError("CubeSphere not supported for VideoInferenceLoader")
    array = ds.load_patch_output(t=t, patch=patch)
    array = array[:, local_start : local_start + patch_size**2]
    array = reorder_morton_to_hpx_pad(
        torch.as_tensor(array), shape=(patch_size, patch_size)
    )
    return array


def _save_video(
    output: list[tuple[torch.Tensor, torch.Tensor]],
    output_path: str,
    window: torch.Tensor | None,
    spec: PlotField,
    lat,
    lon,
):
    fig = plt.figure()
    truth, pred = output[0]

    ax1 = fig.add_subplot(1, 2, 1, projection=cartopy.crs.PlateCarree())
    ax1.set_title("Truth")
    im1 = ax1.pcolormesh(lon, lat, truth, norm=spec.norm, cmap=spec.cmap)
    ax1.coastlines()

    ax2 = fig.add_subplot(1, 2, 2, projection=cartopy.crs.PlateCarree())
    ax2.set_title("Prediction")
    im2 = ax2.pcolormesh(lon, lat, pred, norm=spec.norm, cmap=spec.cmap)
    if window is not None:
        ax2.contour(lon, lat, window.cpu(), colors="black")
    ax2.coastlines()
    cb = plt.colorbar(im2, ax=[ax1, ax2], orientation="horizontal", shrink=0.5)
    field, level = spec.loc
    cb.set_label(f"{field=} {level=} {spec.units}")
    title = fig.suptitle("")

    def update(frame):
        truth, pred = output[frame]
        lead_time = datetime.timedelta(minutes=10 * frame)
        title.set_text(str(lead_time))
        im1.set_array(truth)
        im2.set_array(pred)

    animation = matplotlib.animation.FuncAnimation(
        fig, update, range(len(output)), interval=100
    )
    animation.save(output_path)


class VideoInference:
    def mywindow(self, n, d=0.1):
        x = np.linspace(-1, 1, n)
        out = np.select(
            [x < -1 + d, x > (1 - d)],
            [
                (np.cos((x + 1) / d * np.pi) + 1) / 2,
                (np.cos((x - 1) / d * np.pi) + 1) / 2,
            ],
            0,
        )
        return 1 - out

    def __init__(
        self,
        ds,
        cache: bool = True,
        tile_size: int = 256,
        inference_start_index: int = 200,
    ):
        """
        Initialize VideoInference.
        Args:
            ds: ScreamV2 instance
            cache: Whether to use caching
            inference_start_index: the global index of the first timestep to be used for initializing the video
        """
        self.ds = ds
        self.cache = cache
        # Determine if we're dealing with asymmetric channels
        # is_asymmetric = True: either forcing or diagnostic variables are present
        # is_asymmetric = False: all variables are prognostic
        self.is_asymmetric = ds.variables_forcing or ds.variables_diagnostic

        self.input_channel_index = ds.channel_index_input()
        self.output_channel_index = ds.channel_index_output()
        self.tile_size = tile_size
        self.inference_start_index = inference_start_index

    def _extract_prognostic_from_input(self, input_tensor):
        """Extract prognostic variables from input tensor"""
        if not self.ds.variables_forcing:
            return input_tensor

        ranges_input = ScreamV2.ranges_input(
            variables_prognostic=self.ds.variables_prognostic,
            variables_forcing=self.ds.variables_forcing,
            plevel=self.ds.plevel,
            level_start=self.ds.level_start,
            level_end=self.ds.level_end,
        )
        prognostic_parts = []

        for var in self.ds.variables_prognostic:
            input_slice = ranges_input[var]
            prognostic_parts.append(input_tensor[:, input_slice])

        return torch.cat(prognostic_parts, dim=1)

    def _combine_prognostic_with_current_nonprognostic(
        self, predicted_prognostic, current_full_input
    ):
        """
        Combine predicted prognostic variables with current timestep's non-prognostic variables.

        Args:
            predicted_prognostic: [batch, out_channels, H, W] - predicted prognostic variables
            current_full_input: [batch, in_channels, H, W] - current timestep's full input from dataloader

        Returns:
            combined_input: [batch, in_channels, H, W] - combined input for next prediction
        """
        if not self.ds.variables_forcing:
            return predicted_prognostic

        # Start with current full input (has current non-prognostic variables)
        combined_input = current_full_input.clone()

        # Replace prognostic variables with predictions
        ranges_input = ScreamV2.ranges_input(
            variables_prognostic=self.ds.variables_prognostic,
            variables_forcing=self.ds.variables_forcing,
            plevel=self.ds.plevel,
            level_start=self.ds.level_start,
            level_end=self.ds.level_end,
        )
        prognostic_idx = 0

        for var in self.ds.variables_prognostic:
            input_slice = ranges_input[var]
            slice_size = input_slice.stop - input_slice.start
            combined_input[:, input_slice] = predicted_prognostic[
                :, prognostic_idx : prognostic_idx + slice_size
            ]
            prognostic_idx += slice_size

        return combined_input

    def _update_prognostic_in_full_output(self, full_output, prognostic):
        """Update prognostic variables in full_output"""
        ranges_output = ScreamV2.ranges_output(
            variables_prognostic=self.ds.variables_prognostic,
            variables_diagnostic=self.ds.variables_diagnostic,
            plevel=self.ds.plevel,
            level_start=self.ds.level_start,
            level_end=self.ds.level_end,
        )
        ranges_output_prognostic = ScreamV2.ranges_output(
            variables_prognostic=self.ds.variables_prognostic,
            variables_diagnostic=(),
            plevel=self.ds.plevel,
            level_start=self.ds.level_start,
            level_end=self.ds.level_end,
        )
        for var in self.ds.variables_prognostic:
            full_output[:, ranges_output[var]] = prognostic[
                :, ranges_output_prognostic[var]
            ]
        return full_output

    def __call__(self, *, pipeline, steps, do_window: bool, region: str):
        ts = list(range(self.inference_start_index, self.inference_start_index + steps))
        window = self.mywindow(self.tile_size, d=0.40)
        window = torch.as_tensor(window)
        window = torch.outer(window, window)
        window = window.cuda().float()

        if getattr(self.ds, "grid_type", "healpix") != "healpix":
            raise NotImplementedError(
                "VideoInference is currently implemented only for grid_type='healpix'."
            )

        # select variable to plot
        lon, lat = REGIONS[region]

        patch_size = self.tile_size
        nside = 1024 // patch_size
        pix = healpix_bare.ang2pix(
            nside, torch.tensor([lon]), torch.tensor([lat]), lonlat=True, nest=True
        )
        pix = pix.item()
        patch = pix // (nside * nside)
        sub_pix = pix % (nside * nside)
        start = patch * 1024 * 1024 + sub_pix * (patch_size * patch_size)
        local_start = start % (1024 * 1024)
        size = patch_size * patch_size

        index = np.r_[start : start + size]
        index = reorder_morton_to_hpx_pad(
            torch.as_tensor(index), shape=(patch_size, patch_size)
        )

        lon, lat = healpix_bare.pix2ang(1024, index.flatten(), nest=True, lonlat=True)
        lon = lon.reshape(index.shape)
        lat = lat.reshape(index.shape)
        index = index.unsqueeze(0).cuda()

        ax = plt.subplot(1, 1, 1, projection=cartopy.crs.PlateCarree())
        ax.pcolormesh(lon, lat, window.cpu())
        ax.coastlines()
        plt.savefig("window.png")

        outputs = {field: [] for field in Fields}

        dataset = VideoInferenceLoader(
            self.ds, patch, local_start, size, self.tile_size
        )

        if self.cache:
            dataset = screamcast.cached_dataset.LMDBCacheDataset(
                dataset,
                os.path.join(
                    tempfile.tempdir,
                    f"{patch}.{local_start}.{size}.{self.ds.plevel}.lmdb",
                ),
                map_size=int(10e9),
            )

        loader = torch.utils.data.DataLoader(
            dataset,
            sampler=ts,
            num_workers=1,
            multiprocessing_context="spawn",
            pin_memory=True,
        )

        iterator = iter(loader)
        truth_input = None  # to track prognostic + forcing variables
        truth_output = None  # to track prognostic + diagnostic variables
        predicted_prognostic = (
            None  # Track predicted prognostic variables for next iteration
        )

        for _ in tqdm.tqdm(ts):
            try:
                truth_input, truth_output = next(iterator)
                truth_input = truth_input.cuda()
                truth_output = truth_output.cuda()
            except StopIteration:
                pass

            if truth_input is None:
                raise RuntimeError("No initial condition loaded.")

            if predicted_prognostic is None:

                predicted_prognostic = self._extract_prognostic_from_input(truth_input)
                predicted_full = truth_output

            truth_for_comparison = truth_output
            comparison_channel_index = self.output_channel_index

            for field in outputs:
                loc = field.loc
                scale = field.scale

                try:
                    # Use prognostic channel index for both truth and prediction comparisons
                    i = comparison_channel_index.get_loc(loc)

                    outputs[field].append(
                        (
                            truth_for_comparison[0, i].cpu() * scale,
                            predicted_full[0, i].cpu() * scale,
                        )
                    )
                except (KeyError, IndexError):
                    # Skip fields that don't exist in this data version
                    logging.warning(
                        f"Field {loc} not found in channel index for {self.ds.__class__.__name__}, skipping"
                    )
                    continue

            # Subsequent iterations: use predicted prognostic variables
            combined_input = self._combine_prognostic_with_current_nonprognostic(
                predicted_prognostic, truth_input
            )

            with torch.no_grad():
                with torch.autocast("cuda", enabled=False):
                    predicted_full, predicted_prognostic = pipeline.step(
                        combined_input, index
                    )
                if do_window:
                    initial_state = pipeline.initialize(truth_input)
                    initial_prognostic = self._extract_prognostic_from_input(
                        initial_state
                    )
                    predicted_prognostic = (
                        predicted_prognostic * window
                        + (1 - window) * initial_prognostic
                    )
                    predicted_full = self._update_prognostic_in_full_output(
                        predicted_full, predicted_prognostic
                    )

        for field in outputs:
            if not outputs[field]:  # Skip empty field outputs
                continue

            loc = field.loc
            output_name = f"{loc[0]}"
            if do_window:
                output_name += ".windowed"
            output_path = output_name + ".mp4"
            _save_video(
                outputs[field],
                output_path,
                window if do_window else None,
                field,
                lat,
                lon,
            )


class MeanMetric(torchmetrics.Metric):
    def __init__(self, channels):
        super().__init__()
        self.add_state("sum", default=torch.zeros(channels), dist_reduce_fx="sum")
        self.add_state("count", default=torch.zeros(channels), dist_reduce_fx="sum")

    def update(self, imgs: torch.Tensor):
        """
        imgs: Tensor of shape [B, C, H, W]. NaNs are treated as masked-out.
        """
        # imgs is [B, C, H, W]
        finite = torch.isfinite(imgs)
        imgs_sanitized = torch.where(finite, imgs, torch.zeros_like(imgs))
        self.sum = self.sum + imgs_sanitized.sum(dim=(0, 2, 3))
        self.count = self.count + finite.to(imgs.dtype).sum(dim=(0, 2, 3))

    def compute(self):
        return self.sum / self.count  # Mean per channel


class MeanMetric_tilemeanmae(torchmetrics.Metric):
    def __init__(self, channels):
        super().__init__()
        self.add_state("sum", default=torch.zeros(channels), dist_reduce_fx="sum")
        self.add_state("count", default=torch.zeros(channels), dist_reduce_fx="sum")

    def update(self, imgs: torch.Tensor):
        """
        imgs: Tensor of shape [B, C, H, W]. NaNs are treated as masked-out during tile-mean calculation.
        """
        # imgs is [B, C, H, W]
        finite = torch.isfinite(imgs)
        imgs_sanitized = torch.where(finite, imgs, torch.zeros_like(imgs))
        per_sum = imgs_sanitized.sum(dim=(2, 3))  # [B, C]
        per_cnt = finite.to(imgs.dtype).sum(dim=(2, 3))  # [B, C]
        valid = per_cnt > 0
        # Avoid divide-by-zero
        per_mean = torch.zeros_like(per_sum)
        per_mean[valid] = per_sum[valid] / torch.clamp(per_cnt[valid], min=1e-6)
        per_mean[valid] = torch.abs(per_mean[valid])
        self.sum = self.sum + per_mean.sum(dim=0)
        self.count = self.count + valid.to(imgs.dtype).sum(dim=0)

    def compute(self):
        return self.sum / self.count  # Mean per channel


class MeanMetricSmoothL1(torchmetrics.Metric):
    """Compute per-channel smooth L1 loss (Huber loss) averaged over spatial dims and batch."""

    def __init__(self, channels, beta: float = 1.0):
        super().__init__()
        self.add_state("sum", default=torch.zeros(channels), dist_reduce_fx="sum")
        self.add_state("count", default=torch.zeros(channels), dist_reduce_fx="sum")
        self.beta = beta

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        pred, target: Tensors of shape [B, C, H, W]. NaNs are treated as masked-out.
        """
        finite = torch.isfinite(pred) & torch.isfinite(target)
        pred_clean = torch.where(finite, pred, torch.zeros_like(pred))
        target_clean = torch.where(finite, target, torch.zeros_like(target))
        loss = torch.nn.functional.smooth_l1_loss(
            pred_clean, target_clean, beta=self.beta, reduction="none"
        )
        loss_masked = torch.where(finite, loss, torch.zeros_like(loss))
        self.sum = self.sum + loss_masked.sum(dim=(0, 2, 3))
        self.count = self.count + finite.to(pred.dtype).sum(dim=(0, 2, 3))

    def compute(self):
        return self.sum / self.count.clamp(min=1)  # Mean per channel


class MeanMetricTilemeanSmoothL1(torchmetrics.Metric):
    """Compute per-channel tilemean smooth L1 loss: average over entire tile first, then compute loss."""

    def __init__(self, channels, beta: float = 1.0):
        super().__init__()
        self.add_state("sum", default=torch.zeros(channels), dist_reduce_fx="sum")
        self.add_state("count", default=torch.zeros(channels), dist_reduce_fx="sum")
        self.beta = beta

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        pred, target: Tensors of shape [B, C, H, W]. NaNs are handled via mask-aware averaging.
        """
        # pred, target are [B, C, H, W]
        finite = torch.isfinite(pred) & torch.isfinite(target)
        pred_clean = torch.where(finite, pred, torch.zeros_like(pred))
        target_clean = torch.where(finite, target, torch.zeros_like(target))
        # Compute per-sample, per-channel mean over spatial dims
        per_sum_pred = pred_clean.sum(dim=(2, 3))  # [B, C]
        per_sum_target = target_clean.sum(dim=(2, 3))  # [B, C]
        per_cnt = finite.to(pred.dtype).sum(dim=(2, 3))  # [B, C]
        valid = per_cnt > 0
        # Compute tile means
        pred_mean = torch.zeros_like(per_sum_pred)
        target_mean = torch.zeros_like(per_sum_target)
        pred_mean[valid] = per_sum_pred[valid] / per_cnt[valid].clamp(min=1e-6)
        target_mean[valid] = per_sum_target[valid] / per_cnt[valid].clamp(min=1e-6)
        # Compute smooth L1 on tile means
        loss = torch.nn.functional.smooth_l1_loss(
            pred_mean, target_mean, beta=self.beta, reduction="none"
        )  # [B, C]
        loss_masked = torch.where(valid, loss, torch.zeros_like(loss))
        self.sum = self.sum + loss_masked.sum(dim=0)
        self.count = self.count + valid.to(pred.dtype).sum(dim=0)

    def compute(self):
        return self.sum / self.count.clamp(min=1)  # Mean per channel


def validate(
    *,
    pipeline,
    loader,
    device,
    channel_index: pd.MultiIndex,
    output_dir: str = ".",
    min_samples: int = 1,
    nimg: int = 0,
    writer=None,
    postfix: str = "",
    pooled_loss_weights=None,
    num_steps: int = 1,
    multistep_training_mode: str = "final_only",
    lat_radians: np.ndarray | torch.Tensor | None = None,
    lon_radians: np.ndarray | torch.Tensor | None = None,
    index_is_latlon: bool | None = None,
):
    mae_meter = MeanMetric(len(channel_index)).to(device)
    mae_tilemean_meter = MeanMetric_tilemeanmae(len(channel_index)).to(device)
    mae_persistence_meter = MeanMetric(len(channel_index)).to(device)
    mae_persistence_tilemean_meter = MeanMetric_tilemeanmae(len(channel_index)).to(
        device
    )
    # Per-channel loss meters in normalized space
    loss_meter = MeanMetricSmoothL1(len(channel_index)).to(device)
    loss_tilemean_meter = MeanMetricTilemeanSmoothL1(len(channel_index)).to(device)
    test_loss = torchmetrics.MeanMetric().to(device)
    if pooled_loss_weights:
        try:
            sorted_sizes = sorted(int(k) for k in pooled_loss_weights.keys())
        except Exception:
            # fallback: try to sort as given if keys not castable
            sorted_sizes = sorted(pooled_loss_weights.keys())
        term_names = ["base"] + [f"pooled_{k}" for k in sorted_sizes]
    else:
        term_names = ["base"]
    term_loss_meters: dict[str, torchmetrics.MeanMetric] = {
        name: torchmetrics.MeanMetric(sync_on_compute=True).to(device)
        for name in term_names
    }

    if index_is_latlon is None:
        index_is_latlon = bool(getattr(pipeline.network, "_index_is_latlon", False))

    def _maybe_convert_index(index_in):
        if not index_is_latlon:
            return index_in
        if isinstance(index_in, dict):
            return index_in
        if lat_radians is None or lon_radians is None:
            raise ValueError(
                "lat_radians/lon_radians must be provided when index_is_latlon is True"
            )
        if not torch.is_tensor(index_in):
            index_in = torch.as_tensor(index_in)
        idx = index_in.long()
        lat_base = (
            lat_radians
            if torch.is_tensor(lat_radians)
            else torch.as_tensor(lat_radians)
        )
        lon_base = (
            lon_radians
            if torch.is_tensor(lon_radians)
            else torch.as_tensor(lon_radians)
        )
        idx_cpu = idx.detach().to("cpu")
        lat = torch.as_tensor(lat_base[idx_cpu], device=idx.device)
        lon = torch.as_tensor(lon_base[idx_cpu], device=idx.device)
        return {"lat": lat, "lon": lon}

    n_samples = 0
    iterator = iter(loader)
    while n_samples < min_samples:
        try:
            batch = next(iterator)
        except StopIteration:
            break

        # Unpack allowing optional forcing
        # num_steps=1 -> 4 elements, num_steps>1 -> 5 elements with stacked T dimension
        if len(batch) == 4:
            inputs, targets, index, _ = batch
            s_forcings = None
        elif len(batch) == 5:
            inputs, targets_all, index, _, s_forcings = batch
            # For multi-step: use final target for validation
            targets = targets_all[:, -1]  # [B, C, H, W] - target at t+T
        else:
            raise RuntimeError(f"Unexpected batch length {len(batch)} in validate")

        # Compute loss
        with torch.no_grad():
            pred_normed, target_normed = None, None
            index = _maybe_convert_index(index)
            if num_steps == 1:
                # Single-step: simple loss computation
                result = pipeline.get_loss(
                    inputs,
                    targets,
                    index,
                    return_details=True,
                    pooled_loss_weights=pooled_loss_weights,
                    return_next_state=True,
                    return_normalized=True,
                )
                loss_result, output_full, _, pred_normed, target_normed = result
            else:
                # Multi-step: use get_multistep_loss method with return_normalized
                result = pipeline.get_multistep_loss(
                    inputs=inputs,
                    targets_all=targets_all,
                    index=index,
                    s_forcings=s_forcings,
                    num_steps=num_steps,
                    multistep_training_mode=multistep_training_mode,
                    return_details=True,
                    pooled_loss_weights=pooled_loss_weights,
                    return_final_output=True,
                    return_normalized=True,
                )
                loss_result, output_full, pred_normed, target_normed = result

            # Parse loss_result for metrics
            if isinstance(loss_result, tuple) and len(loss_result) == 2:
                loss, loss_terms = loss_result
            else:
                loss = loss_result
                loss_terms = None
            test_loss.update(loss)
            if loss_terms is not None:
                for name in term_loss_meters.keys():
                    if name in loss_terms:
                        term_loss_meters[name].update(loss_terms[name])

            # Update per-channel loss meters with normalized pred/target
            if pred_normed is not None and target_normed is not None:
                loss_meter.update(pred_normed, target_normed)
                loss_tilemean_meter.update(pred_normed, target_normed)

        diff = output_full - targets
        mae_meter.update(torch.abs(diff))
        # Use 4D update to let metric compute tile-mean internally with NaN handling
        mae_tilemean_meter.update(diff)

        if pipeline.variables_forcing or pipeline.variables_diagnostic:
            # For pipelines that has forcing or diagnostic variables, need to extract prognostic variables from inputs and targets for persistence comparison
            inputs_prognostic = pipeline._extract_prognostic_from_input(inputs)
            targets_prognostic = pipeline._extract_prognostic_from_output(targets)

            persistence_diff_prognostic = targets_prognostic - inputs_prognostic
            targets_diagnostic = pipeline._extract_diagnostic_from_output(targets)
            persistence_diff_diagnostic = torch.full_like(targets_diagnostic, torch.nan)
            persistence_diff = torch.cat(
                [persistence_diff_prognostic, persistence_diff_diagnostic], dim=1
            )
        else:
            persistence_diff = targets - inputs

        mae_persistence_meter.update(torch.abs(persistence_diff))
        mae_persistence_tilemean_meter.update(persistence_diff)
        n_samples += diff.shape[0]

    mae = mae_meter.compute().cpu()
    mae_tilemean = mae_tilemean_meter.compute().cpu()
    mae_persistence = mae_persistence_meter.compute().cpu()
    mae_persistence_tilemean = mae_persistence_tilemean_meter.compute().cpu()
    # Per-channel loss in normalized space
    loss_per_channel = loss_meter.compute().cpu()
    loss_tilemean_per_channel = loss_tilemean_meter.compute().cpu()
    test_loss_value = test_loss.compute().item()

    frame = pd.DataFrame(
        {
            "mae": mae,
            "mae_persistence": mae_persistence,
            "mae_tilemean": mae_tilemean,
            "mae_persistence_tilemean": mae_persistence_tilemean,
        },
        index=channel_index,
    )
    out_frame = 1 - frame.mae / frame.mae_persistence
    median_score = out_frame.median()
    out_fram_tilemean = 1 - frame.mae_tilemean / frame.mae_persistence_tilemean
    median_score_tilemean = out_fram_tilemean.median()

    # Per-channel loss frame (normalized units)
    loss_frame = pd.DataFrame(
        {
            "loss": loss_per_channel,
            "loss_tilemean": loss_tilemean_per_channel,
        },
        index=channel_index,
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.abspath(
            os.path.join(output_dir, f"tendency_scores{postfix}.csv")
        )
        logging.getLogger(__name__).info(f"Writing scores to {path}")
        frame.to_csv(path)

        # Save per-channel loss to separate CSV
        loss_path = os.path.abspath(
            os.path.join(output_dir, f"per_channel_loss{postfix}.csv")
        )
        logging.getLogger(__name__).info(f"Writing per-channel loss to {loss_path}")
        loss_frame.to_csv(loss_path)

    computed_term_values: dict[str, float] = {}
    if term_loss_meters:
        for name, meter in term_loss_meters.items():
            computed_term_values[name] = meter.compute().item()

    if writer is not None:
        writer.add_scalar(f"test_loss{postfix}", test_loss_value, global_step=nimg)
        # Log averaged per-term test losses (including base and pooled terms if provided)
        for name, avg_val in computed_term_values.items():
            writer.add_scalar(f"test_loss{postfix}/{name}", avg_val, global_step=nimg)
        writer.add_scalar(f"median_score{postfix}", median_score, global_step=nimg)
        writer.add_scalar(
            f"median_score_tilemean{postfix}", median_score_tilemean, global_step=nimg
        )

    return test_loss.compute().item(), median_score, median_score_tilemean


def compute_split_shapes(size: int, num_chunks: int) -> List[int]:
    # treat trivial case first
    if num_chunks == 1:
        return [size]

    # first, check if we can split using div-up to balance the load:
    chunk_size = (size + num_chunks - 1) // num_chunks
    last_chunk_size = max(0, size - chunk_size * (num_chunks - 1))
    if last_chunk_size == 0:
        # in this case, the last shard would be empty, split with floor instead:
        chunk_size = size // num_chunks
        last_chunk_size = size - chunk_size * (num_chunks - 1)

    # generate sections list
    sections = [chunk_size for _ in range(num_chunks - 1)] + [last_chunk_size]

    return sections


def split_tensor_along_dim(tensor, dim, num_chunks):
    if dim >= tensor.dim():
        raise ValueError(
            f"Error, tensor dimension is {tensor.dim()} which cannot be split along {dim}"
        )
    if tensor.shape[dim] < num_chunks:
        raise ValueError(
            "Error, cannot split dim {dim} of size {tensor.shape[dim]} into \
        {num_chunks} chunks. Empty slices are currently not supported."
        )

    # get split
    sections = compute_split_shapes(tensor.shape[dim], num_chunks)
    tensor_list = torch.split(tensor, sections, dim=dim)

    return tensor_list


def scatter_patches(patches, group=None, dtype=torch.float32):
    comm_size = dist.get_world_size(group=group)
    comm_rank = dist.get_rank(group=group)
    device = torch.cuda.current_device()
    if comm_size > 1:
        if comm_rank == 0:
            patch_split = list(split_tensor_along_dim(patches, 0, comm_size))
            sections = compute_split_shapes(patches.shape[0], comm_size)
            split_shapes = [
                torch.tensor(
                    [
                        sections[i],
                    ]
                    + list(patches.shape[1:]),
                    device=device,
                )
                for i in range(comm_size)
            ]
        else:
            patch_split = None
            split_shapes = None
        split_shape = torch.empty(4, dtype=torch.int64, device=device)
        # Scatter each rank's size first
        dist.scatter(split_shape, split_shapes, src=0, group=group)
        split_shape = list(split_shape.cpu().numpy())

        # Create and scatter actual patch data
        local_patches = torch.empty(split_shape, dtype=dtype, device=device)
        dist.scatter(local_patches, patch_split, src=0, group=group)
    else:
        local_patches = patches

    return local_patches


def gather_patches(local_patches, group=None, dtype=torch.float32):
    comm_size = dist.get_world_size(group=group)
    comm_rank = dist.get_rank(group=group)
    device = torch.cuda.current_device()
    if comm_size > 1:
        split_shape = torch.tensor(local_patches.shape, device=device)
        if comm_rank == 0:
            split_shapes = [
                torch.empty(4, dtype=torch.int64, device=device)
                for i in range(comm_size)
            ]
        else:
            split_shapes = None
        dist.gather(split_shape, split_shapes, dst=0, group=group)

        if comm_rank == 0:
            patch_split = [
                torch.empty(list(shape.cpu().numpy()), dtype=dtype, device=device)
                for shape in split_shapes
            ]
        else:
            patch_split = None

        dist.gather(local_patches, patch_split, dst=0, group=group)
        patches = torch.cat(patch_split, dim=0) if comm_rank == 0 else None
    else:
        patches = local_patches

    return patches


def broadcast_scalar(value, src=0, group=None):
    data = torch.tensor(value, device=torch.cuda.current_device())
    dist.broadcast(data, src=src, group=group)
    return data.item()
