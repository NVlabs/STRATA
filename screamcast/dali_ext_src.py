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
import logging
import os
import random
import time

import dotenv
import earth2grid
import einops
import numpy as np
import pandas as pd
import torch
import xarray as xr
import zarr

from screamcast import zarr_helper
from screamcast.cubesphere_transforms import (
    create_padded_faces_batched,
    reorder_cubesphere_to_2d_tensor,
    unstructured_to_6faces,
    unstructured_to_padded_faces_knn,
)
from screamcast.storage import get_s3fs_options_clone_config

dotenv.load_dotenv()


def open_consolidated_zarr_group(zarr_path: str, storage_env: str):
    """Open a consolidated Zarr group with optional S3 storage options."""
    if "s3://" in zarr_path:
        return zarr.open_consolidated(
            zarr_path,
            storage_options=get_s3fs_options_clone_config(storage_env),
        )
    return zarr.open_consolidated(zarr_path)


def spread_bits(bits):
    """
    bits is a 32 bit number (stored in int64)
    algorithm starts by moving the first 16 bits to the left by 16
    and proceeding recursively
    """
    # example implementation for a byte
    # 0000abcd
    # 00ab00cd # (x | x <<2)  & 00110011 = 0x33
    # 0a0b0c0d # (x | x <<1)  & 01010100 = 0x55
    # --------
    # abc0d
    bits = (bits | (bits << 16)) & 0x0000FFFF0000FFFF  # noqa
    bits = (bits | (bits << 8)) & 0x00FF00FF00FF00FF  # noqa
    bits = (bits | (bits << 4)) & 0x0F0F0F0F0F0F0F0F  # noqa
    bits = (bits | (bits << 2)) & 0x3333333333333333  # noqa
    bits = (bits | (bits << 1)) & 0x5555555555555555  # noqa
    return bits


def reorder_morton_to_hpx_pad(array, shape):
    other_dims = array.shape[:-1]
    i = torch.arange(array.size(-1), dtype=torch.int64, device=array.device)
    x = i % shape[-1]
    y = i // shape[-1]

    # hpx pad convention
    # (0, 0) is the north in hpx pad
    # (0, +1) point to SE

    # current convention
    # (0, 0) is the S now
    # (0, +1) point to NE

    # flip
    x, y = y, x

    # rotate 180
    y = -y - 1 % shape[-2]
    x = -x - 1 % shape[-1]

    i_morton = spread_bits(x) | (spread_bits(y) << 1)
    array = array[..., i_morton]
    return array.reshape((*other_dims, *shape))


def shuffle(x: list):
    rng = random.Random(0)
    x = x.copy()
    rng.shuffle(x)
    return x


def output_channels(config) -> list[str]:
    """Return flattened output channel names for a TrainConfig."""
    data = config.data
    levels = list(range(data.level_start, data.level_end, data.plevel))
    variables_dimensions = ScreamV2.get_default_variables_dimensions()
    names: list[str] = []
    for variable in data.variables_prognostic + data.variables_diagnostic:
        if variables_dimensions[variable] == "3D":
            names.extend(f"{variable}_{level}" for level in levels)
        else:
            names.append(variable)
    return names


class ScreamV2:
    def __init__(
        self,
        batch_size,
        split="",
        num_shards=1,
        shard_id=0,
        mock: bool = False,
        plevel: int = 4,
        level_start: int = 3,
        level_end: int = 128,
        num_steps: int = 1,
        load_final_target_only: bool = True,
        variables_prognostic: tuple = (
            "PotentialTemperature",
            "U",
            "V",
            "z_mid",
            "omega",
            "qv",
            "T_2m",
        ),
        variables_forcing: tuple = ("coszr", "phis"),
        variables_diagnostic: tuple = (
            "precip_ice_surf_mass_flux",
            "precip_liq_surf_mass_flux",
        ),
        main_zarr_path: str = None,
        aux_zarr_path: str = None,
        resolution: int = 1024,
        nside: int = 1024,
        use_time_variation_seed: bool = False,
        use_fixed_seed: bool = False,
        split_by_time: bool = None,
        train_start_index: int = 0,
        train_end_index: int = None,
        test_start_index: int = None,
        test_end_index: int = None,
        test_stride: int = 1,
        grid_type: str = "healpix",
        cubesphere_ne: int = 1024,
        cubesphere_npg: int = 2,
        index_is_latlon: bool = False,
        latlon_path: str | None = None,
    ):
        storage_env = os.getenv("SCREAM_ZARR_PROFILE", "")
        self.batch_size = batch_size
        self.plevel = plevel
        self.level_start = level_start
        self.level_end = level_end
        self.variables_prognostic = variables_prognostic
        self.variables_forcing = variables_forcing
        self.variables_diagnostic = variables_diagnostic
        self.variables_input = self.variables_prognostic + self.variables_forcing
        self.variables_output = self.variables_prognostic + self.variables_diagnostic
        self.variables = (
            self.variables_prognostic
            + self.variables_forcing
            + self.variables_diagnostic
        )
        # Control multistep loading:
        # num_steps=1: input at t, output at t+1 (single step)
        # num_steps=n: input at t, forcing at t+1..t+n-1, output at t+1..t+n
        # load_final_target_only=True: only load target at t+n (not t+1..t+n)
        self.num_steps = max(1, int(num_steps))
        self.load_final_target_only = load_final_target_only
        self.grid_type = (grid_type or "healpix").lower()
        if self.grid_type not in {"healpix", "cubesphere"}:
            raise ValueError(
                f"Unsupported grid_type='{grid_type}'. Expected 'healpix' or 'cubesphere'."
            )
        # Grid geometry
        # - healpix: keep historical behavior (respect provided resolution/nside)
        # - cubesphere: user only needs to set grid_type; we overwrite resolution/nside
        #   using cubesphere_ne/cubesphere_npg (defaults: ne=1024, npg=2 -> 2048).
        if self.grid_type == "cubesphere":
            self.cubesphere_ne = int(cubesphere_ne)
            self.cubesphere_npg = int(cubesphere_npg)
            if self.cubesphere_ne <= 0 or self.cubesphere_npg <= 0:
                raise ValueError(
                    f"cubesphere_ne and cubesphere_npg must be positive, got ne={self.cubesphere_ne}, npg={self.cubesphere_npg}"
                )
            self.resolution = int(self.cubesphere_ne * self.cubesphere_npg)
            self.nside = int(self.resolution)
        else:
            self.cubesphere_ne = None
            self.cubesphere_npg = None
            self.resolution = resolution
            self.nside = nside

        self.main_zarr_path = main_zarr_path or os.getenv("SCREAM_MAIN_ZARR_PATH")
        self.aux_zarr_path = aux_zarr_path or os.getenv("SCREAM_AUX_ZARR_PATH")
        if not self.main_zarr_path:
            raise ValueError(
                "No main Zarr path configured. Pass main_zarr_path or set the "
                "SCREAM_MAIN_ZARR_PATH environment variable."
            )
        if not self.aux_zarr_path:
            raise ValueError(
                "No auxiliary Zarr path configured. Pass aux_zarr_path or set the "
                "SCREAM_AUX_ZARR_PATH environment variable."
            )
        self.split_by_time = split_by_time
        self.train_start_index = train_start_index
        self.train_end_index = train_end_index
        self.test_start_index = test_start_index
        self.test_end_index = test_end_index
        self.test_stride = test_stride

        self.list_of_zarr_file = (self.main_zarr_path, self.aux_zarr_path)
        self.patch_size = self.resolution * self.resolution

        # using consolidated metadata avoids I/O for array open when run from
        # parallel in many jobs/data-workers this led to I/O failures
        # This reduces metadata lookups to 1 per rank per training.
        group_main = open_consolidated_zarr_group(self.main_zarr_path, storage_env)
        if self.aux_zarr_path != self.main_zarr_path:
            group_aux = open_consolidated_zarr_group(self.aux_zarr_path, storage_env)
        else:
            group_aux = group_main

        if mock:
            # For mock mode, clone each zarr group in the dictionary
            mock_group_main = zarr_helper.clone_zarr_group_metadata(group_main, {})
            mock_group_aux = zarr_helper.clone_zarr_group_metadata(group_aux, {})
            group_main = mock_group_main
            group_aux = mock_group_aux

        self.mock = mock
        self.group_main = group_main
        self.group_aux = group_aux

        self.nt, self.level, self.cells = self.group_main["U"].shape
        self.levels = np.r_[level_start : level_end : self.plevel]
        num_of_levels = len(self.levels)
        self.n_levels = num_of_levels

        def get_num_of_channels(variables):
            num_of_channels = 0
            for v in variables:
                num_of_channels += (
                    num_of_levels
                    if self.get_default_variables_dimensions()[v] == "3D"
                    else 1
                )
            return num_of_channels

        self.in_channels = get_num_of_channels(self.variables_input)
        self.out_channels = get_num_of_channels(self.variables_output)
        self.prognostic_channels = get_num_of_channels(self.variables_prognostic)
        self.diagnostic_channels = get_num_of_channels(self.variables_diagnostic)
        self.forcing_channels = get_num_of_channels(self.variables_forcing)
        self.npatches = self.cells // self.patch_size
        self._last_patch = (None, None)

        self.index = self._initial_index()

        if self.split_by_time:
            self.set_split_by_time(split)
        else:
            self.set_split(split)
        self.n_samples_total = self.__len__()
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.n_samples_shard = self.n_samples_total // self.num_shards
        self.full_iterations = self.n_samples_shard // self.batch_size
        self.shard_offset = self.shard_id * self.n_samples_shard
        self.perm = None
        self.last_seen_epoch = None
        self.use_time_variation_seed = use_time_variation_seed
        self.use_fixed_seed = use_fixed_seed

        # Precompute lat/lon arrays for index_is_latlon mode
        self.index_is_latlon = index_is_latlon
        self._latlon_radians = None
        if self.index_is_latlon:
            if self.grid_type == "healpix":
                # Use earth2grid to get lat/lon for HEALPix
                level = earth2grid.healpix.nside2level(self.nside)
                src_grid = earth2grid.healpix.Grid(
                    level=level, pixel_order=earth2grid.healpix.PixelOrder.NEST
                )
                lat_deg = np.array(src_grid.lat, dtype=np.float32)
                lon_deg = np.array(src_grid.lon, dtype=np.float32)
                # Convert to radians
                self._latlon_radians = (
                    torch.from_numpy(np.deg2rad(lat_deg)),
                    torch.from_numpy(np.deg2rad(lon_deg)),
                )
            elif self.grid_type == "cubesphere":
                # Load from latlon_path for cubesphere
                if not latlon_path:
                    raise ValueError(
                        "latlon_path is required for index_is_latlon=True "
                        "with cubesphere grid_type."
                    )
                with xr.open_dataset(latlon_path) as ds_ll:
                    lat_deg = ds_ll["lat"].values.astype(np.float32)
                    lon_deg = ds_ll["lon"].values.astype(np.float32)
                # Convert to radians
                self._latlon_radians = (
                    torch.from_numpy(np.deg2rad(lat_deg)),
                    torch.from_numpy(np.deg2rad(lon_deg)),
                )
            else:
                raise ValueError(
                    f"Unsupported grid_type for index_is_latlon: {self.grid_type}"
                )

    @property
    def variables_source_zarr_file(self):
        return {
            "PotentialTemperature": self.main_zarr_path,
            "T_2m": self.main_zarr_path,
            "U": self.main_zarr_path,
            "V": self.main_zarr_path,
            "geopotential_mid": self.main_zarr_path,
            "z_mid": self.main_zarr_path,
            "omega": self.main_zarr_path,
            "qv": self.main_zarr_path,
            "ps": self.main_zarr_path,
            "precip_ice_surf_mass_flux": self.main_zarr_path,
            "precip_liq_surf_mass_flux": self.main_zarr_path,
            "coszr": self.aux_zarr_path,
            "sst": self.aux_zarr_path,
            "phis": self.aux_zarr_path,
            "sgh30": self.aux_zarr_path,
            "landfrac": self.aux_zarr_path,
            "icefrac": self.aux_zarr_path,
            "diag_equiv_reflectivity": self.main_zarr_path,
            "qc": self.main_zarr_path,
            "qr": self.main_zarr_path,
            "qi": self.main_zarr_path,
        }

    @classmethod
    def get_default_variables_dimensions(cls):
        """
        Class method to get default variable dimensionality for other class methods
        For the definition of dimension: 1D (ncol), 2D (ntime, ncol), 3D (ntime, nlev, ncol)
        """
        return {
            "PotentialTemperature": "3D",
            "T_2m": "2D",
            "U": "3D",
            "V": "3D",
            "geopotential_mid": "3D",
            "z_mid": "3D",
            "omega": "3D",
            "qv": "3D",
            "ps": "2D",
            "precip_ice_surf_mass_flux": "2D",
            "precip_liq_surf_mass_flux": "2D",
            "coszr": "2D",
            "sst": "2D",
            "phis": "1D",
            "sgh30": "1D",
            "landfrac": "1D",
            "icefrac": "2D",
            "diag_equiv_reflectivity": "3D",
            "qc": "3D",
            "qr": "3D",
            "qi": "3D",
        }

    @classmethod
    def num_of_input_channels(
        cls,
        variables_prognostic,
        variables_forcing,
        plevel,
        level_start,
        level_end,
    ):
        levels = np.r_[level_start:level_end:plevel]
        num_of_levels = len(levels)
        num_of_channels = 0
        variables_dimensions = cls.get_default_variables_dimensions()
        for v in variables_prognostic + variables_forcing:
            num_of_channels += num_of_levels if variables_dimensions[v] == "3D" else 1
        return num_of_channels

    @classmethod
    def num_of_output_channels(
        cls,
        variables_prognostic,
        variables_diagnostic,
        plevel,
        level_start,
        level_end,
    ):
        levels = np.r_[level_start:level_end:plevel]
        num_of_levels = len(levels)
        num_of_channels = 0
        variables_dimensions = cls.get_default_variables_dimensions()
        for v in variables_prognostic + variables_diagnostic:
            num_of_channels += num_of_levels if variables_dimensions[v] == "3D" else 1
        return num_of_channels

    @classmethod
    def ranges_input(
        cls,
        variables_prognostic=(
            "PotentialTemperature",
            "U",
            "V",
            "geopotential_mid",
            "omega",
            "qv",
            "T_2m",
        ),
        variables_forcing=("coszr", "sst", "phis"),
        plevel=4,
        level_start=3,
        level_end=128,
    ):
        """Returns channel ranges for input (prognostic + forcing variables)"""
        ranges = {}
        variables_input = variables_prognostic + variables_forcing
        variables_dimensions = cls.get_default_variables_dimensions()
        i = 0
        levels = np.r_[level_start:level_end:plevel]
        num_of_levels = len(levels)
        for v in variables_input:
            if variables_dimensions[v] != "3D":
                ranges[v] = slice(i, i + 1)
                i += 1
            else:
                ranges[v] = slice(i, i + num_of_levels)
                i += num_of_levels
        return ranges

    @classmethod
    def ranges_output(
        cls,
        variables_prognostic=(
            "PotentialTemperature",
            "U",
            "V",
            "geopotential_mid",
            "omega",
            "qv",
            "T_2m",
        ),
        variables_diagnostic=("precip_ice_surf_mass_flux", "precip_liq_surf_mass_flux"),
        plevel=4,
        level_start=3,
        level_end=128,
    ):
        """Returns channel ranges for output (prognostic + diagnostic variables)"""
        ranges = {}
        variables_output = variables_prognostic + variables_diagnostic
        variables_dimensions = cls.get_default_variables_dimensions()
        i = 0
        levels = np.r_[level_start:level_end:plevel]
        num_of_levels = len(levels)
        for v in variables_output:
            if variables_dimensions[v] != "3D":
                ranges[v] = slice(i, i + 1)
                i += 1
            else:
                ranges[v] = slice(i, i + num_of_levels)
                i += num_of_levels
        return ranges

    def channel_index_input(self):
        """Returns channel index for input (prognostic + forcing variables)"""
        channels = []
        for v in self.variables_input:
            ndim = self.get_default_variables_dimensions()[v]  # "3D" or "2D" or "1D"
            if ndim == "3D":
                channels.extend(
                    [
                        (v, level)
                        for level in range(
                            self.level_start, self.level_end, self.plevel
                        )
                    ]
                )
            elif ndim == "2D":
                channels.append((v, None))
            elif ndim == "1D":
                channels.append((v, None))

        return pd.MultiIndex.from_tuples(channels, names=["name", "level"])

    def channel_index_output(self):
        """Returns channel index for output (prognostic + diagnostic variables)"""
        channels = []
        for v in self.variables_output:
            ndim = self.get_default_variables_dimensions()[v]
            if ndim == "3D":
                channels.extend(
                    [
                        (v, level)
                        for level in range(
                            self.level_start, self.level_end, self.plevel
                        )
                    ]
                )
            elif ndim == "2D":
                channels.append((v, None))
            elif ndim == "1D":
                channels.append((v, None))

        return pd.MultiIndex.from_tuples(channels, names=["name", "level"])

    def __len__(self):
        return len(self.index)

    def _post_process(self, x):
        xt = torch.as_tensor(x)
        if self.grid_type == "healpix":
            return reorder_morton_to_hpx_pad(
                xt, shape=(self.resolution, self.resolution)
            )
        elif self.grid_type == "cubesphere":
            return reorder_cubesphere_to_2d_tensor(
                xt, ne=self.cubesphere_ne, npg=self.cubesphere_npg
            )
        else:
            raise RuntimeError(f"Unhandled grid_type={self.grid_type!r}")

    def _initial_index(self):
        # For num_steps loading, we need t+num_steps to be valid -> t in [1, nt-num_steps-1]
        # time_len = nt - num_steps - 1
        time_len = max(0, self.nt - self.num_steps - 1)
        return list(range(time_len * self.npatches))

    def set_split(self, split):
        index = self._initial_index()
        n_test = len(index) // 20
        index_shuffled = shuffle(index)
        splits = {}
        splits["test"] = sorted(index_shuffled[:n_test])
        splits["train"] = sorted(index_shuffled[n_test:])
        splits[""] = self.index
        self.index = splits[split]

    def set_split_by_time(self, split):
        if self.split_by_time and (
            self.train_start_index is None
            or self.train_end_index is None
            or self.test_start_index is None
            or self.test_end_index is None
            or self.test_stride is None
        ):
            raise ValueError(
                "train_start_index, train_end_index, test_start_index, test_end_index, and test_stride must be set when split_by_time is True"
            )
        splits = {}

        # Valid time range: [1, time_len] where time_len = nt - num_steps - 1
        # - Lower bound t=1: Skip t=0 because the first time slice in zarr may have NaN/zero values
        # - Upper bound: For num_steps loading, we need t+num_steps to be valid (< nt)
        # Index structure: j = patch * time_len + (t - 1)
        time_len = max(1, self.nt - self.num_steps - 1)

        # Build train index: select ALL faces for each time step in [train_start_index, train_end_index)
        index_train = []
        for t in range(self.train_start_index, self.train_end_index):
            # Ensure t is within valid range [1, time_len]
            if 1 <= t <= time_len:
                # Add indices for all patches at this time step
                for patch in range(self.npatches):
                    j = patch * time_len + (t - 1)
                    index_train.append(j)
        splits["train"] = sorted(index_train)

        # Build test index: select ALL faces for each strided time step
        index_test = []
        for t in range(self.test_start_index, self.test_end_index, self.test_stride):
            # Ensure t is within valid range [1, time_len]
            if 1 <= t <= time_len:
                # Add indices for all patches at this time step
                for patch in range(self.npatches):
                    j = patch * time_len + (t - 1)
                    index_test.append(j)
        splits["test"] = index_test
        splits[""] = self.index
        self.index = splits[split]

    def __call__(self, batch_info):
        logging.info(batch_info)
        iteration = batch_info.iteration
        if iteration >= self.full_iterations:
            raise StopIteration

        if self.last_seen_epoch != batch_info.epoch_idx:
            self.last_seen_epoch = batch_info.epoch_idx
            if self.use_fixed_seed:
                seed = 42
            else:
                if self.use_time_variation_seed:
                    time_variation = int(time.time()) % 100000
                    seed = 42 + batch_info.epoch_idx + time_variation
                else:
                    seed = 42 + batch_info.epoch_idx
            self.perm = np.random.default_rng(seed=seed)
            self.perm = self.perm.permutation(self.n_samples_total)

        s0 = []
        s_outputs = []  # will be stacked to [T, C, H, W] per sample
        s_forcings = (
            []
        )  # will be stacked to [T-1, C, H, W] per sample (if num_steps > 1)
        indices = []
        js = []

        for b in range(self.batch_size):
            sample = self.perm[self.shard_offset + iteration * self.batch_size + b]
            j = self.index[sample]
            # Map flat index to time 't' respecting multistep regime
            # For num_steps loading: t in [1, nt - num_steps - 1]
            time_len = max(1, self.nt - self.num_steps - 1)
            t = j % time_len + 1
            patch = j // time_len

            # Use different loading methods for input vs target
            inp_patch = self.load_patch_input(t, patch)
            inp_patch = self._post_process(inp_patch)

            # Load outputs at t+1, t+2, ..., t+num_steps and stack along T dimension
            # If load_final_target_only=True, only load the final target at t+num_steps
            tar_patches = []
            if self.load_final_target_only:
                # Only load final target
                tar_patch = self.load_patch_output(t + self.num_steps, patch)
                tar_patch = self._post_process(tar_patch)
                tar_patches.append(tar_patch)
            else:
                # Load all targets
                for step in range(self.num_steps):
                    tar_patch = self.load_patch_output(t + step + 1, patch)
                    tar_patch = self._post_process(tar_patch)
                    tar_patches.append(tar_patch)
            # Stack to [T, C, H, W] (T=1 if load_final_target_only=True)
            stacked_outputs = torch.stack(tar_patches, dim=0)
            s_outputs.append(stacked_outputs)

            # Load forcings at t+1, t+2, ..., t+num_steps-1 (if num_steps > 1)
            if self.num_steps > 1:
                force_patches = []
                for step in range(self.num_steps - 1):
                    force_patch = self.load_patch_forcing(t + step + 1, patch)
                    force_patch = self._post_process(force_patch)
                    force_patches.append(force_patch)
                # Stack to [T-1, C, H, W]
                stacked_forcings = torch.stack(force_patches, dim=0)
                s_forcings.append(stacked_forcings)

            start = patch * self.patch_size
            size = self.patch_size

            if self.index_is_latlon:
                # Build [2, H, W] lat/lon index for this patch from preloaded 1D arrays.
                lat_rad, lon_rad = self._latlon_radians
                patch_lat = self._post_process(lat_rad[start : start + size])
                patch_lon = self._post_process(lon_rad[start : start + size])
                index = torch.stack([patch_lat, patch_lon], dim=0)
            else:
                index = np.r_[start : start + size]
                index = self._post_process(index)

            s0.append(inp_patch)
            indices.append(index)
            js.append(torch.tensor([j], dtype=torch.int64))

        # Return format (all torch tensors):
        # For num_steps=1: s0, s1, indices, js (backwards compatible)
        # For num_steps>1: s0, s_outputs [T,C,H,W], indices, js, s_forcings [T-1,C,H,W]
        if self.num_steps == 1:
            # Squeeze the T dimension for backwards compatibility
            return s0, [s.squeeze(0) for s in s_outputs], indices, js
        else:
            return s0, s_outputs, indices, js, s_forcings

    def get_var(self, v, t, start, size):
        # Determine which group to use
        if self.variables_source_zarr_file[v] == self.main_zarr_path:
            source_group = self.group_main
        elif self.variables_source_zarr_file[v] == self.aux_zarr_path:
            source_group = self.group_aux
        else:
            raise ValueError(
                f"Unsupported source zarr file for variable {v}: {self.variables_source_zarr_file[v]}"
            )

        if v not in source_group:
            raise ValueError(f"Variable {v} not found in data source")

        ndim = self.get_default_variables_dimensions()[v]  # "3D" or "2D" or "1D"

        if ndim == "1D":
            return source_group[v][start : start + size][None]
        elif ndim == "2D":
            return source_group[v][t, start : start + size][None]
        elif ndim == "3D":
            return source_group[v][
                t,
                self.level_start : self.level_end : self.plevel,
                start : start + size,
            ]
        else:
            raise ValueError(f"Unsupported dimensionality for variable {v}: {ndim}")

    def _assemble_patch_vars(
        self, t: int, start: int, size: int, variables: tuple
    ) -> np.ndarray:
        return np.concatenate([self.get_var(v, t, start, size) for v in variables])

    def load_patch_input(self, t, patch):
        """Load patch with prognostic and forcing variables for input"""
        if self.mock:
            return np.random.normal(size=(self.in_channels, self.patch_size)).astype(
                np.float32
            )

        start = patch * self.patch_size
        size = self.patch_size
        return self._assemble_patch_vars(t, start, size, self.variables_input)

    def load_patch_output(self, t, patch):
        """Load patch with prognostic and diagnostic variables for output"""
        if self.mock:
            return np.random.normal(size=(self.out_channels, self.patch_size)).astype(
                np.float32
            )

        start = patch * self.patch_size
        size = self.patch_size
        return self._assemble_patch_vars(t, start, size, self.variables_output)

    def load_patch_prognostic(self, t, patch):
        """Load patch with prognostic variables"""
        if self.mock:
            return np.random.normal(
                size=(self.prognostic_channels, self.patch_size)
            ).astype(np.float32)

        start = patch * self.patch_size
        size = self.patch_size

        return self._assemble_patch_vars(t, start, size, self.variables_prognostic)

    def load_patch_diagnostic(self, t, patch):
        """Load patch with diagnostic variables"""
        if self.mock:
            return np.random.normal(
                size=(self.diagnostic_channels, self.patch_size)
            ).astype(np.float32)

        start = patch * self.patch_size
        size = self.patch_size

        return self._assemble_patch_vars(t, start, size, self.variables_diagnostic)

    def load_patch_forcing(self, t, patch):
        """Load patch with forcing variables"""
        if self.mock:
            return np.random.normal(
                size=(self.forcing_channels, self.patch_size)
            ).astype(np.float32)

        start = patch * self.patch_size
        size = self.patch_size

        return self._assemble_patch_vars(t, start, size, self.variables_forcing)


class MultiScreamV2:
    """Multi-source wrapper for ScreamV2 that mixes samples across datasets."""

    def __init__(
        self,
        *,
        batch_size: int,
        split: str,
        num_shards: int,
        shard_id: int,
        mock: bool = False,
        plevel: int = 4,
        level_start: int = 3,
        level_end: int = 128,
        num_steps: int = 1,
        load_final_target_only: bool = True,
        variables_prognostic: tuple = (
            "PotentialTemperature",
            "U",
            "V",
            "z_mid",
            "omega",
            "qv",
            "T_2m",
        ),
        variables_forcing: tuple = ("coszr", "phis"),
        variables_diagnostic: tuple = (
            "precip_ice_surf_mass_flux",
            "precip_liq_surf_mass_flux",
        ),
        main_zarr_paths: tuple[str, ...] | list[str] = (),
        aux_zarr_paths: tuple[str, ...] | list[str] | None = None,
        zarr_weights: tuple[float, ...] | list[float] | None = None,
        resolution: int = 1024,
        nside: int = 1024,
        use_time_variation_seed: bool = False,
        use_fixed_seed: bool = False,
        split_by_time: bool | None = None,
        train_start_index: int = 0,
        train_end_index: int | None = None,
        train_start_indices: tuple[int, ...] | list[int] | None = None,
        train_end_indices: tuple[int | None, ...] | list[int | None] | None = None,
        test_start_index: int | None = None,
        test_end_index: int | None = None,
        test_stride: int = 1,
        grid_type: str = "healpix",
        cubesphere_ne: int = 1024,
        cubesphere_npg: int = 2,
        index_is_latlon: bool = False,
        latlon_path: str | None = None,
    ):
        if not main_zarr_paths:
            raise ValueError("main_zarr_paths must contain at least one Zarr path")
        if train_start_indices is not None and len(train_start_indices) != len(
            main_zarr_paths
        ):
            raise ValueError("train_start_indices must match main_zarr_paths length")
        if train_end_indices is not None and len(train_end_indices) != len(
            main_zarr_paths
        ):
            raise ValueError("train_end_indices must match main_zarr_paths length")
        if aux_zarr_paths is not None and len(aux_zarr_paths) != len(main_zarr_paths):
            raise ValueError("aux_zarr_paths must match main_zarr_paths length")
        if zarr_weights is not None and len(zarr_weights) != len(main_zarr_paths):
            raise ValueError("zarr_weights must match main_zarr_paths length")

        self.sources = []
        for idx, main_path in enumerate(main_zarr_paths):
            aux_path = aux_zarr_paths[idx] if aux_zarr_paths is not None else None
            per_source_train_start = (
                train_start_indices[idx]
                if train_start_indices is not None
                else train_start_index
            )
            per_source_train_end = (
                train_end_indices[idx]
                if train_end_indices is not None
                else train_end_index
            )
            src = ScreamV2(
                batch_size=batch_size,
                split=split,
                num_shards=1,
                shard_id=0,
                mock=mock,
                plevel=plevel,
                level_start=level_start,
                level_end=level_end,
                num_steps=num_steps,
                load_final_target_only=load_final_target_only,
                variables_prognostic=variables_prognostic,
                variables_forcing=variables_forcing,
                variables_diagnostic=variables_diagnostic,
                main_zarr_path=main_path,
                aux_zarr_path=aux_path,
                resolution=resolution,
                nside=nside,
                use_time_variation_seed=use_time_variation_seed,
                use_fixed_seed=use_fixed_seed,
                split_by_time=split_by_time,
                train_start_index=per_source_train_start,
                train_end_index=per_source_train_end,
                test_start_index=test_start_index,
                test_end_index=test_end_index,
                test_stride=test_stride,
                grid_type=grid_type,
                cubesphere_ne=cubesphere_ne,
                cubesphere_npg=cubesphere_npg,
                index_is_latlon=index_is_latlon,
                latlon_path=latlon_path,
            )
            self.sources.append(src)
            logging.info(
                "MultiScreamV2 source %d: main_zarr_path=%s aux_zarr_path=%s "
                "train_start=%s train_end=%s",
                idx,
                main_path,
                aux_path or main_path,
                per_source_train_start,
                per_source_train_end,
            )

        self.batch_size = batch_size
        self.split = split
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.mock = mock
        self.use_time_variation_seed = use_time_variation_seed
        self.use_fixed_seed = use_fixed_seed

        self._validate_compatible_sources()
        self._init_common_properties()

        self.zarr_weights = (
            [float(w) for w in zarr_weights] if zarr_weights is not None else None
        )
        if self.zarr_weights is not None:
            if any(w < 0 for w in self.zarr_weights):
                raise ValueError("zarr_weights must be non-negative")
            if sum(self.zarr_weights) == 0:
                raise ValueError("zarr_weights must sum to a positive value")

        self.n_samples_total_by_source = [src.n_samples_total for src in self.sources]
        if any(n <= 0 for n in self.n_samples_total_by_source):
            raise ValueError("Each source must have at least one sample")
        self.n_samples_total = int(sum(self.n_samples_total_by_source))
        self.n_samples_shard = self.n_samples_total // self.num_shards
        self.full_iterations = self.n_samples_shard // self.batch_size
        self.shard_offset = self.shard_id * self.n_samples_shard
        self.last_seen_epoch = None
        self._epoch_samples = None

    def _validate_compatible_sources(self):
        base = self.sources[0]
        compare_attrs = [
            "grid_type",
            "nside",
            "resolution",
            "plevel",
            "level_start",
            "level_end",
            "num_steps",
            "load_final_target_only",
            "variables_prognostic",
            "variables_forcing",
            "variables_diagnostic",
            "patch_size",
            "npatches",
            "in_channels",
            "out_channels",
            "n_levels",
        ]
        for src in self.sources[1:]:
            for attr in compare_attrs:
                if getattr(src, attr) != getattr(base, attr):
                    raise ValueError(
                        f"Multi-source mismatch for '{attr}': "
                        f"{getattr(base, attr)} != {getattr(src, attr)}"
                    )

    def _init_common_properties(self):
        base = self.sources[0]
        self.grid_type = base.grid_type
        self.nside = base.nside
        self.resolution = base.resolution
        self.plevel = base.plevel
        self.level_start = base.level_start
        self.level_end = base.level_end
        self.num_steps = base.num_steps
        self.load_final_target_only = base.load_final_target_only
        self.variables_prognostic = base.variables_prognostic
        self.variables_forcing = base.variables_forcing
        self.variables_diagnostic = base.variables_diagnostic
        self.variables_input = base.variables_input
        self.variables_output = base.variables_output
        self.variables = base.variables
        self.patch_size = base.patch_size
        self.npatches = base.npatches
        self.in_channels = base.in_channels
        self.out_channels = base.out_channels
        self.prognostic_channels = base.prognostic_channels
        self.diagnostic_channels = base.diagnostic_channels
        self.forcing_channels = base.forcing_channels
        self.n_levels = base.n_levels
        self.index_is_latlon = base.index_is_latlon

    def __len__(self):
        return self.n_samples_total

    def _build_epoch_samples(self, epoch_idx):
        if self.use_fixed_seed:
            seed = 42
        else:
            if self.use_time_variation_seed:
                time_variation = int(time.time()) % 100000
                seed = 42 + epoch_idx + time_variation
            else:
                seed = 42 + epoch_idx
        rng = np.random.default_rng(seed=seed)

        source_perms = [rng.permutation(n) for n in self.n_samples_total_by_source]
        if self.zarr_weights is None:
            source_ids = []
            for src_id, count in enumerate(self.n_samples_total_by_source):
                source_ids.extend([src_id] * count)
            source_ids = rng.permutation(source_ids)
        else:
            probs = np.asarray(self.zarr_weights, dtype=np.float64)
            probs = probs / probs.sum()
            source_ids = rng.choice(
                len(self.sources), size=self.n_samples_total, p=probs
            )

        source_positions = [0 for _ in self.sources]
        epoch_samples = []
        for src_id in source_ids:
            pos = source_positions[src_id]
            idx = int(
                source_perms[src_id][pos % self.n_samples_total_by_source[src_id]]
            )
            source_positions[src_id] += 1
            epoch_samples.append((int(src_id), idx))

        self._epoch_samples = epoch_samples

    def __call__(self, batch_info):
        logging.info(batch_info)
        iteration = batch_info.iteration
        if iteration >= self.full_iterations:
            raise StopIteration

        if self.last_seen_epoch != batch_info.epoch_idx:
            self.last_seen_epoch = batch_info.epoch_idx
            self._build_epoch_samples(batch_info.epoch_idx)

        s0 = []
        s_outputs = []
        s_forcings = []
        indices = []
        js = []

        for b in range(self.batch_size):
            sample = self.shard_offset + iteration * self.batch_size + b
            src_id, src_sample_idx = self._epoch_samples[sample]
            src = self.sources[src_id]
            j = src.index[src_sample_idx]
            time_len = max(1, src.nt - src.num_steps - 1)
            t = j % time_len + 1
            patch = j // time_len

            inp_patch = src.load_patch_input(t, patch)
            inp_patch = src._post_process(inp_patch)

            tar_patches = []
            if src.load_final_target_only:
                tar_patch = src.load_patch_output(t + src.num_steps, patch)
                tar_patch = src._post_process(tar_patch)
                tar_patches.append(tar_patch)
            else:
                for step in range(src.num_steps):
                    tar_patch = src.load_patch_output(t + step + 1, patch)
                    tar_patch = src._post_process(tar_patch)
                    tar_patches.append(tar_patch)
            stacked_outputs = torch.stack(tar_patches, dim=0)
            s_outputs.append(stacked_outputs)

            if src.num_steps > 1:
                force_patches = []
                for step in range(src.num_steps - 1):
                    force_patch = src.load_patch_forcing(t + step + 1, patch)
                    force_patch = src._post_process(force_patch)
                    force_patches.append(force_patch)
                stacked_forcings = torch.stack(force_patches, dim=0)
                s_forcings.append(stacked_forcings)

            start = patch * src.patch_size
            size = src.patch_size

            if src.index_is_latlon:
                lat_rad, lon_rad = src._latlon_radians
                patch_lat = src._post_process(lat_rad[start : start + size])
                patch_lon = src._post_process(lon_rad[start : start + size])
                index = torch.stack([patch_lat, patch_lon], dim=0)
            else:
                index = np.r_[start : start + size]
                index = src._post_process(index)

            s0.append(inp_patch)
            indices.append(index)
            js.append(torch.tensor([j], dtype=torch.int64))

        if src.num_steps == 1:
            return s0, [s.squeeze(0) for s in s_outputs], indices, js
        return s0, s_outputs, indices, js, s_forcings


class GlobalCrossFaceSrc:
    """
    DALI ExternalSource that yields a padded face from full global snapshots.
    Shapes per sample:
      - one-step:   s0 [C,Hpad,Wpad], s1 [C,Hpad,Wpad], index [Hpad,Wpad], j [1]
      - two-step:   s0 [C,Hpad,Wpad], s2 [C,Hpad,Wpad], index [Hpad,Wpad], j [1], sF1 [C,Hpad,Wpad]
    """

    def __init__(
        self,
        *,
        batch_size: int,
        split: str,
        num_shards: int,
        shard_id: int,
        mock: bool = False,
        plevel: int,
        level_start: int,
        level_end: int,
        variables_prognostic: tuple,
        variables_forcing: tuple,
        variables_diagnostic: tuple,
        num_steps: int,
        load_final_target_only: bool = True,
        tile_size: int,
        use_time_variation_seed: bool = False,
        use_fixed_seed: bool = False,
        train_start_index: int | None = None,
        train_end_index: int | None = None,
        test_start_index: int | None = None,
        test_end_index: int | None = None,
        test_stride: int = 1,
        nside: int = 1024,
        grid_type: str = "healpix",
        cubesphere_ne: int = 1024,
        cubesphere_npg: int = 2,
        main_zarr_path: str | None = None,
        aux_zarr_path: str | None = None,
        use_duo_padding: bool = False,
        duo_padding_scrip_src_path: str | None = None,
        duo_padding_scrip_tgt_path: str | None = None,
        index_is_latlon: bool = False,
        latlon_path: str | None = None,
    ):
        self.batch_size = batch_size
        self.split = split
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.mock = mock
        self.num_steps = max(1, int(num_steps))
        self.load_final_target_only = load_final_target_only
        self.tile_size = tile_size
        if self.tile_size % 2 != 0:
            raise ValueError(f"tile_size must be even, got {tile_size}")
        self.pad_width = self.tile_size // 2
        self.overlap_size = self.tile_size // 2
        self.use_time_variation_seed = use_time_variation_seed
        self.use_fixed_seed = use_fixed_seed
        self.train_start_index = train_start_index
        self.train_end_index = train_end_index
        self.test_start_index = test_start_index
        self.test_end_index = test_end_index
        self.test_stride = max(1, int(test_stride) if test_stride is not None else 1)

        # Grid type handling
        self.grid_type = (grid_type or "healpix").lower()
        if self.grid_type not in {"healpix", "cubesphere"}:
            raise ValueError(
                f"Unsupported grid_type='{grid_type}'. Expected 'healpix' or 'cubesphere'."
            )

        # Set grid-specific parameters
        if self.grid_type == "cubesphere":
            self.cubesphere_ne = int(cubesphere_ne)
            self.cubesphere_npg = int(cubesphere_npg)
            self.nside = self.cubesphere_ne * self.cubesphere_npg  # face size
            self.n_faces = 6
        else:
            self.cubesphere_ne = None
            self.cubesphere_npg = None
            self.nside = nside
            self.n_faces = 12

        # Underlying loader to access global arrays (re-use ScreamV2 machinery)
        self.ds = ScreamV2(
            batch_size=1,
            split="",
            num_shards=1,
            shard_id=0,
            mock=mock,
            plevel=plevel,
            level_start=level_start,
            level_end=level_end,
            variables_prognostic=variables_prognostic,
            variables_forcing=variables_forcing,
            variables_diagnostic=variables_diagnostic,
            num_steps=1,  # not using ScreamV2 sampling here
            grid_type=self.grid_type,
            cubesphere_ne=cubesphere_ne,
            cubesphere_npg=cubesphere_npg,
            main_zarr_path=main_zarr_path,
            aux_zarr_path=aux_zarr_path,
        )
        self.npix = self.n_faces * self.nside * self.nside
        # Channel counts for mock generation
        if hasattr(ScreamV2, "num_of_input_channels"):
            self._in_channels = ScreamV2.num_of_input_channels(
                variables_prognostic,
                variables_forcing,
                plevel,
                level_start,
                level_end,
            )
            self._out_channels = ScreamV2.num_of_output_channels(
                variables_prognostic,
                variables_diagnostic,
                plevel,
                level_start,
                level_end,
            )
        else:
            # Fallback to ds attributes
            self._in_channels = getattr(self.ds, "in_channels", 0)
            self._out_channels = getattr(self.ds, "out_channels", 0)
        self._forcing_channels = getattr(self.ds, "forcing_channels", 0)

        # Face reorder/pad grids (HEALPix-specific)
        if self.grid_type == "healpix":
            level = earth2grid.healpix.nside2level(self.nside)
            self.src_grid = earth2grid.healpix.Grid(
                level=level, pixel_order=earth2grid.healpix.PixelOrder.NEST
            )
            self.xy_grid = earth2grid.healpix.Grid(
                level=level, pixel_order=earth2grid.healpix.HEALPIX_PAD_XY
            )
        else:
            # CubeSphere: no earth2grid grids needed
            self.src_grid = None
            self.xy_grid = None

        # Duo-style padding (KNN halo fill) for cubesphere faces
        self.use_duo_padding = use_duo_padding
        self.duo_pad_fn = None
        if self.use_duo_padding:
            if self.grid_type != "cubesphere":
                raise ValueError(
                    "use_duo_padding is only supported for grid_type='cubesphere'."
                )
            if not duo_padding_scrip_src_path or not duo_padding_scrip_tgt_path:
                raise ValueError(
                    "duo_padding_scrip_src_path and duo_padding_scrip_tgt_path are required "
                    "when use_duo_padding=True."
                )
            with xr.open_dataset(
                duo_padding_scrip_src_path
            ) as scrip_src, xr.open_dataset(duo_padding_scrip_tgt_path) as scrip_tgt:
                for ds, path in [
                    (scrip_src, duo_padding_scrip_src_path),
                    (scrip_tgt, duo_padding_scrip_tgt_path),
                ]:
                    if "grid_center_lon" not in ds or "grid_center_lat" not in ds:
                        raise ValueError(
                            f"{path} must include grid_center_lon/grid_center_lat"
                        )
                grid_src = {
                    "lon": scrip_src["grid_center_lon"].values.astype(np.float32),
                    "lat": scrip_src["grid_center_lat"].values.astype(np.float32),
                }
                grid_tgt = {
                    "lon": scrip_tgt["grid_center_lon"].values.astype(np.float32),
                    "lat": scrip_tgt["grid_center_lat"].values.astype(np.float32),
                }
            self.duo_pad_fn = unstructured_to_padded_faces_knn(
                grid_src,
                grid_tgt,
                num_elements=self.cubesphere_ne,
                halo_width=self.pad_width,
                num_pg_cells=self.cubesphere_npg,
            )

        # Precompute padded lat/lon faces for direct lat/lon indexing.
        # When index_is_latlon=True, the dataloader emits [2, Hpad, Wpad] float
        # (lat_rad, lon_rad) instead of [Hpad, Wpad] int64 pixel indices.
        # This is essential for duo padding where halo pixels are KNN-interpolated
        # and integer indices are meaningless in the halo region.
        self.index_is_latlon = index_is_latlon
        self._latlon_faces = None
        if self.index_is_latlon:
            if self.use_duo_padding:
                # The target SCRIP grid file contains a pre-created padded
                # cubesphere (e.g. pad_file=512).  Reshape its lat/lon to
                # 6 faces, then crop the centre to face_size + 2*pad_width.
                face_size = self.cubesphere_ne * self.cubesphere_npg
                n_tgt = grid_tgt["lon"].shape[0]
                face_size_file = int(round((n_tgt / 6) ** 0.5))
                ne_file = face_size_file // self.cubesphere_npg
                lon_faces_full = unstructured_to_6faces(
                    torch.from_numpy(grid_tgt["lon"]),
                    ne=ne_file,
                    npg=self.cubesphere_npg,
                )  # [6, face_size_file, face_size_file]
                lat_faces_full = unstructured_to_6faces(
                    torch.from_numpy(grid_tgt["lat"]),
                    ne=ne_file,
                    npg=self.cubesphere_npg,
                )  # [6, face_size_file, face_size_file]

                halo_width_file = (face_size_file - face_size) // 2
                i0 = halo_width_file - self.pad_width
                i1 = halo_width_file + face_size + self.pad_width
                lat_crop = lat_faces_full[:, i0:i1, i0:i1]  # [6, Hpad, Wpad]
                lon_crop = lon_faces_full[:, i0:i1, i0:i1]

                # deg → rad, stack as [6, 2, Hpad, Wpad]
                self._latlon_faces = torch.stack(
                    [
                        torch.deg2rad(lat_crop),
                        torch.deg2rad(lon_crop),
                    ],
                    dim=1,
                ).float()
            else:
                # Non-duo cases: build 1-D lat/lon, pad through _faces_padded
                if self.grid_type == "healpix":
                    lat_rad_np = np.deg2rad(
                        np.array(self.src_grid.lat, dtype=np.float32)
                    )
                    lon_rad_np = np.deg2rad(
                        np.array(self.src_grid.lon, dtype=np.float32)
                    )
                else:
                    # CubeSphere without duo padding: load from latlon_path
                    if not latlon_path:
                        raise ValueError(
                            "latlon_path is required for index_is_latlon=True "
                            "without duo padding on cubesphere grid."
                        )
                    with xr.open_dataset(latlon_path) as ds_ll:
                        lat_rad_np = np.deg2rad(ds_ll["lat"].values.astype(np.float32))
                        lon_rad_np = np.deg2rad(ds_ll["lon"].values.astype(np.float32))

                # Pad lat/lon directly through _faces_padded. Corner halo
                # values may have ill-averaged lon near the 0°/360° seam, but
                # those tiles are skipped by the subtiler (skip_corner_tiles).
                latlon_1d = np.stack([lat_rad_np, lon_rad_np])  # [2, npix]
                latlon_t = torch.from_numpy(latlon_1d.astype(np.float32)).unsqueeze(
                    0
                )  # [1, 2, npix]
                self._latlon_faces = self._faces_padded(
                    latlon_t
                )  # [n_faces, 2, Hpad, Wpad]

        # Build time index honoring windows and two-step constraints; shard across ranks
        self.time_index = self._build_time_index()
        # Times assigned to this shard (DDP): disjoint subset
        self.times_shard = self.time_index[self.shard_id :: self.num_shards]
        # Total samples = times * n_faces
        self.n_samples_shard = len(self.times_shard) * self.n_faces
        self.full_iterations = max(0, self.n_samples_shard // self.batch_size)
        self.time_perm = None  # permutation over time positions per epoch
        self.last_seen_epoch = None

    def __len__(self):
        return self.n_samples_shard

    def _faces_padded(self, img_nc_np: torch.Tensor):
        # img_nc_np: [1, C, npix]
        # return [n_faces, C, Hpad, Wpad]
        if self.grid_type == "healpix":
            return self._faces_padded_healpix(img_nc_np)
        else:
            return self._faces_padded_cubesphere(img_nc_np)

    def _faces_padded_healpix(self, img_nc_np: torch.Tensor):
        # img_nc_np: [1, C, npix]
        # return [12, C, Hpad, Wpad]
        # Reorder and pad;
        img = self.src_grid.reorder(self.xy_grid.pixel_order, img_nc_np)
        img_reshaped = einops.rearrange(
            img, "n c (f x y) -> (n c) f x y", f=12, x=self.nside, y=self.nside
        )
        padded = earth2grid.healpix.pad(img_reshaped, padding=self.overlap_size)
        faces = einops.rearrange(
            padded, "(n c) f x y -> n f c x y", n=img.shape[0]
        ).squeeze(0)
        if img_nc_np.dtype == torch.long:
            faces = faces.long()
        return faces

    def _faces_padded_cubesphere(self, img_nc_np: torch.Tensor):
        # img_nc_np: [1, C, npix] where npix = 6 * nside * nside
        # return [6, C, Hpad, Wpad]

        if self.use_duo_padding and self.duo_pad_fn is not None:
            # KNN-based halo fill: duo_pad_fn expects (..., npix) and returns (..., 6, Hpad, Wpad)
            padded = self.duo_pad_fn(img_nc_np)
        else:
            # Default padding via face-boundary replication
            faces_2d = unstructured_to_6faces(
                img_nc_np, ne=self.cubesphere_ne, npg=self.cubesphere_npg
            )
            padded = create_padded_faces_batched(faces_2d, pad_width=self.overlap_size)
        faces = einops.rearrange(padded, "n c f x y -> n f c x y").squeeze(0)
        if img_nc_np.dtype == torch.long:
            faces = faces.long()
        return faces

    def _build_time_index(self):
        nt = int(self.ds.nt)
        # Valid time range: [1, nt - num_steps - 1]
        # - Lower bound t=1: Skip t=0 because the first time slice in zarr may have NaN/zero values
        # - Upper bound: For num_steps loading, we need t+num_steps to be valid (< nt)
        max_valid_time = max(1, nt - self.num_steps - 1)

        def to_time_range(start_time, end_time, stride=1):
            """Build list of valid times in [start_time, end_time) with stride."""
            s = 1 if start_time is None else max(1, int(start_time))
            e = (
                max_valid_time + 1
                if end_time is None
                else min(max_valid_time + 1, int(end_time))
            )
            st = max(1, int(stride))
            if e <= s:
                return []
            # Filter to ensure all times are within valid range [1, max_valid_time]
            return [t for t in range(s, e, st) if 1 <= t <= max_valid_time]

        if self.split == "train":
            times = to_time_range(self.train_start_index, self.train_end_index, 1)
        elif self.split == "test":
            times = to_time_range(
                self.test_start_index, self.test_end_index, self.test_stride
            )
        else:
            times = list(range(1, max_valid_time + 1))
        return times

    def _ensure_cached(self, t):
        """Load and cache padded face tensors for time step *t* (no-op if already cached)."""
        if getattr(self, "_cache_t", None) == t:
            return
        # Load global input and multiple outputs/forcings for multistep
        if self.mock:
            inp = np.random.normal(size=(self._in_channels, self.npix)).astype(
                np.float32
            )
            if self.load_final_target_only:
                tar_list = [
                    np.random.normal(size=(self._out_channels, self.npix)).astype(
                        np.float32
                    )
                ]
            else:
                tar_list = [
                    np.random.normal(size=(self._out_channels, self.npix)).astype(
                        np.float32
                    )
                    for _ in range(self.num_steps)
                ]
            if self.num_steps > 1 and self._forcing_channels > 0:
                force_list = [
                    np.random.normal(size=(self._forcing_channels, self.npix)).astype(
                        np.float32
                    )
                    for _ in range(self.num_steps - 1)
                ]
            else:
                force_list = []
        else:
            inp = self.ds._assemble_patch_vars(t, 0, self.npix, self.ds.variables_input)
            if self.load_final_target_only:
                tar_list = [
                    self.ds._assemble_patch_vars(
                        t + self.num_steps,
                        0,
                        self.npix,
                        self.ds.variables_output,
                    )
                ]
            else:
                tar_list = [
                    self.ds._assemble_patch_vars(
                        t + step + 1, 0, self.npix, self.ds.variables_output
                    )
                    for step in range(self.num_steps)
                ]
            if self.num_steps > 1 and self._forcing_channels > 0:
                force_list = [
                    self.ds._assemble_patch_vars(
                        t + step + 1, 0, self.npix, self.ds.variables_forcing
                    )
                    for step in range(self.num_steps - 1)
                ]
            else:
                force_list = []

        # Convert to torch and add batch dim [1,C,N]
        inp_t = torch.from_numpy(inp.astype(np.float32)).unsqueeze(0)

        # Build per-face padded tensors
        self._cache_inp_faces = self._faces_padded(inp_t)  # [n_faces,C,Hpad,Wpad]

        # Index faces: precomputed lat/lon or integer pixel indices
        if not self.index_is_latlon:
            idx_t = torch.arange(self.npix, dtype=torch.long).view(1, 1, self.npix)
            self._cache_idx_faces = self._faces_padded(idx_t).squeeze(
                1
            )  # [n_faces,Hpad,Wpad]

        # Cache all output faces → [n_faces, T, C, Hpad, Wpad]
        tar_faces_list = []
        for tar in tar_list:
            tar_t = torch.from_numpy(tar.astype(np.float32)).unsqueeze(0)
            tar_faces_list.append(self._faces_padded(tar_t))
        self._cache_tar_faces_stacked = torch.stack(tar_faces_list, dim=1)

        # Cache all forcing faces → [n_faces, T-1, C, Hpad, Wpad]
        if force_list:
            fn_faces_list = []
            for force in force_list:
                fn_t = torch.from_numpy(force.astype(np.float32)).unsqueeze(0)
                fn_faces_list.append(self._faces_padded(fn_t))
            self._cache_fn_faces_stacked = torch.stack(fn_faces_list, dim=1)
        else:
            self._cache_fn_faces_stacked = None

        self._cache_t = t

    def _clear_cache(self):
        """Release cached tensors to free memory."""
        self._cache_t = None
        self._cache_inp_faces = None
        self._cache_tar_faces_stacked = None
        self._cache_fn_faces_stacked = None
        self._cache_idx_faces = None

    def _get_face_sample(self, face):
        """Extract a single face sample from the cache. Returns (s0, target, index, forcing_or_None)."""
        s0 = self._cache_inp_faces[face]  # [C, Hpad, Wpad]
        target = self._cache_tar_faces_stacked[face]  # [T, C, Hpad, Wpad]
        if self.index_is_latlon:
            index = self._latlon_faces[face]  # [2, Hpad, Wpad]
        else:
            index = self._cache_idx_faces[face]  # [Hpad, Wpad]
        forcing = (
            self._cache_fn_faces_stacked[face]
            if self._cache_fn_faces_stacked is not None
            else None
        )
        return s0, target, index, forcing

    def __call__(self, batch_info):
        logging.info(batch_info)
        iteration = batch_info.iteration
        if iteration >= self.full_iterations:
            raise StopIteration

        if self.last_seen_epoch != batch_info.epoch_idx:
            self.last_seen_epoch = batch_info.epoch_idx
            # Shuffle time order per epoch; tiles within time remain sequential
            if self.use_fixed_seed:
                seed = 42
            else:
                if self.use_time_variation_seed:
                    time_variation = int(time.time()) % 100000
                    seed = 42 + batch_info.epoch_idx + time_variation
                else:
                    seed = 42 + batch_info.epoch_idx
            rng = np.random.default_rng(seed=seed)
            self.time_perm = rng.permutation(len(self.times_shard))

        s0 = []
        s_outputs = []  # will be stacked to [T, C, H, W] per sample
        s_forcings = (
            []
        )  # will be stacked to [T-1, C, H, W] per sample (if num_steps > 1)
        indices = []
        js = []

        for b in range(self.batch_size):
            # Compute which (time_slot, face_slot) this sample corresponds to
            sample_idx = iteration * self.batch_size + b
            time_slot = sample_idx // self.n_faces  # which time slot (0, 1, 2, ...)
            face_slot = (
                sample_idx % self.n_faces
            )  # which face within that time slot (0 to n_faces-1)

            # Get actual time value: apply time permutation, then look up in times_shard
            permuted_time_slot = int(self.time_perm[time_slot])
            t = int(self.times_shard[permuted_time_slot])

            # Shuffle faces within each time step for randomization
            face_seed = (
                t if not self.use_time_variation_seed else t + int(time.time()) % 100000
            )
            rng_faces = np.random.default_rng(seed=face_seed)
            shuffled_faces = rng_faces.permutation(self.n_faces)
            face = shuffled_faces[face_slot]  # actual face index to load

            self._ensure_cached(t)
            inp, target, index, forcing = self._get_face_sample(face)

            s0.append(inp)
            s_outputs.append(target)
            indices.append(index)
            js.append(torch.tensor([t], dtype=torch.int64))
            if forcing is not None:
                s_forcings.append(forcing)

        # Return format:
        # For num_steps=1: s0, s1, indices, js (backwards compatible)
        # For num_steps>1: s0, s_outputs [T,C,H,W], indices, js, s_forcings [T-1,C,H,W]
        if self.num_steps == 1:
            # Squeeze the T dimension for backwards compatibility
            return s0, [s.squeeze(0) for s in s_outputs], indices, js
        return s0, s_outputs, indices, js, s_forcings


class MultiGlobalCrossFaceSrc:
    """Multi-source wrapper for GlobalCrossFaceSrc that mixes samples across datasets."""

    def __init__(
        self,
        *,
        batch_size: int,
        split: str,
        num_shards: int,
        shard_id: int,
        mock: bool = False,
        plevel: int,
        level_start: int,
        level_end: int,
        variables_prognostic: tuple,
        variables_forcing: tuple,
        variables_diagnostic: tuple,
        num_steps: int,
        load_final_target_only: bool = True,
        tile_size: int,
        use_time_variation_seed: bool = False,
        use_fixed_seed: bool = False,
        train_start_index: int | None = None,
        train_end_index: int | None = None,
        test_start_index: int | None = None,
        test_end_index: int | None = None,
        test_stride: int = 1,
        nside: int = 1024,
        grid_type: str = "healpix",
        cubesphere_ne: int = 1024,
        cubesphere_npg: int = 2,
        main_zarr_paths: tuple[str, ...] | list[str] = (),
        aux_zarr_paths: tuple[str, ...] | list[str] | None = None,
        zarr_weights: tuple[float, ...] | list[float] | None = None,
        train_start_indices: tuple[int, ...] | list[int] | None = None,
        train_end_indices: tuple[int | None, ...] | list[int | None] | None = None,
        use_duo_padding: bool = False,
        duo_padding_scrip_src_path: str | None = None,
        duo_padding_scrip_tgt_path: str | None = None,
        index_is_latlon: bool = False,
        latlon_path: str | None = None,
    ):
        if not main_zarr_paths:
            raise ValueError("main_zarr_paths must contain at least one Zarr path")
        if aux_zarr_paths is not None and len(aux_zarr_paths) != len(main_zarr_paths):
            raise ValueError("aux_zarr_paths must match main_zarr_paths length")
        if zarr_weights is not None and len(zarr_weights) != len(main_zarr_paths):
            raise ValueError("zarr_weights must match main_zarr_paths length")
        if train_start_indices is not None and len(train_start_indices) != len(
            main_zarr_paths
        ):
            raise ValueError("train_start_indices must match main_zarr_paths length")
        if train_end_indices is not None and len(train_end_indices) != len(
            main_zarr_paths
        ):
            raise ValueError("train_end_indices must match main_zarr_paths length")

        self.sources: list[GlobalCrossFaceSrc] = []
        for idx, main_path in enumerate(main_zarr_paths):
            aux_path = aux_zarr_paths[idx] if aux_zarr_paths is not None else None
            per_source_train_start = (
                train_start_indices[idx]
                if train_start_indices is not None
                else train_start_index
            )
            per_source_train_end = (
                train_end_indices[idx]
                if train_end_indices is not None
                else train_end_index
            )
            src = GlobalCrossFaceSrc(
                batch_size=1,
                split=split,
                num_shards=1,
                shard_id=0,
                mock=mock,
                plevel=plevel,
                level_start=level_start,
                level_end=level_end,
                variables_prognostic=variables_prognostic,
                variables_forcing=variables_forcing,
                variables_diagnostic=variables_diagnostic,
                num_steps=num_steps,
                load_final_target_only=load_final_target_only,
                tile_size=tile_size,
                use_time_variation_seed=use_time_variation_seed,
                use_fixed_seed=use_fixed_seed,
                train_start_index=per_source_train_start,
                train_end_index=per_source_train_end,
                test_start_index=test_start_index,
                test_end_index=test_end_index,
                test_stride=test_stride,
                nside=nside,
                grid_type=grid_type,
                cubesphere_ne=cubesphere_ne,
                cubesphere_npg=cubesphere_npg,
                main_zarr_path=main_path,
                aux_zarr_path=aux_path,
                use_duo_padding=use_duo_padding,
                duo_padding_scrip_src_path=duo_padding_scrip_src_path,
                duo_padding_scrip_tgt_path=duo_padding_scrip_tgt_path,
                index_is_latlon=index_is_latlon,
                latlon_path=latlon_path,
            )
            self.sources.append(src)
            logging.info(
                "MultiGlobalCrossFaceSrc source %d: main=%s aux=%s "
                "train_start=%s train_end=%s",
                idx,
                main_path,
                aux_path or main_path,
                per_source_train_start,
                per_source_train_end,
            )

        self.batch_size = batch_size
        self.split = split
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.mock = mock
        self.use_time_variation_seed = use_time_variation_seed
        self.use_fixed_seed = use_fixed_seed
        self.num_steps = max(1, int(num_steps))
        self.index_is_latlon = index_is_latlon

        # Validate compatible sources
        self._validate_compatible_sources()

        # Copy common properties from the first source
        base = self.sources[0]
        self.nside = base.nside
        self.n_faces = base.n_faces
        self.npix = base.npix
        self._in_channels = base._in_channels
        self._out_channels = base._out_channels
        self._forcing_channels = base._forcing_channels

        # Weights
        self.zarr_weights = (
            [float(w) for w in zarr_weights] if zarr_weights is not None else None
        )
        if self.zarr_weights is not None:
            if any(w < 0 for w in self.zarr_weights):
                raise ValueError("zarr_weights must be non-negative")
            if sum(self.zarr_weights) == 0:
                raise ValueError("zarr_weights must sum to a positive value")

        # Each source's sample count = len(time_index) * n_faces
        self.n_samples_total_by_source = [
            len(src.time_index) * src.n_faces for src in self.sources
        ]
        if any(n <= 0 for n in self.n_samples_total_by_source):
            raise ValueError("Each source must have at least one sample")
        self.n_samples_total = int(sum(self.n_samples_total_by_source))
        self.n_samples_shard = self.n_samples_total // self.num_shards
        self.full_iterations = self.n_samples_shard // self.batch_size
        self.shard_offset = self.shard_id * self.n_samples_shard
        self.last_seen_epoch = None
        self._epoch_samples = None
        # Track the active (source, time) across __call__ invocations so we
        # can clear the previous source's cache exactly once at block boundaries.
        self._active_src_id = None
        self._active_t = None

    def _validate_compatible_sources(self):
        base = self.sources[0]
        compare_attrs = [
            "grid_type",
            "nside",
            "n_faces",
            "npix",
            "num_steps",
            "load_final_target_only",
            "index_is_latlon",
            "_in_channels",
            "_out_channels",
            "_forcing_channels",
        ]
        for src in self.sources[1:]:
            for attr in compare_attrs:
                if getattr(src, attr) != getattr(base, attr):
                    raise ValueError(
                        f"Multi-source mismatch for '{attr}': "
                        f"{getattr(base, attr)} != {getattr(src, attr)}"
                    )

    def __len__(self):
        return self.n_samples_total

    def _build_epoch_samples(self, epoch_idx):
        """Build a flat sample list for the epoch, grouped by (source, time) blocks.

        Each block contains ``n_faces`` samples that share the same source and
        time step.  Blocks are shuffled across sources (respecting optional
        ``zarr_weights``), but within a block the faces are emitted
        contiguously so that ``_ensure_cached(t)`` is called only once per
        block — no cache thrashing.
        """
        if self.use_fixed_seed:
            seed = 42
        else:
            if self.use_time_variation_seed:
                time_variation = int(time.time()) % 100000
                seed = 42 + epoch_idx + time_variation
            else:
                seed = 42 + epoch_idx
        rng = np.random.default_rng(seed=seed)

        # Number of time-step blocks per source
        n_times_by_source = [len(src.time_index) for src in self.sources]

        # Shuffle which time slots each source will serve
        time_perms = [rng.permutation(n) for n in n_times_by_source]

        # Decide the order of (source, time) blocks across sources
        if self.zarr_weights is None:
            block_source_ids = []
            for src_id, n_t in enumerate(n_times_by_source):
                block_source_ids.extend([src_id] * n_t)
            block_source_ids = rng.permutation(block_source_ids)
        else:
            n_blocks_total = sum(n_times_by_source)
            probs = np.asarray(self.zarr_weights, dtype=np.float64)
            probs = probs / probs.sum()
            block_source_ids = rng.choice(
                len(self.sources), size=n_blocks_total, p=probs
            )

        # Walk through block order, emitting n_faces samples per block
        source_time_pos = [0 for _ in self.sources]
        epoch_samples = []
        for src_id in block_source_ids:
            src_id = int(src_id)
            pos = source_time_pos[src_id]
            time_slot = int(time_perms[src_id][pos % n_times_by_source[src_id]])
            source_time_pos[src_id] += 1

            # Shuffle face order within this block
            face_order = rng.permutation(self.n_faces)
            epoch_samples.extend((src_id, time_slot, int(face)) for face in face_order)

        self._epoch_samples = epoch_samples

    def __call__(self, batch_info):
        logging.info(batch_info)
        iteration = batch_info.iteration
        if iteration >= self.full_iterations:
            raise StopIteration

        if self.last_seen_epoch != batch_info.epoch_idx:
            self.last_seen_epoch = batch_info.epoch_idx
            self._build_epoch_samples(batch_info.epoch_idx)

        s0 = []
        s_outputs = []
        s_forcings = []
        indices = []
        js = []

        for b in range(self.batch_size):
            sample = self.shard_offset + iteration * self.batch_size + b
            src_id, time_slot, face = self._epoch_samples[sample]
            src = self.sources[src_id]
            t = int(src.time_index[time_slot])

            # If we moved to a different (source, time) block, clear the old
            # source's cache so at most one source holds data at a time.
            if self._active_src_id is not None and (
                src_id != self._active_src_id or t != self._active_t
            ):
                self.sources[self._active_src_id]._clear_cache()

            src._ensure_cached(t)
            self._active_src_id = src_id
            self._active_t = t

            inp, target, index, forcing = src._get_face_sample(face)

            s0.append(inp)
            s_outputs.append(target)
            indices.append(index)
            js.append(torch.tensor([t], dtype=torch.int64))
            if forcing is not None:
                s_forcings.append(forcing)

        if self.num_steps == 1:
            return s0, [s.squeeze(0) for s in s_outputs], indices, js
        return s0, s_outputs, indices, js, s_forcings
