#!/usr/bin/env python3
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
# TODO
"""
import itertools
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import tqdm
import xarray
import zarr
from earth2studio.models.px import ace2

from screamcast.catalog_core import Grids
from screamcast.history import history_entry
from screamcast.horizontal_regridding import UnstructuredToLatLonRegridder


def _load_ace_grid():
    print("Loading ACE grid...")
    model = ace2.ACE2ERA5.load_model(ace2.ACE2ERA5.load_default_package())
    coords = model.input_coords()
    target_lat = np.asarray(coords["lat"])
    target_lon = np.asarray(coords["lon"])
    return target_lat, target_lon


def regrid(input, output, selection, prefetch=4, device="cuda", chunk_size: int = 8):
    """Regrid dataset

    all arrays shaped liked (..., face, x, y) are regridded using to the ace grid

    Args:
        input: str or catalog_core object w/ to_xarray()
        chunk_size: number of chunks loaded per-thread at once

    """
    try:
        ds = input.to_xarray(chunks=None)
    except AttributeError:
        ds = xarray.open_zarr(input, chunks=None)

    grid = ds.attrs.get("grid", "")
    if grid == Grids.ne1024pg2:
        spatial_coords = ("cell",)
    else:
        spatial_coords = ("face", "x", "y")

    ndim = len(spatial_coords)

    ds = ds.isel(selection)
    device = torch.device(device)

    # --- ACE grid ---
    target_lat, target_lon = _load_ace_grid()
    n_lat, n_lon = len(target_lat), len(target_lon)

    # --- Regridder ---
    print("Building SCREAM->ACE regridder...")
    scream_lat = ds["lat"].values
    scream_lon = ds["lon"].values
    scream2ace = UnstructuredToLatLonRegridder(
        scream_lon, scream_lat, target_lat, target_lon
    )
    scream2ace = scream2ace.float().to(device)
    # print("done")

    # --- SCREAM zarr ---
    def _is_spatial_field(arr):
        return tuple(arr.dims)[-len(spatial_coords) :] == spatial_coords

    all_variables = list(ds.data_vars) + list(ds.coords)
    all_variables.remove("lat")
    all_variables.remove("lon")

    regrid_vars = [name for name in all_variables if _is_spatial_field(ds[name])]
    copy_vars = [
        name
        for name in all_variables
        if not _is_spatial_field(ds[name]) and name not in spatial_coords
    ]

    out = zarr.open_group(output, mode="w")
    out.attrs.update(ds.attrs)
    out.attrs["history"] = history_entry(previous=ds.attrs.get("history"))
    out.create_array(
        "lat",
        data=target_lat.astype(np.float32),
        dimension_names=("lat",),
        overwrite=True,
    )
    out.create_array(
        "lon",
        data=target_lon.astype(np.float32),
        dimension_names=("lon",),
        overwrite=True,
    )

    # Copy non-spatial variables/coords directly
    for var in copy_vars:
        print(f"copying {var}")
        out.create_array(
            var,
            data=ds[var].values,
            attributes=dict(ds[var].attrs),
            dimension_names=ds[var].dims,
            overwrite=True,
        )

    # Initialize output arrays for spatial variables
    for var in regrid_vars:
        arr = ds[var]
        # Replace last 3 dims (face, x, y) with (lat, lon); subsample levels if 128
        prefix = arr.shape[:-ndim]  # e.g. (n_times,) or (n_times, n_levels)
        dimension_names = arr.dims[:-ndim] + ("lat", "lon")
        new_shape = prefix + (n_lat, n_lon)
        out.create_array(
            var,
            shape=new_shape,
            dtype=ds[var].dtype,
            chunks=(1,) * len(prefix) + new_shape[-2:],
            attributes=dict(arr.attrs),
            dimension_names=dimension_names,
            overwrite=True,
        )
    zarr.consolidate_metadata(output)

    def field_iterator():
        for variable in regrid_vars:
            prefix_shape = ds[variable].shape[:-ndim]
            if not prefix_shape:
                yield variable, ()
                continue
            *shape, n = prefix_shape
            coord_iter = itertools.product(*[range(n) for n in shape])
            # chunk over the final non spatial dimension
            nchunks = math.ceil(n / chunk_size)
            for coord in coord_iter:
                for j in range(nchunks):
                    final_dim_indexer = slice(j * chunk_size, (j + 1) * chunk_size)
                    yield variable, (*coord, final_dim_indexer)

    output_group = zarr.open_group(output, mode="a")

    def _load(arg):
        name, i = arg
        field = ds[name][i].values
        x = torch.from_numpy(np.ascontiguousarray(field)).to(device)
        return name, i, x

    tasks = list(field_iterator())
    with ThreadPoolExecutor(max_workers=prefetch) as pool:
        for name, i, field in tqdm.tqdm(pool.map(_load, tasks), total=len(tasks)):
            with torch.no_grad():
                field = field.reshape(*field.shape[:-ndim], -1)
                regridded = scream2ace(field).cpu().numpy()  # (batch, n_lat, n_lon)
                output_group[name][i] = regridded
            # print("processed", name,i )

    print(f"Done: {output}")
