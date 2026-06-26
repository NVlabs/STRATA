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
import functools
import json
import os

import numpy as np
import xarray
import zarr
from zarr.storage import WrapperStore

from screamcast.dali_ext_src import open_consolidated_zarr_group

DEFAULT_DIMENSION_NAMES_BY_RANK = {
    1: ["cell"],
    2: ["time", "cell"],
    3: ["time", "level", "cell"],
}


class Grids:
    ne1024pg2 = "ne1024pg2"


GRID_INFO: dict[str, tuple[int, int]] = {Grids.ne1024pg2: (1024, 2)}


class _PatchedMetadataStore(WrapperStore):
    def __init__(self, store, metadata_overrides: dict[str, bytes]) -> None:
        super().__init__(store)
        self._metadata_overrides = metadata_overrides

    def _with_store(self, store):
        return type(self)(store=store, metadata_overrides=self._metadata_overrides)

    async def get(self, key, prototype, byte_range=None):
        if byte_range is None and key in self._metadata_overrides:
            return prototype.buffer.from_bytes(self._metadata_overrides[key])
        return await self._store.get(key, prototype, byte_range)

    async def exists(self, key):
        if key in self._metadata_overrides:
            return True
        return await self._store.exists(key)

    def with_read_only(self, read_only: bool = False):
        return self


def _runtime_xarray_store(group, dimension_names_by_rank: dict[int, list[str]]):
    """Wrap the group store with xarray-compatible dimension names in metadata."""
    root_metadata = group.metadata.to_dict()
    inline_metadata = root_metadata["consolidated_metadata"]["metadata"]

    for key, metadata in inline_metadata.items():
        if metadata.get("node_type") != "array":
            continue
        patched = dict(metadata)
        rank = len(patched["shape"])
        try:
            patched["dimension_names"] = dimension_names_by_rank[rank]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported array rank for runtime xarray metadata: {patched['shape']}"
            ) from exc
        inline_metadata[key] = patched

    metadata_overrides = {
        "zarr.json": json.dumps(root_metadata).encode("utf-8"),
    }
    return _PatchedMetadataStore(group.store, metadata_overrides)


def open_zarr_with_dimension_names(
    group,
    *,
    dimension_names_by_rank: dict[int, list[str]] | None = None,
    **kwargs,
):
    """used to open zarr data as an xarray that was not saved w/ a dimension_names attribute"""
    if dimension_names_by_rank is None:
        dimension_names_by_rank = DEFAULT_DIMENSION_NAMES_BY_RANK
    store = _runtime_xarray_store(group, dimension_names_by_rank)
    return xarray.open_zarr(
        store,
        consolidated=True,
        zarr_format=3,
        **kwargs,
    )


@dataclasses.dataclass
class ScreamData:
    path: str
    profile: str
    vertical_subset: slice
    grid_file: str
    vertical_coordinate_file: str
    reference_time: np.datetime64
    native_timestep: np.timedelta64

    def to_data_source(self):
        from screamcast.earth2studio_wrappers import ScreamDataSource

        main_group = open_consolidated_zarr_group(self.path, storage_env=self.profile)
        var_group = {key: main_group for key in main_group}
        return ScreamDataSource(var_group=var_group)

    def to_zarr(self) -> zarr.Group:
        return open_consolidated_zarr_group(self.path, storage_env=self.profile)

    def to_hybrid_vertical_coordinates(self):
        with xarray.open_dataset(self.vertical_coordinate_file) as ds:
            hyam = ds["hyam"].isel(lev=self.vertical_subset).values
            hybm = ds["hybm"].isel(lev=self.vertical_subset).values
            level = ds["lev"].isel(lev=self.vertical_subset).values
        return hyam, hybm, level

    @functools.cached_property
    def time(self) -> np.ndarray:
        ds = open_zarr_with_dimension_names(self.to_zarr(), chunks=None)
        return (
            self.reference_time
            + np.arange(ds.sizes["time"], dtype=np.int64) * self.native_timestep
        ).astype("datetime64[s]")

    def _latlon_coords(self):
        with xarray.open_dataset(self.grid_file) as ds:
            lat = ds["lat"].values
            lon = ds["lon"].values
        return lat, lon

    def to_xarray(self, **kwargs):
        group = self.to_zarr()
        ds = open_zarr_with_dimension_names(group, **kwargs)
        if "time" in ds.dims:
            ds = ds.assign_coords(time=("time", self.time))

        if "level" in ds.dims:
            hyam, hybm, level = self.to_hybrid_vertical_coordinates()
            ds = ds.assign_coords(
                hyam=("level", hyam),
                hybm=("level", hybm),
                level=("level", level),
            )
            for coord_name in ("hyam", "hybm", "level"):
                ds.coords[coord_name].attrs[
                    "source_location"
                ] = self.vertical_coordinate_file

        if os.path.exists(self.grid_file):
            lat, lon = self._latlon_coords()
            ds = ds.assign_coords(
                lat=("cell", lat),
                lon=("cell", lon),
            )
            for coord_name in ("lat", "lon"):
                ds.coords[coord_name].attrs["source_location"] = self.grid_file

        ds.attrs["grid"] = Grids.ne1024pg2

        return ds
