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
from nvidia.dali import fn, types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator

from screamcast import dataloader_transforms
from screamcast.dali_ext_src import (
    GlobalCrossFaceSrc,
    MultiGlobalCrossFaceSrc,
    MultiScreamV2,
    ScreamV2,
)


class DALIWrapper:
    def __init__(self, dali_iterator, src):
        self.dali_iterator = dali_iterator
        self.len = src.n_samples_shard

    def __len__(self):
        return self.len

    def __iter__(self):
        for data in self.dali_iterator:
            batch = data[0]
            yield batch["s0"], batch["s1"], batch["index"], batch["j"]


def get_dataloader(
    *,
    global_rank,
    world_size,
    device_id,
    batch_size,
    num_workers,
    split,
    mock: bool,
    grid_type: str = "healpix",
    plevel: int = 4,
    tile_size: int = 256,
    level_start: int = 3,
    level_end: int = 128,
    num_steps: int = 1,
    load_final_target_only: bool = False,
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
    use_time_variation_seed: bool = False,
    use_fixed_seed: bool = False,
    split_by_time: bool = False,
    train_start_index: int = 0,
    train_end_index: int = None,
    test_start_index: int = None,
    test_end_index: int = None,
    test_stride: int = 1,
    cross_face_tiles: bool = False,
    balance_cross_face: bool = False,
    shuffle_tiles: bool = True,
    train_main_zarr_paths: tuple[str, ...] | None = None,
    train_aux_zarr_paths: tuple[str, ...] | None = None,
    train_zarr_weights: tuple[float, ...] | None = None,
    train_start_indices: tuple[int, ...] | None = None,
    train_end_indices: tuple[int | None, ...] | None = None,
    main_zarr_path: str | None = None,
    aux_zarr_path: str | None = None,
    cubesphere_ne: int = 1024,
    cubesphere_npg: int = 2,
    use_duo_padding: bool = False,
    duo_padding_scrip_src_path: str | None = None,
    duo_padding_scrip_tgt_path: str | None = None,
    skip_corner_tiles: bool = False,
    index_is_latlon: bool = False,
    latlon_path: str | None = None,
):
    grid_type = (grid_type or "healpix").lower()
    if grid_type not in {"healpix", "cubesphere"}:
        raise ValueError(
            f"Unsupported grid_type='{grid_type}'. Expected 'healpix' or 'cubesphere'."
        )

    # Compute nside based on grid type
    if grid_type == "cubesphere":
        nside = cubesphere_ne * cubesphere_npg
    else:
        nside = 1024  # default HEALPix nside

    num_steps = max(1, int(num_steps))
    if cross_face_tiles:
        # Choose single-source or multi-source cross-face loader
        if split == "train" and train_main_zarr_paths:
            src = MultiGlobalCrossFaceSrc(
                batch_size=1,
                split=split,
                num_shards=world_size,
                shard_id=global_rank,
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
                train_start_index=train_start_index,
                train_end_index=train_end_index,
                test_start_index=test_start_index,
                test_end_index=test_end_index,
                test_stride=test_stride,
                nside=nside,
                grid_type=grid_type,
                cubesphere_ne=cubesphere_ne,
                cubesphere_npg=cubesphere_npg,
                main_zarr_paths=train_main_zarr_paths,
                aux_zarr_paths=train_aux_zarr_paths,
                zarr_weights=train_zarr_weights,
                train_start_indices=train_start_indices,
                train_end_indices=train_end_indices,
                use_duo_padding=use_duo_padding,
                duo_padding_scrip_src_path=duo_padding_scrip_src_path,
                duo_padding_scrip_tgt_path=duo_padding_scrip_tgt_path,
                index_is_latlon=index_is_latlon,
                latlon_path=latlon_path,
            )
        else:
            src = GlobalCrossFaceSrc(
                batch_size=1,
                split=split,
                num_shards=world_size,
                shard_id=global_rank,
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
                train_start_index=train_start_index,
                train_end_index=train_end_index,
                test_start_index=test_start_index,
                test_end_index=test_end_index,
                test_stride=test_stride,
                nside=nside,
                grid_type=grid_type,
                cubesphere_ne=cubesphere_ne,
                cubesphere_npg=cubesphere_npg,
                main_zarr_path=main_zarr_path,
                aux_zarr_path=aux_zarr_path,
                use_duo_padding=use_duo_padding,
                duo_padding_scrip_src_path=duo_padding_scrip_src_path,
                duo_padding_scrip_tgt_path=duo_padding_scrip_tgt_path,
                index_is_latlon=index_is_latlon,
                latlon_path=latlon_path,
            )

        # When index_is_latlon, index is [2,H,W] FLOAT (lat/lon radians);
        # otherwise [H,W] INT64 (pixel indices).
        idx_dtype = types.FLOAT if index_is_latlon else types.INT64
        idx_layout = "CHW" if index_is_latlon else "HW"

        pipeline = Pipeline(
            batch_size=1,
            device_id=device_id,
            num_threads=4,
            py_start_method="spawn",
            py_num_workers=num_workers,
            prefetch_queue_depth=1,
        )

        with pipeline:
            if num_steps == 1:
                s0, s1, index, j = fn.external_source(
                    source=src,
                    num_outputs=4,
                    batch=True,
                    dtype=(types.FLOAT, types.FLOAT, idx_dtype, types.INT64),
                    layout=["CHW", "CHW", idx_layout, ""],
                    batch_info=True,
                    parallel=True,
                    prefetch_queue_depth=1,
                )
                pipeline.set_outputs(s0.gpu(), s1.gpu(), index.gpu(), j.gpu())
            else:
                # For num_steps>1: s0, s_outputs [T,C,H,W], index, j, s_forcings [T-1,C,H,W]
                # Total outputs: 5
                s0, s_outputs, index, j, s_forcings = fn.external_source(
                    source=src,
                    num_outputs=5,
                    batch=True,
                    dtype=(
                        types.FLOAT,
                        types.FLOAT,
                        idx_dtype,
                        types.INT64,
                        types.FLOAT,
                    ),
                    layout=["CHW", "TCHW", idx_layout, "", "TCHW"],
                    batch_info=True,
                    parallel=True,
                    prefetch_queue_depth=1,
                )
                pipeline.set_outputs(
                    s0.gpu(), s_outputs.gpu(), index.gpu(), j.gpu(), s_forcings.gpu()
                )

        pipeline.start_py_workers()
        pipeline.build()

        if num_steps == 1:
            iterator = DALIGenericIterator(
                pipelines=pipeline,
                output_map=["s0", "s1", "index", "j"],
                auto_reset=True,
            )
        else:
            iterator = DALIGenericIterator(
                pipelines=pipeline,
                output_map=["s0", "s_outputs", "index", "j", "s_forcings"],
                auto_reset=True,
            )

        iterator = dataloader_transforms.WithLength(iterator, src.full_iterations)
        iterator = dataloader_transforms.ToTuple(iterator, num_steps=num_steps)
        # Subtile one padded face into overlapped tiles
        iterator = dataloader_transforms.CrossFaceSubTiler(
            iterator,
            nside=src.nside,
            tile_size=tile_size,
            shuffle_tiles=shuffle_tiles,
            num_steps=num_steps,
            skip_corner_tiles=skip_corner_tiles,
            balance_cross_face=balance_cross_face,
        )
        iterator = dataloader_transforms.Unbatch(iterator, batch_size=1)
        iterator = dataloader_transforms.Batch(iterator, batch_size)
        return iterator
    if split == "train" and train_main_zarr_paths:
        src = MultiScreamV2(
            batch_size=1,
            split=split,
            num_shards=world_size,
            shard_id=global_rank,
            mock=mock,
            grid_type=grid_type,
            plevel=plevel,
            level_start=level_start,
            level_end=level_end,
            variables_prognostic=variables_prognostic,
            variables_forcing=variables_forcing,
            variables_diagnostic=variables_diagnostic,
            use_time_variation_seed=use_time_variation_seed,
            use_fixed_seed=use_fixed_seed,
            split_by_time=split_by_time,
            train_start_index=train_start_index,
            train_end_index=train_end_index,
            test_start_index=test_start_index,
            test_end_index=test_end_index,
            test_stride=test_stride,
            num_steps=num_steps,
            load_final_target_only=load_final_target_only,
            main_zarr_paths=train_main_zarr_paths,
            aux_zarr_paths=train_aux_zarr_paths,
            zarr_weights=train_zarr_weights,
            train_start_indices=train_start_indices,
            train_end_indices=train_end_indices,
            index_is_latlon=index_is_latlon,
            latlon_path=latlon_path,
        )
    else:
        src = ScreamV2(
            batch_size=1,
            split=split,
            num_shards=world_size,
            shard_id=global_rank,
            mock=mock,
            grid_type=grid_type,
            plevel=plevel,
            level_start=level_start,
            level_end=level_end,
            variables_prognostic=variables_prognostic,
            variables_forcing=variables_forcing,
            variables_diagnostic=variables_diagnostic,
            use_time_variation_seed=use_time_variation_seed,
            use_fixed_seed=use_fixed_seed,
            split_by_time=split_by_time,
            train_start_index=train_start_index,
            train_end_index=train_end_index,
            test_start_index=test_start_index,
            test_end_index=test_end_index,
            test_stride=test_stride,
            num_steps=num_steps,
            load_final_target_only=load_final_target_only,
            main_zarr_path=main_zarr_path,
            aux_zarr_path=aux_zarr_path,
            index_is_latlon=index_is_latlon,
            latlon_path=latlon_path,
        )

    # When index_is_latlon, index is [2,H,W] FLOAT (lat/lon radians);
    # otherwise [H,W] INT64 (pixel indices).
    idx_dtype = types.FLOAT if index_is_latlon else types.INT64
    idx_layout = "CHW" if index_is_latlon else "HW"

    pipeline = Pipeline(
        batch_size=1,
        device_id=device_id,
        num_threads=4,
        py_start_method="spawn",
        py_num_workers=num_workers,
        prefetch_queue_depth=1,
    )

    with pipeline:
        if num_steps == 1:
            s0, s1, index, j = fn.external_source(
                source=src,
                num_outputs=4,
                batch=True,
                dtype=(types.FLOAT, types.FLOAT, idx_dtype, types.INT64),
                layout=["CHW", "CHW", idx_layout, ""],
                batch_info=True,
                parallel=True,
                prefetch_queue_depth=4 * num_workers,
            )
            pipeline.set_outputs(s0.gpu(), s1.gpu(), index.gpu(), j.gpu())
        else:
            # For num_steps>1: s0, s_outputs [T,C,H,W], index, j, s_forcings [T-1,C,H,W]
            # Total outputs: 5
            s0, s_outputs, index, j, s_forcings = fn.external_source(
                source=src,
                num_outputs=5,
                batch=True,
                dtype=(types.FLOAT, types.FLOAT, idx_dtype, types.INT64, types.FLOAT),
                layout=["CHW", "TCHW", idx_layout, "", "TCHW"],
                batch_info=True,
                parallel=True,
                prefetch_queue_depth=4 * num_workers,
            )
            pipeline.set_outputs(
                s0.gpu(), s_outputs.gpu(), index.gpu(), j.gpu(), s_forcings.gpu()
            )

    pipeline.start_py_workers()
    pipeline.build()

    if num_steps == 1:
        iterator = DALIGenericIterator(
            pipelines=pipeline,
            output_map=["s0", "s1", "index", "j"],
            auto_reset=True,
        )
    else:
        iterator = DALIGenericIterator(
            pipelines=pipeline,
            output_map=["s0", "s_outputs", "index", "j", "s_forcings"],
            auto_reset=True,
        )

    # transforms
    iterator = dataloader_transforms.WithLength(iterator, src.full_iterations)
    iterator = dataloader_transforms.ToTuple(iterator, num_steps=num_steps)
    iterator = dataloader_transforms.SubTiler(
        iterator,
        nside=src.nside,
        tile_size=tile_size,
        shuffle=shuffle_tiles,
        num_steps=num_steps,
    )
    iterator = dataloader_transforms.Unbatch(iterator, batch_size=1)
    iterator = dataloader_transforms.Batch(iterator, batch_size)

    return iterator
