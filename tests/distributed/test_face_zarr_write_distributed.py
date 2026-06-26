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
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import zarr

from screamcast.distributed_halo import TileTopology
from screamcast.zarr_writer import (
    FaceZarrWriteStep,
    _split_variable_levels,
    prepare_output_store,
)


def _shared_output_path() -> Path:
    master_port = os.environ.get("MASTER_PORT", "unknown")
    root = Path(tempfile.gettempdir()) / f"screamcast-dist-write-{master_port}"
    return root / "face_zarr_write_test.zarr"


def test_face_zarr_write_step_distributed_zero_batch(distributed_cuda_context):
    rank, world_size, device = distributed_cuda_context
    output_path = _shared_output_path()

    variable_names = ["A", "B", "C", "D", "E", "F", "G", "H_0", "H_1"]
    grouped = _split_variable_levels(variable_names)

    topology = TileTopology(
        world_size=world_size,
        rank=rank,
        face_size=4,
        tile_size=2,
        halo_width=0,
    )

    assert topology.tiles_per_rank == 3

    if rank == 0:
        shutil.rmtree(output_path.parent, ignore_errors=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lat = np.zeros((6, 4, 4), dtype=np.float32)
        lon = np.zeros((6, 4, 4), dtype=np.float32)
        store = prepare_output_store(
            output_path=str(output_path),
            grouped_variables=grouped,
            n_times=1,
            n_steps=1,
            nside=4,
            tile_size=2,
            time_values=np.array(["2020-01-01"], dtype="datetime64[s]"),
            lat=lat,
            lon=lon,
        )
        for base_name in grouped:
            store[base_name][:] = np.nan

    dist.barrier()

    store = zarr.open_group(str(output_path), mode="a")
    write_step = FaceZarrWriteStep(
        output_steps=1,
        topology=topology,
        grouped_variables=grouped,
        store=store,
    )

    out_tensor = torch.zeros(
        (
            1,
            len(variable_names),
            topology.tiles_per_rank,
            topology.tile_size,
            topology.tile_size,
        ),
        device=device,
        dtype=torch.float32,
    )
    write_step(step_index=0, out_tensor=out_tensor)
    write_step.close()

    dist.barrier()

    if rank == 0:
        verify_store = zarr.open_group(str(output_path), mode="r")
        for base_name, entries in grouped.items():
            data = verify_store[base_name][:]
            assert not np.isnan(data).any(), f"{base_name} still contains NaNs"
            assert np.all(data == 0.0), f"{base_name} was not written as zeros"
            if entries[0][0] is None:
                assert data.shape == (1, 1, 6, 4, 4)
            else:
                assert data.shape == (1, 1, len(entries), 6, 4, 4)

    dist.barrier()

    if rank == 0:
        shutil.rmtree(output_path.parent, ignore_errors=True)

    dist.barrier()
