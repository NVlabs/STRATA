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
import numpy as np
import zarr

from screamcast.dali_ext_src import ScreamV2


def _mk_consolidated_zarr(tmp_path, *, nt: int, nlev: int, ncol: int):
    """
    Create a minimal consolidated Zarr store compatible with ScreamV2.
    """
    path = tmp_path / "main.zarr"
    g = zarr.open_group(str(path), mode="w")
    # Minimal variable required by ScreamV2.__init__
    g.create_dataset(
        "U",
        shape=(nt, nlev, ncol),
        chunks=(1, nlev, min(ncol, 64)),
        dtype="f4",
        fill_value=0.0,
    )
    zarr.consolidate_metadata(str(path))
    return str(path)


def test_screamv2_cubesphere_postprocess_matches_reference(tmp_path):
    # Small CubeSphere face: ne=2, npg=2 -> resolution=4 -> patch_size=16
    ne, npg = 2, 2
    resolution = ne * npg
    patch_size = resolution * resolution
    # Simulate 6 faces worth of columns
    nfaces = 6
    ncol = nfaces * patch_size
    nt, nlev = 4, 2

    main_path = _mk_consolidated_zarr(tmp_path, nt=nt, nlev=nlev, ncol=ncol)

    src = ScreamV2(
        batch_size=1,
        split="",
        num_shards=1,
        shard_id=0,
        mock=False,
        plevel=1,
        level_start=0,
        level_end=nlev,
        variables_prognostic=("U",),
        variables_forcing=(),
        variables_diagnostic=(),
        main_zarr_path=main_path,
        aux_zarr_path=main_path,
        resolution=resolution,
        grid_type="cubesphere",
        cubesphere_ne=ne,
        cubesphere_npg=npg,
    )

    idx = np.arange(patch_size, dtype=np.int64)
    idx_2d = src._post_process(idx).cpu().numpy()
    assert idx_2d.shape == (resolution, resolution)
