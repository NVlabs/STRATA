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
from types import SimpleNamespace

import pytest

from screamcast.dali_ext_src import GlobalCrossFaceSrc

# Use the HEALPix dataset for tests in this module.
HEALPIX_MAIN_ZARR_PATH = "s3://SCREAM_zarrv3/sdecadal.ne1024pg2_ne1024pg2.F20TR-SCREAMv1.c10-sep11.out10min.zarr"
HEALPIX_AUX_ZARR_PATH = "s3://SCREAM_zarrv3/sdecadal.ne1024pg2_ne1024pg2.F20TR-SCREAMv1.c10-sep11.out10min.zarr"

# Skip tests that require S3/rclone access when running in CI or when explicitly requested
skip_s3 = (
    os.getenv("GITLAB_CI") == "true"
    or os.getenv("CI") == "true"
    or os.getenv("SCREAM_SKIP_S3_TESTS") in {"1", "true", "True"}
)
pytestmark = pytest.mark.skipif(
    skip_s3, reason="S3/rclone not available in CI environment"
)


@pytest.fixture(autouse=True)
def _set_healpix_zarr_paths(monkeypatch):
    # Make the data source explicit so these tests don't depend on whatever happens
    # to be in the user's environment.
    monkeypatch.setenv("SCREAM_MAIN_ZARR_PATH", HEALPIX_MAIN_ZARR_PATH)
    monkeypatch.setenv("SCREAM_AUX_ZARR_PATH", HEALPIX_AUX_ZARR_PATH)


def _mk_cross_src(two_step: bool):
    return GlobalCrossFaceSrc(
        batch_size=1,
        split="train",
        num_shards=1,
        shard_id=0,
        plevel=4,
        level_start=3,
        level_end=128,
        variables_prognostic=("T_2m",),
        variables_forcing=("coszr",),
        variables_diagnostic=(),
        num_steps=2 if two_step else 1,
        tile_size=32,
        use_time_variation_seed=False,
        use_fixed_seed=True,
        train_start_index=1,
        train_end_index=2,
        test_start_index=None,
        test_end_index=None,
        test_stride=1,
        nside=1024,
        mock=True,
    )


def test_global_cross_face_src_one_step():
    src = _mk_cross_src(two_step=False)
    assert len(src) == 12  # one timestep * 12 faces

    batch_info = SimpleNamespace(iteration=0, epoch_idx=0)
    s0, s1, index, j = src(batch_info)

    assert len(s0) == 1 and len(s1) == 1 and len(index) == 1 and len(j) == 1
    Hpad = src.nside + 2 * src.overlap_size
    assert s0[0].shape == (src._in_channels, Hpad, Hpad)
    assert s1[0].shape == (src._out_channels, Hpad, Hpad)
    assert index[0].shape == (Hpad, Hpad)
    assert j[0].shape == (1,)


def test_global_cross_face_src_two_step():
    src = _mk_cross_src(two_step=True)
    assert len(src) == 12

    batch_info = SimpleNamespace(iteration=0, epoch_idx=0)
    s0, s1, index, j, sF1 = src(batch_info)

    assert (
        len(s0) == 1
        and len(s1) == 1
        and len(index) == 1
        and len(j) == 1
        and len(sF1) == 1
    )
    Hpad = src.nside + 2 * src.overlap_size
    assert s0[0].shape == (src._in_channels, Hpad, Hpad)
    assert s1[0].shape == (1, src._out_channels, Hpad, Hpad)
    assert index[0].shape == (Hpad, Hpad)
    assert j[0].shape == (1,)
    assert sF1[0].shape == (1, src._forcing_channels, Hpad, Hpad)
