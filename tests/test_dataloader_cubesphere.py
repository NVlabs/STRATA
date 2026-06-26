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
import torch

from screamcast.cubesphere_transforms import (
    faces_to_unstructured,
    reorder_2d_tensor_to_cubesphere,
    reorder_cubesphere_to_2d_tensor,
    unstructured_to_6faces,
)
from screamcast.dali_ext_src import (
    GlobalCrossFaceSrc,
    MultiGlobalCrossFaceSrc,
    ScreamV2,
)
from screamcast.pipelines import get_dataloader

# Use the CubeSphere dataset for tests in this module.
CUBESPHERE_ZARR_PATH = "s3://SCREAM_zarrv3/sdecadal.ne1024pg2_ne1024pg2.F20TR-SCREAMv1.c10-sep11.out10min.cubesphere.zarr"

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
def _set_cubesphere_zarr_paths(monkeypatch):
    # Make the data source explicit so these tests don't depend on whatever happens
    # to be in the user's environment.
    monkeypatch.setenv("SCREAM_MAIN_ZARR_PATH", CUBESPHERE_ZARR_PATH)
    monkeypatch.setenv("SCREAM_AUX_ZARR_PATH", CUBESPHERE_ZARR_PATH)


def test_screamv2_cubesphere_basic_load_and_postprocess():
    # Note: this test avoids DALI and only checks ScreamV2 + postprocess.
    ds = ScreamV2(
        batch_size=1,
        split="",
        num_shards=1,
        shard_id=0,
        mock=True,
        grid_type="cubesphere",
        main_zarr_path=CUBESPHERE_ZARR_PATH,
        aux_zarr_path=CUBESPHERE_ZARR_PATH,
        # Keep channels tiny to avoid huge S3 reads
        plevel=1,
        level_start=0,
        level_end=1,
        variables_prognostic=("T_2m",),
        variables_forcing=(),
        variables_diagnostic=(),
    )

    # CubeSphere defaults ne=1024, npg=2 -> face side 2048, total ncol = 6 * 2048^2
    assert ds.nside == 2048
    assert ds.patch_size == 2048 * 2048
    assert ds.cells == 6 * 2048 * 2048
    assert ds.npatches == 6

    # Load one face worth of a single 2D channel and reshape to [C,H,W]
    arr_1d = ds.load_patch_input(t=1, patch=0)  # [1, patch_size]
    assert arr_1d.shape == (1, ds.patch_size)
    arr_2d = ds._post_process(arr_1d)
    assert torch.is_tensor(arr_2d)
    assert arr_2d.shape == (1, ds.nside, ds.nside)


def test_get_dataloader_cubesphere_one_batch():
    # This mirrors the HEALPix test in tests/test_dataloader.py but for CubeSphere.
    # It requires DALI + CUDA.
    pytest.importorskip("nvidia.dali")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Keep reads small (1 channel, 1 level, small tile) to avoid heavy I/O.
    variables_prognostic = ("T_2m", "U")
    variables_forcing = ()
    variables_diagnostic = ()

    dataloader = get_dataloader(
        global_rank=0,
        world_size=1,
        device_id=0,
        batch_size=1,
        num_workers=1,
        split="train",
        mock=True,
        grid_type="cubesphere",
        plevel=1,
        tile_size=32,
        level_start=0,
        level_end=4,
        variables_prognostic=variables_prognostic,
        variables_forcing=variables_forcing,
        variables_diagnostic=variables_diagnostic,
        split_by_time=False,
    )
    inp, tar, index, j = next(iter(dataloader))

    in_ch = ScreamV2.num_of_input_channels(
        variables_prognostic=variables_prognostic,
        variables_forcing=variables_forcing,
        plevel=1,
        level_start=0,
        level_end=4,
    )
    out_ch = ScreamV2.num_of_output_channels(
        variables_prognostic=variables_prognostic,
        variables_diagnostic=variables_diagnostic,
        plevel=1,
        level_start=0,
        level_end=4,
    )
    assert in_ch == 5
    assert out_ch == 5
    assert inp.shape == (1, in_ch, 32, 32)
    assert tar.shape == (1, out_ch, 32, 32)
    assert index.shape == (1, 32, 32)
    assert j.shape[0] == 1


def _mk_cross_src_cubesphere(two_step: bool, ne: int = 1024, npg: int = 2):
    return GlobalCrossFaceSrc(
        batch_size=1,
        split="train",
        num_shards=1,
        shard_id=0,
        mock=True,
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
        nside=ne * npg,
        grid_type="cubesphere",
        cubesphere_ne=ne,
        cubesphere_npg=npg,
    )


def test_global_cross_face_src_cubesphere_one_step():
    src = _mk_cross_src_cubesphere(two_step=False)
    assert src.n_faces == 6
    assert len(src) == 6  # one timestep * 6 faces

    batch_info = SimpleNamespace(iteration=0, epoch_idx=0)
    s0, s1, index, j = src(batch_info)

    assert len(s0) == 1 and len(s1) == 1 and len(index) == 1 and len(j) == 1
    Hpad = src.nside + 2 * src.overlap_size
    assert s0[0].shape == (src._in_channels, Hpad, Hpad)
    assert s1[0].shape == (src._out_channels, Hpad, Hpad)
    assert index[0].shape == (Hpad, Hpad)
    assert j[0].shape == (1,)


def test_global_cross_face_src_cubesphere_two_step():
    src = _mk_cross_src_cubesphere(two_step=True)
    assert src.n_faces == 6
    assert len(src) == 6

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


def test_reorder_inverse_functions():
    """Test that reorder_2d_tensor_to_cubesphere inverts reorder_cubesphere_to_2d_tensor."""
    ne, npg = 4, 2  # Small values for fast testing
    npts = ne * ne * npg * npg  # 64

    # Test with various batch dimensions
    for shape in [(npts,), (3, npts), (2, 5, npts)]:
        original = torch.randn(shape)

        # Forward: cubesphere ordering -> 2D grid
        grid_2d = reorder_cubesphere_to_2d_tensor(original, ne=ne, npg=npg)
        expected_2d_shape = shape[:-1] + (ne * npg, ne * npg)
        assert grid_2d.shape == expected_2d_shape

        # Inverse: 2D grid -> cubesphere ordering
        reconstructed = reorder_2d_tensor_to_cubesphere(grid_2d, ne=ne, npg=npg)
        assert reconstructed.shape == original.shape
        assert torch.allclose(reconstructed, original)

    ne, npg = 4, 2
    total_pts = 6 * (ne * npg) ** 2

    # Use sequential values to easily verify correctness
    original = torch.arange(total_pts, dtype=torch.float32).reshape(1, 1, total_pts)
    faces_2d = unstructured_to_6faces(original, ne=ne, npg=npg)
    reconstructed = faces_to_unstructured(faces_2d, ne=ne, npg=npg)

    assert torch.allclose(reconstructed, original)


# ---------------------------------------------------------------------------
# MultiGlobalCrossFaceSrc tests
# ---------------------------------------------------------------------------


def _mk_multi_cross_src(train_end_index=3):
    """Helper: two-source MultiGlobalCrossFaceSrc in mock mode."""
    return MultiGlobalCrossFaceSrc(
        batch_size=1,
        split="train",
        num_shards=1,
        shard_id=0,
        mock=True,
        plevel=4,
        level_start=3,
        level_end=128,
        variables_prognostic=("T_2m",),
        variables_forcing=("coszr",),
        variables_diagnostic=(),
        num_steps=1,
        load_final_target_only=True,
        tile_size=32,
        use_time_variation_seed=False,
        use_fixed_seed=True,
        train_start_index=1,
        train_end_index=train_end_index,
        test_start_index=None,
        test_end_index=None,
        test_stride=1,
        grid_type="cubesphere",
        cubesphere_ne=1024,
        cubesphere_npg=2,
        nside=2048,
        # Use the same S3 zarr path for both sources (mock mode clones metadata)
        main_zarr_paths=[CUBESPHERE_ZARR_PATH, CUBESPHERE_ZARR_PATH],
    )


def test_multi_global_cross_face_src_shapes():
    """Two mock sources produce correct shapes and combined sample count."""
    src = _mk_multi_cross_src(train_end_index=3)
    # 2 sources × 2 times × 6 faces = 24 samples
    assert src.n_samples_shard == 24
    assert src.full_iterations == 24

    batch_info = SimpleNamespace(iteration=0, epoch_idx=0)
    s0, s1, index, j = src(batch_info)

    Hpad = src.nside + 2 * src.sources[0].overlap_size
    assert s0[0].shape == (src._in_channels, Hpad, Hpad)
    assert s1[0].shape == (src._out_channels, Hpad, Hpad)
    assert index[0].shape == (Hpad, Hpad)
    assert j[0].shape == (1,)


def test_multi_global_cross_face_src_block_contiguous_and_cache():
    """Epoch samples are grouped by (source, time) blocks of n_faces,
    and only one source holds a cache at any point during iteration."""
    src = _mk_multi_cross_src(train_end_index=3)
    n_faces = src.n_faces  # 6

    # Force epoch sample construction
    src._build_epoch_samples(epoch_idx=0)

    # Verify block-contiguous property: every consecutive group of n_faces
    # samples shares the same (src_id, time_slot).
    for block_start in range(0, len(src._epoch_samples), n_faces):
        block = src._epoch_samples[block_start : block_start + n_faces]
        src_ids = {s for s, t, f in block}
        time_slots = {t for s, t, f in block}
        faces = sorted(f for s, t, f in block)
        assert len(src_ids) == 1, "Block must belong to a single source"
        assert len(time_slots) == 1, "Block must belong to a single time slot"
        assert faces == list(
            range(n_faces)
        ), "Block must contain all faces exactly once"

    # Iterate all 24 samples; verify cache hygiene — after each __call__
    # at most one source should have a live cache.
    for it in range(src.full_iterations):
        batch_info = SimpleNamespace(iteration=it, epoch_idx=0)
        src(batch_info)
        caches_alive = sum(
            1 for s in src.sources if getattr(s, "_cache_t", None) is not None
        )
        assert (
            caches_alive <= 1
        ), f"iteration {it}: {caches_alive} sources have a live cache (expected ≤ 1)"
