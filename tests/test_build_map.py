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
# ruff: noqa: S603 S607
"""The parallel map build against a serial ``ncremap -a fv2fv_flx --lrg2sml``.

Needs conda-forge ``nco`` and ``tempest-remap`` on PATH; skipped otherwise.
"""
from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

from initial_conditions.build_map import build_part, merge_parts  # noqa: E402

TOOLS = ("ncremap", "GenerateCSMesh", "GenerateOverlapMesh", "GenerateOfflineMap")
pytestmark = pytest.mark.skipif(
    any(shutil.which(tool) is None for tool in TOOLS),
    reason="needs nco and tempest-remap on PATH",
)


def _run(cmd, cwd):
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True)


@pytest.fixture(scope="module")
def grids(tmp_path_factory):
    """A 10-degree lat-lon source and an ne4pg2 destination (384 columns)."""
    d = tmp_path_factory.mktemp("grids")
    src = str(d / "src_19x36.nc")
    _run(
        ["ncremap", f"--tmp_dir={d}", "-G"]
        + ["ttl=test#latlon=19,36#lat_typ=uni#lat_drc=n2s#lon_typ=grn_ctr", "-g", src],
        d,
    )
    _run(["GenerateCSMesh", "--res", "4", "--file", "ne4.g"], d)
    _run(
        ["GenerateVolumetricMesh", "--in", "ne4.g", "--out", "ne4pg2.g"]
        + ["--np", "2", "--uniform"],
        d,
    )
    dst = str(d / "ne4pg2_scrip.nc")
    _run(["ConvertMeshToSCRIP", "--in", "ne4pg2.g", "--out", dst], d)
    serial = str(d / "map_serial.nc")
    _run(
        ["ncremap", "-a", "fv2fv_flx", f"--src_grd={src}", f"--dst_grd={dst}"]
        + [f"--map_file={serial}", f"--tmp_dir={d}", "--lrg2sml"],
        d,
    )
    return d, src, dst, serial


def test_parts_stack_into_the_serial_map(grids):
    d, src, dst, serial = grids
    with netCDF4.Dataset(dst) as ds:
        ncol = ds.dimensions["grid_size"].size
    # uneven ranges, so part boundaries fall at different places on the cube
    bounds = np.array([0, 50, 200, ncol])
    part_dir = d / "parts"
    part_dir.mkdir()
    parts = [
        build_part(src, dst, str(part_dir), k, int(bounds[k]), int(bounds[k + 1]))
        for k in range(len(bounds) - 1)
    ]
    merged = str(d / "map_parallel.nc")
    merge_parts(parts, bounds, merged, dst)

    with netCDF4.Dataset(serial) as ref, netCDF4.Dataset(merged) as new:
        assert new.dimensions["n_b"].size == ncol
        for name in ("row", "col"):
            np.testing.assert_array_equal(new[name][:], ref[name][:])
        np.testing.assert_allclose(new["S"][:], ref["S"][:], rtol=1e-9, atol=0)
        for name in ("area_b", "frac_b", "xc_b", "yc_b", "xv_b", "yv_b"):
            np.testing.assert_allclose(new[name][:], ref[name][:], rtol=1e-9, atol=0)
        np.testing.assert_allclose(new["frac_a"][:], ref["frac_a"][:], rtol=1e-9)
        # first-order conservative: every destination row sums to one
        row_sum = np.bincount(np.asarray(new["row"][:]) - 1, new["S"][:], ncol)
        np.testing.assert_allclose(row_sum, 1.0, rtol=0, atol=1e-12)
