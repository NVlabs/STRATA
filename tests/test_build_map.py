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

from initial_conditions.build_map import build, looks_complete  # noqa: E402

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


def _assert_same_map(new_path, ref_path):
    """Every variable of the serial map, to rounding."""
    with netCDF4.Dataset(ref_path) as ref, netCDF4.Dataset(new_path) as new:
        assert dict(new.dimensions.items()).keys() == ref.dimensions.keys()
        for name, dim in ref.dimensions.items():
            assert new.dimensions[name].size == dim.size, name
        for name, var in ref.variables.items():
            got, want = np.asarray(new[name][:]), np.asarray(var[:])
            if np.issubdtype(want.dtype, np.integer):
                np.testing.assert_array_equal(got, want, err_msg=name)
            else:
                np.testing.assert_allclose(got, want, rtol=1e-12, atol=0, err_msg=name)


def test_parallel_build_matches_the_serial_map(grids, tmp_path):
    d, src, dst, serial = grids
    # a long working path: TempestRemap 2.2.0 aborts on long output paths
    tmp_root = tmp_path / ("x" * 120)
    tmp_root.mkdir()
    (tmp_root / "keep.txt").write_text("not ours")
    part_dir = str(tmp_root / "map.nc.parts")
    out = str(tmp_path / "map.nc")
    build(src, dst, out, workers=2, n_parts=3, part_dir=part_dir, history="test")

    _assert_same_map(out, serial)
    assert (tmp_root / "keep.txt").read_text() == "not ours"
    assert not (tmp_root / "map.nc.parts").exists()
    with netCDF4.Dataset(out) as new:
        # first-order conservative: every destination row sums to one
        ncol = new.dimensions["n_b"].size
        row_sum = np.bincount(np.asarray(new["row"][:]) - 1, new["S"][:], ncol)
        np.testing.assert_allclose(row_sum, 1.0, rtol=0, atol=1e-12)
        assert new.getncattr("history") == "test"


def test_parts_from_other_grids_are_not_reused(grids, tmp_path):
    d, src, dst, serial = grids
    part_dir = tmp_path / "map.nc.parts"
    part_dir.mkdir()
    (part_dir / "inputs.txt").write_text("src /other/src.nc\ndst /other/dst.nc\n")
    with pytest.raises(SystemExit, match="other grids"):
        build(src, dst, str(tmp_path / "map.nc"), 1, 2, str(part_dir), "test")


def test_only_complete_maps_are_adopted(grids, tmp_path):
    d, src, dst, serial = grids
    assert looks_complete(serial)
    header_only = str(tmp_path / "header_only.nc")
    with netCDF4.Dataset(serial) as ref, netCDF4.Dataset(header_only, "w") as out:
        for name, dim in ref.dimensions.items():
            out.createDimension(name, dim.size)
        for name, var in ref.variables.items():
            out.createVariable(name, var.dtype, var.dimensions)
    assert not looks_complete(header_only)
