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
import torch

from screamcast.horizontal_regridding import (
    LatLonToPointGridRegridder,
    UnstructuredToLatLonRegridder,
)


def test_unstructured_to_latlon_regridder_handles_flat_and_gridded_inputs():
    regridder = UnstructuredToLatLonRegridder(
        source_lon=np.array([[0.0, 90.0], [180.0, 270.0]], dtype=np.float32),
        source_lat=np.array([[-45.0, -45.0], [45.0, 45.0]], dtype=np.float32),
        target_lat=np.array([-30.0, 30.0], dtype=np.float32),
        target_lon=np.array([45.0, 225.0], dtype=np.float32),
        source_hpx_level=6,
        target_hpx_level=6,
    )

    y_gridded = regridder(torch.arange(8, dtype=torch.float32).reshape(2, 2, 2))
    y_flat = regridder(torch.arange(8, dtype=torch.float32).reshape(2, 4))

    assert tuple(y_gridded.shape) == (2, 2, 2)
    assert tuple(y_flat.shape) == (2, 2, 2)
    torch.testing.assert_close(y_gridded, y_flat, equal_nan=True)

    try:
        regridder(torch.zeros(2, 3, dtype=torch.float32))
        raise AssertionError(
            "Expected ValueError for mismatched trailing source shape."
        )
    except ValueError as exc:
        assert "trailing shape" in str(exc)


def test_latlon_to_point_grid_regridder_handles_flat_and_gridded_inputs():
    regridder = LatLonToPointGridRegridder(
        target_lon=np.array([[45.0, 225.0], [45.0, 225.0]], dtype=np.float32),
        target_lat=np.array([[-30.0, -30.0], [30.0, 30.0]], dtype=np.float32),
        source_lat=np.array([-30.0, 30.0], dtype=np.float32),
        source_lon=np.array([45.0, 225.0], dtype=np.float32),
    )

    y_gridded = regridder(torch.arange(8, dtype=torch.float32).reshape(2, 2, 2))
    y_flat = regridder(torch.arange(8, dtype=torch.float32).reshape(2, 4))

    assert tuple(y_gridded.shape) == (2, 2, 2)
    assert tuple(y_flat.shape) == (2, 2, 2)
    torch.testing.assert_close(y_gridded, y_flat, equal_nan=True)

    try:
        regridder(torch.zeros(2, 3, dtype=torch.float32))
        raise AssertionError(
            "Expected ValueError for mismatched trailing source shape."
        )
    except ValueError as exc:
        assert "trailing shape" in str(exc)


def test_latlon_to_point_grid_regridder_keeps_seam_and_out_of_bounds_lat_targets_finite():
    regridder = LatLonToPointGridRegridder(
        target_lon=np.array([0.0, 359.75, 1.0, 5.0], dtype=np.float32),
        target_lat=np.array([-90.0, -30.0, 30.0, 90.0], dtype=np.float32),
        source_lat=np.array([-30.0, 30.0], dtype=np.float32),
        source_lon=np.array([0.5, 90.5, 180.5, 270.5], dtype=np.float32),
    )

    y = regridder(torch.ones((1, 2, 4), dtype=torch.float32))

    assert torch.isfinite(y).all()
    torch.testing.assert_close(y, torch.ones_like(y))
