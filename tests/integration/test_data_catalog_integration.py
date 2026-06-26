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

import numpy as np
import pytest

import data_catalog

if "SCREAM_MAIN_ZARR_PATH" not in os.environ:
    pytest.skip(
        "Dataset not configured. Set SCREAM_MAIN_ZARR_PATH", allow_module_level=True
    )


def test_scream_sdecadal_xarray_matches_zarr():
    ds = data_catalog.scream_sdecadal.to_xarray(chunks=None)
    group = data_catalog.scream_sdecadal.to_zarr()

    np.testing.assert_array_equal(ds["T_2m"][1].values, group["T_2m"][1])
