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

import dotenv
import numpy as np

from screamcast.catalog_core import ScreamData

dotenv.load_dotenv(".")

scream_sdecadal = ScreamData(
    os.getenv("SCREAM_MAIN_ZARR_PATH", ""),
    os.getenv("SCREAM_ZARR_PROFILE", ""),
    vertical_subset=slice(3, None, 4),
    grid_file=os.path.join(
        os.environ.get("AUX_DATA_ROOT", "data"), "latlon_ne1024pg2.nc"
    ),
    vertical_coordinate_file=os.path.join(
        os.environ.get("AUX_DATA_ROOT", "data"), "scream_vertical_coordinate.nc"
    ),
    reference_time=np.datetime64("2020-10-01T00:00:00"),
    native_timestep=np.timedelta64(10, "m"),
)


def scream():
    return scream_sdecadal.to_data_source()
