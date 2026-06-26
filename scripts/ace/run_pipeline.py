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

import data_catalog
from scripts.ace import regrid_zarr
from scripts.ace.build_ace_forecast_pairs import build_ace_forecast_pairs

TEST = False


base = "../../project-data/screamcast/inferences/pixeldit_sem1024d24l_pix128d4l_2stepft/sdecadal.ne1024pg2_ne1024pg2.F20TR-SCREAMv1.c10-sep11/"
truth_data = "../../project-data/screamcast/sdecadal.ne1024pg2_ne1024pg2.F20TR-SCREAMv1.c10-sep11.6hr.ace.v2.zarr/"
forecast_data = f"{base}/6hour_forecasts.ace_grid.zarr"
rev2_data = f"{base}/ace_rev2_pairs.step36.nc"

if not os.path.exists(truth_data):
    regrid_zarr.regrid(
        input=data_catalog.scream_sdecadal,
        output=truth_data,
        selection={"time": slice(0, None, 36)},
        prefetch=1 if TEST else 4,
    )

if not os.path.exists(forecast_data):
    regrid_zarr.regrid(
        input=f"{base}/6hour_forecasts.zarr/",
        output=forecast_data,
        selection={"step": slice(35, 36)},
        prefetch=1 if TEST else 4,
    )

if not os.path.exists(rev2_data):
    build_ace_forecast_pairs(
        forecast_data=forecast_data,
        truth_data=truth_data,
        output=rev2_data,
        forecast_step=36,
        min_init_time="2020-10-02T00:00:00",
        prefetch=1 if TEST else 2,
    )
