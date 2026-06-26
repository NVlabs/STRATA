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
# ruff: noqa: S101
import matplotlib.pyplot as plt
import msc_netcdf_classic
import numpy as np

url = "msc://pbss/SCREAM/ndec.ne1024pg2_ne1024pg2.F20TR-SCREAMv1.c01-jan9.2d.n0512.mk.out10min/run/output.scream.10minINST_state_ne1024pg2.INSTANT.nmins_x10.1994-10-03-00000.nc"

# +
dataset = msc_netcdf_classic.MscNetCDF.from_url(url)
qv = dataset.variables["qv"]

lev = dataset.read_var("lev")
qv
# -

qv = dataset.variables["qv"][0, 0]
pt = dataset.variables["PotentialTemperature"][0, 0]

plt.plot(pt, lev)
plt.ylim(1000, 2)

# +
plt.plot(qv, lev, ".-")
for subsample in [4, 8, 16]:
    s = slice((lev.size - 1) % subsample, lev.size, subsample)
    plt.plot(qv[s], lev[s], ".-", label=f"{subsample=}")

plt.ylim(1001, 700)
plt.legend()
plt.ylabel("level")
plt.xlabel("qv")
# -

# 4 point subsampling seems adequate

plt.plot(lev.size - np.arange(lev.size), lev)
plt.xlabel("cumulative num levels from bottom")
plt.ylim(1001, 2)

density = np.diff(lev)
plt.plot(density, (lev[1:] + lev[:-1]) / 2)
plt.ylim(1000, 2)
