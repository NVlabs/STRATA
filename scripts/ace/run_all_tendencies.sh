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
root=../../project-data/screamcast/
model=pixeldit_sem1024d24l_pix128d4l_2stepft
data=sdecadal.ne1024pg2_ne1024pg2.F20TR-SCREAMv1.c10-sep11

outputdir=$root/inferences/$model/$data
mkdir -p $outputdir

python3 scripts/ace/build_screamcast_forecast.py \
--checkpoint $root/$model/output/best.pth \
--output $outputdir/6hour_forecasts.zarr \
--n-times 40 \
--forecast-steps 36 \
--time-start 0 \
--tile-size 512 --device cuda
