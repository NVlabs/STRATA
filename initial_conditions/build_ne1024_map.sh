#!/bin/bash
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
# Build (or reuse) the ERA5 721x1440 -> ne1024pg2 map with ncremap fv2fv_flx --lrg2sml.
#
#   SRC_SCRIP=.../scrip_ERA5_721x1440.nc DST_SCRIP=.../ne1024pg2_scrip.nc \
#   MAP=.../map_721x1440_to_ne1024pg2.nc bash initial_conditions/build_ne1024_map.sh
#
# Serial GenerateOverlapMesh at this size takes ~18 h; prefer reusing an existing map
# when you have one. run_hiccup_eamxx.py reuses MAP only when MAP.command.txt
# sits next to it, so an existing map without one is adopted only after its source
# and destination sizes are checked (ncdump, from the NCO/netCDF environment).
set -euo pipefail
: "${SRC_SCRIP:?set SRC_SCRIP to the ERA5 721x1440 SCRIP file}"
: "${DST_SCRIP:?set DST_SCRIP to the ne1024pg2 SCRIP file}"
: "${MAP:?set MAP to the output map path}"
CMD="ncremap -a fv2fv_flx --src_grd=$SRC_SCRIP --dst_grd=$DST_SCRIP --map_file=$MAP --lrg2sml"
N_SRC=1038240     # 721 x 1440 ERA5 cells
N_DST=25165824    # ne1024pg2 columns
mkdir -p "$(dirname "$MAP")"
if [[ ! -s "$MAP" ]]; then
  $CMD
  echo "$CMD" > "${MAP}.command.txt"
elif [[ -f "${MAP}.command.txt" ]]; then
  echo "reusing $MAP"
else
  header=$(ncdump -h "$MAP")
  if ! grep -qE "^\s*n_a = ${N_SRC} ;" <<< "$header" || ! grep -qE "^\s*n_b = ${N_DST} ;" <<< "$header"; then
    echo "$MAP is not a ${N_SRC} -> ${N_DST} map (n_a/n_b); not adopting it" >&2
    exit 1
  fi
  echo "adopting existing $MAP (n_a=${N_SRC}, n_b=${N_DST})"
  echo "$CMD" > "${MAP}.command.txt"
fi
