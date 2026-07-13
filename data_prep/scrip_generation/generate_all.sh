#!/usr/bin/env bash
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
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="data"
HALO="256"

usage() {
    cat <<EOF
Usage: bash data_prep/scrip_generation/generate_all.sh [--output-dir DIR] [--halo N]

Options:
  --output-dir DIR  Shared output directory for both workflows (default: data)
  --halo N          Halo size for the halo workflow (default: 256)
  -h, --help        Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)
            [[ $# -lt 2 || "$2" == --* ]] && { echo "Error: --output-dir requires a value" >&2; exit 1; }
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --halo)
            [[ $# -lt 2 || "$2" == --* ]] && { echo "Error: --halo requires a value" >&2; exit 1; }
            HALO="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

echo "Generating no-halo Exodus meshes..."
python data_prep/scrip_generation/generate_exodus_meshes.py --res 1024 --np 2 --halo 0 --out-dir "${OUTPUT_DIR}"

echo "Converting no-halo pg mesh to SCRIP..."
python data_prep/scrip_generation/scrip_convert.py --in-exodus "${OUTPUT_DIR}/ne1024pg2.g"

echo "Deriving per-column lat/lon from the no-halo SCRIP..."
python data_prep/scrip_generation/derive_latlon.py \
    --in-scrip "${OUTPUT_DIR}/ne1024pg2_scrip.nc" \
    --out "${OUTPUT_DIR}/latlon_ne1024pg2.nc"

echo "Generating halo Exodus meshes..."
python data_prep/scrip_generation/generate_exodus_meshes.py --res 1024 --np 2 --halo "${HALO}" --out-dir "${OUTPUT_DIR}"

echo "Converting halo pg mesh to SCRIP..."
python data_prep/scrip_generation/scrip_convert.py --in-exodus "${OUTPUT_DIR}/ne1024halo${HALO}pg2.g"

echo "Done."
echo "Created ${OUTPUT_DIR}/ne1024pg2_scrip.nc"
echo "Created ${OUTPUT_DIR}/latlon_ne1024pg2.nc"
echo "Created ${OUTPUT_DIR}/ne1024halo${HALO}pg2_scrip.nc"
