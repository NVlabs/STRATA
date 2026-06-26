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
#SBATCH --job-name=strata_rollout
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
#
# Public example: global cubed-sphere rollout with
# scripts/ace/run_screamcast_nudged.py. Adapt the SBATCH directives and the
# container/module setup to your cluster, and edit the paths below. None of the
# values here point at any specific cluster.
#
# --output-levels selects which sub-sampled sigma levels are written to the
# output zarr; widen or narrow the list to the levels you analyze.

set -euo pipefail

# --- edit these for your environment ---
RUN_NAME="my_run"
CHECKPOINT="/path/to/checkpoint/best.pth"
OUTPUT="/path/to/output_rollout/${RUN_NAME}.zarr"
INIT_TIME="2020-10-13T00:00:00"
# ----------------------------------------

srun python3 scripts/ace/run_screamcast_nudged.py \
    --run-name "${RUN_NAME}" \
    --checkpoint "${CHECKPOINT}" \
    --forecast-model screamcast \
    --correction none \
    --n-steps 144 \
    --tile-size 128 \
    --halo-width 16 \
    --halo-adjoint \
    --inference-batch-size 4 \
    --sht-omega-lmax 3 \
    --output-levels 13,18,24,25,29,30 \
    --initial-time "${INIT_TIME}" \
    "${OUTPUT}"
