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
# STRATA v1 forecast. One Slurm task per node; torchrun starts one process per GPU.
# Export IMAGE, AUX_DATA_ROOT, SCREAM_MAIN_ZARR_PATH, and OUTPUT before sbatch.
# Build docker/Dockerfile with CUDA_ARCH=8.0 on A100 (9.0 on H100) and pass that image as IMAGE.
#SBATCH -A m4935
#SBATCH -C gpu&hbm80g
#SBATCH -q interactive
#SBATCH -t 01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH -J strata-v1

set -euo pipefail

: "${IMAGE:?set IMAGE to your container tag}"
: "${AUX_DATA_ROOT:?set AUX_DATA_ROOT to the local strata bundle}"
: "${SCREAM_MAIN_ZARR_PATH:?set SCREAM_MAIN_ZARR_PATH to the local cubed-sphere zarr}"
OUTPUT=${OUTPUT:-$PWD/strata_v1_forecast}

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=29500
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

srun -N 1 --ntasks-per-node=1 --gpus-per-node=4 --cpu-bind=cores --mpi=pmix \
  shifter --image="$IMAGE" bash -lc "\
    cd $PWD && \
    export AUX_DATA_ROOT=$AUX_DATA_ROOT && \
    export SCREAM_MAIN_ZARR_PATH=$SCREAM_MAIN_ZARR_PATH && \
    export SCREAM_AUX_ZARR_PATH=$SCREAM_MAIN_ZARR_PATH && \
    export MASTER_ADDR=$MASTER_ADDR && \
    export MASTER_PORT=$MASTER_PORT && \
    export PYTHONPATH=$PWD && \
    export PYTHONUNBUFFERED=1 && \
    export TORCH_COMPILE_DISABLE=1 && \
    torchrun \
      --nnodes=1 \
      --nproc_per_node=4 \
      --node_rank=\$SLURM_NODEID \
      --rdzv_backend=c10d \
      --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
      scripts/ace/run_screamcast_nudged.py \
        --run-name strata_v1_smoke \
        --checkpoint $AUX_DATA_ROOT/checkpoints/strata_v1.pth \
        --forecast-model screamcast \
        --correction none \
        --n-steps 2 \
        --tile-size 128 \
        --halo-width 16 \
        --halo-adjoint \
        --inference-batch-size 1 \
        --tile-jitter 4 \
        --tile-jitter-mode cycle16 \
        --tile-jitter-fill replicate \
        --sht-omega-lmax 0 \
        --no-zarr-output \
        --initial-time 2020-10-13T00:00:00 \
        $OUTPUT"
