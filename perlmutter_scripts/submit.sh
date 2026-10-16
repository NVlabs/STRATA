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
#SBATCH -A m4935
#SBATCH -C gpu&hbm80g
#SBATCH -q regular
#SBATCH -t 47:59:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH -J m4935-screamcast
#SBATCH --dependency singleton
#SBATCH --image=registry.nersc.gov/m4935/screamcast/screamcast:0.2

export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
workdir=$PWD

# set local dataset paths
export SCREAM_MAIN_ZARR_PATH=/global/cfs/cdirs/m4935/screamdata/sdecadal.ne1024pg2_ne1024pg2.F20TR-SCREAMv1.c10-sep11.out10min.zarr
export SCREAM_AUX_ZARR_PATH=/global/cfs/cdirs/m4935/screamdata/sdecadal.ne1024pg2_ne1024pg2.F20TR-SCREAMv1.c10-sep11.out10min.zarr
export SPLIT_BY_TIME=True
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export CUDNN_V8_API_ENABLED=1
export OMP_NUM_THREADS=8
unset TORCH_DISTRIBUTED_DEBUG

export TORCH_DISTRIBUTED_DEBUG=DETAIL

srun -n $SLURM_NTASKS -u  \
     --mpi=pmix   \
		 --cpu-bind=ldoms \
    shifter bash -c "cd $workdir && source perlmutter_scripts/export_DDP_vars.sh && ${RUN_CMD}"
