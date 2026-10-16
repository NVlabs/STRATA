#!/bin/bash
# NOTE: kept as a record of the old Shifter wrapper. eval_global_video.py
# does not load the STRATA v1 checkpoint. Use strata_v1_rollout.sh.
#SBATCH -A m4935
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH -J m4935-screamcast
#SBATCH --dependency singleton
#SBATCH --image=registry.nersc.gov/m4935/screamcast/screamcast:0.2

cd "$PROJECT_ROOT"

export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500
export SCREAM_MAIN_ZARR_PATH=/global/cfs/cdirs/m4935/screamdata/sdecadal.ne1024pg2_ne1024pg2.F20TR-SCREAMv1.c10-sep11.out10min.zarr
export SCREAM_AUX_ZARR_PATH=/global/cfs/cdirs/m4935/screamdata/sdecadal.ne1024pg2_ne1024pg2.F20TR-SCREAMv1.c10-sep11.out10min.zarr
export OMP_NUM_THREADS=8

srun -N $SLURM_NNODES --ntasks-per-node=1 --gpus-per-node=4 --cpu-bind=cores --mpi=pmix \
  shifter --image=registry.nersc.gov/m4935/screamcast/screamcast:0.2 bash -lc "\
    cd $PWD && \
    torchrun \
      --nnodes=$SLURM_NNODES \
      --nproc_per_node=4 \
      --node_rank=\$SLURM_NODEID \
      --rdzv_backend=c10d \
      --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
      eval_global_video.py \
        --prediction-faces -1 \
        --nsteps 6 \
        --batch-size 64 \
        --experiment-name global_rollout_test1 \
        --rollout-start-step 1728 \
        --variables-prognostic PotentialTemperature,U,V,z_mid,omega,qv,ps,T_2m \
        --variables-forcing coszr,phis,sgh30,landfrac \
        --variables-diagnostic precip_ice_surf_mass_flux,precip_liq_surf_mass_flux \
        --variables-prognostic-state omega \
        --level-start 55 --level-end 128 --plevel 4"