# Perlmutter

This page is only for running on NERSC Perlmutter. The generic checkpoint
download and rollout command are in the README.

`perlmutter_scripts/strata_v1_rollout.sh` requests account `m4935`,
constraint `gpu&hbm80g`, and 4 GPUs. It starts the container with
`shifter --image="$IMAGE"`. Put `OUTPUT` on scratch, not in your home
directory.

The sdecadal store is already on CFS:

```bash
export IMAGE=registry.nersc.gov/e3sm/screamcast/strata:1.0
export AUX_DATA_ROOT=/pscratch/sd/$USER/strata_v1
export SCREAM_MAIN_ZARR_PATH=/global/cfs/cdirs/m4935/screamdata/sdecadal.ne1024pg2_ne1024pg2.F20TR-SCREAMv1.c10-sep11.out10min.cubesphere.zarr
export OUTPUT=/pscratch/sd/$USER/strata_v1_forecast
sbatch perlmutter_scripts/strata_v1_rollout.sh
```

`shifterimg pull` that tag once if Shifter says the image is missing.
Members of the `e3sm` project can use this image. Rebuild it only when
the container itself changes.

Stay on one login node for the whole build. The build cache is local to
that node. From the repo root, with `PYTHONPATH` unset:

```bash
podman-hpc login registry.nersc.gov
unset PYTHONPATH
podman-hpc build \
  --build-arg CUDA_ARCH=8.0 \
  -f docker/Dockerfile \
  -t registry.nersc.gov/e3sm/screamcast/strata:1.0 .
podman-hpc push registry.nersc.gov/e3sm/screamcast/strata:1.0
shifterimg -v pull registry.nersc.gov/e3sm/screamcast/strata:1.0
```

`CUDA_ARCH=8.0` is the A100. The Dockerfile default, `9.0`, is the H100.
NATTEN is compiled inside the image for that one architecture, so this
build takes a while. After it finishes, `shifterimg pull` is what makes
the tag visible to a job. The rollout script does not call `podman`; it
only passes `IMAGE` to Shifter.

## Training

Request an interactive GPU node, then run `torchrun` there:

```bash
salloc --nodes=1 --qos=interactive --time=01:00:00 --constraint="gpu&hbm80g" --gpus-per-node=4 --account=m4935
```

For a queued multi-node job, `experiment_manager.py` submits through
`perlmutter_scripts/submit.sh`. Point `envs/perlmutter.env`, or a `.env`
in the repo root, at that script:

```bash
PROJECT_ROOT=/path/to/run/outputs
SUBMIT_SCRIPT=perlmutter_scripts/submit.sh
```

```bash
python3 experiment_manager.py submit --nodes 1 --account m4935 --queue regular \
  --time 47:59:00 --constraint "gpu&hbm80g" --name strata_v1 \
  train.py pixeldit_sem1024d24l_pix128d4l_bilineardwgeluproject_unfreeze_3src_const1em5_t128_4stepft
```
