# Strata

Strata is a deep-learning weather emulator for the
[SCREAM](https://e3sm.org/scream/) global atmosphere model. It trains
transformer-based neural networks to emulate SCREAM atmospheric physics on
the cubed-sphere grid and supports multi-day global forecasting.

- 📄 **Paper:** [Scaling Storm-Resolving Atmospheric AI Simulation to the Entire Planet](https://arxiv.org/html/2606.31248v1) — Accepted at **NeurIPS 2026, Main Track**.
- 🤗 **Model checkpoint:** Will be shared soon.

The Strata architecture (a two-stage 3D neighborhood-attention transformer
with stereographic rotary position embeddings) lives in
[NVIDIA PhysicsNeMo](https://github.com/NVIDIA/physicsnemo) as
`physicsnemo.experimental.models.strata`. This repository adds the
SCREAM-specific pieces around it: spherical tile geometry, tile-local wind
rotation, data pipelines, training configs, and rollout/inference tooling
(see `screamcast/strata_wrappers.py`).

The public release is named *Strata*, but the Python package, command-line
scripts, and environment variables keep the project's original name: you import
`screamcast`, run the `screamcast`-prefixed entry points, and configure
`SCREAM_*` variables. *Strata* and *screamcast* refer to the same project.

> This code is provided for research and development purposes only.

## Setup

The models need a recent NVIDIA GPU software stack that cannot come from PyPI
alone: a CUDA-tuned PyTorch build, NVIDIA DALI (data pipelines), and a NATTEN
build that matches the installed torch. The supported
base environment is the [NGC PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
`nvcr.io/nvidia/pytorch:26.09-py3` (anonymous pulls generally work; a free
NGC account may be required depending on NGC policy).

### Docker (recommended)

Build the training/inference image from the repository root:

    docker build -f docker/Dockerfile -t strata .

The image contains the complete environment, including a from-source NATTEN
build (set `CUDA_ARCH` in [`docker/build_natten.sh`](docker/build_natten.sh)
for non-Hopper GPUs) and PhysicsNeMo pinned to a commit that includes the
Strata models.

### pip install (plain PyPI environment)

Outside NGC containers the package also installs as a normal Python project.
The one extra flag is NATTEN's wheel index, which lets pip pick the CUDA
wheel matching your torch build (without it, pip falls back to the PyPI
sdist and a long from-source compile):

    pip install -e . -f https://whl.natten.org

Optional extras: `.[s3]` (S3 data access), `.[analysis]` (plotting),
`.[cu12]`/`.[cu13]` (DALI for your CUDA major version), `.[dev]` (tests),
`.[notebooks]` (marimo). earth2grid installs automatically from its pinned
GitHub archive (it has no PyPI release). Note this path resolves the newest
compatible versions from PyPI rather than reproducing the container
environment above — use the Docker image for the verified stack.

### Data, checkpoints, and auxiliary files

Create a `.env` file at the repository root that points to your data and output
locations. Copy the template and fill in the values:

    cp envs/example.env .env

The key variables are `PROJECT_ROOT` (training and rollout outputs are written
here), `ZARR_ROOT` (the SCREAM zarr dataset), `AUX_DATA_ROOT` (auxiliary files,
below), and `WANDB_API_KEY` (experiment logging).

The SCREAM training dataset (cubed-sphere zarr v3 stores) is published on
Hugging Face as three dataset repositories — a 14-day simulation
(*sdecadal*) plus two 7-day simulations (*sdy1*, *sdy2*) used as the
additional sources in the three-source training configs:

- [nvidia/STRATA-SCREAM-sdecadal](https://huggingface.co/datasets/nvidia/STRATA-SCREAM-sdecadal) — 14 days, 10-minute steps
- [nvidia/STRATA-SCREAM-sdy1](https://huggingface.co/datasets/nvidia/STRATA-SCREAM-sdy1) — 7 days, 10-minute steps
- [nvidia/STRATA-SCREAM-sdy2](https://huggingface.co/datasets/nvidia/STRATA-SCREAM-sdy2) — 7 days, 10-minute steps

Each repository is the zarr store itself, so it can be read directly over
HTTP via `huggingface_hub`'s fsspec filesystem (installed in the Docker
image) — convenient for exploration without downloading ~TB of data:

```python
import xarray as xr

ds = xr.open_zarr("hf://datasets/nvidia/STRATA-SCREAM-sdy1")
ds["T_2m"].isel(time=500, ncol=slice(0, 100)).values  # streams just the chunks it needs
```

For training, download the stores instead (e.g. with `huggingface-cli
download`) — the dataloader reads tiles at a rate that HTTP streaming cannot
sustain — and place each under `ZARR_ROOT` with the directory name the
training configs expect (see `_ZARR_SDECADAL` / `_ZARR_SDY1` / `_ZARR_SDY2`
in `train_configs.py`), e.g.
`$ZARR_ROOT/sdecadal.ne1024pg2_ne1024pg2.F20TR-SCREAMv1.c10-sep11.out10min.cubesphere.zarr`.
(Setting `ZARR_ROOT` itself to an `hf://` prefix does not work: the configs
join it with the original store names, which differ from the Hugging Face
repository names.)

The auxiliary files used for training and cubed-sphere inference are
`latlon_ne1024pg2.nc`, `ne1024pg2_scrip.nc`, `ne1024halo256pg2_scrip.nc`,
and `scream_vertical_coordinate.nc`; place them under `AUX_DATA_ROOT`.
`scream_vertical_coordinate.nc` ships in `data/`; the other three are
generated from scratch (pure numpy + netCDF4, ~15 minutes) by

    bash data_prep/scrip_generation/generate_all.sh --output-dir <AUX_DATA_ROOT>

(see [`data_prep/scrip_generation/`](data_prep/scrip_generation/); the
derived `latlon_ne1024pg2.nc` is bit-identical to the file the shipped
configs were trained with).

> **Availability**: the STRATA v1 checkpoint and its auxiliary files
> (static fields, `latlon_ne1024pg2.nc`, and both SCRIP grids) are in the
> Hugging Face dataset `qx7m-kestrel/strata-v1-bundle`. That repository is
> private for now and will be replaced by a public Hugging Face repository
> soon; this path will be updated then. Download it and set `AUX_DATA_ROOT`
> to that directory. A `.env` file is not required if you export
> `AUX_DATA_ROOT` and `SCREAM_MAIN_ZARR_PATH`.

## Running the STRATA v1 checkpoint

The checkpoint bundle is small:

```bash
hf download qx7m-kestrel/strata-v1-bundle --local-dir /path/to/strata_v1
export AUX_DATA_ROOT=/path/to/strata_v1
```

The initial-condition store is not. `nvidia/STRATA-SCREAM-sdecadal` is
about 35 TB. Read it from remote storage first: set `SCREAM_MAIN_ZARR_PATH`
to an `s3://` path (with `SCREAM_ZARR_PROFILE` set to that rclone remote)
or to the Hugging Face dataset, and run the forecast against that path.
Download a local copy only if throughput is the bottleneck:

```bash
hf download nvidia/STRATA-SCREAM-sdecadal --repo-type dataset \
  --local-dir /path/to/sdecadal.ne1024pg2_ne1024pg2.F20TR-SCREAMv1.c10-sep11.out10min.cubesphere.zarr

export SCREAM_MAIN_ZARR_PATH=/path/to/sdecadal.ne1024pg2_ne1024pg2.F20TR-SCREAMv1.c10-sep11.out10min.cubesphere.zarr
```

The forecast itself is a 4-GPU, 2-step `torchrun`, on whatever machine can
launch it. Build `docker/Dockerfile` for your GPU (`--build-arg CUDA_ARCH=8.0`
on A100, `9.0` on H100). `TORCH_COMPILE_DISABLE=1` keeps the first step from
sitting in Inductor. PhysicsNeMo is installed with `--no-deps` in that image,
so `treelib`, `gitpython`, `hydra-core`, and `termcolor` have to be
importable or the process exits at startup.

```bash
export SCREAM_AUX_ZARR_PATH=$SCREAM_MAIN_ZARR_PATH
export TORCH_COMPILE_DISABLE=1
export OUTPUT=/path/to/forecast_out
torchrun --nproc_per_node=4 scripts/ace/run_screamcast_nudged.py \
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
  $OUTPUT
```

A finished run exits 0 and logs `run_scream step 2/2`,
`Forecast done: 2 steps = 0.3 h`, and liquid precip about `0.002 mm/day`.
Ice precip `0.000` is a known bug of this checkpoint, not normal behavior:
ice precip is a diagnostic, and training applied a ReLU non-negative
constraint (`do_precip_relu`), so ice precip was not learned well. It
should not affect the other state variables.

On Perlmutter, do not hand-copy the launch. `perlmutter_scripts/strata_v1_rollout.sh`
is this same forecast wrapped for Shifter, and it hardcodes the NERSC
account and GPU constraint. The exports and `sbatch` line are in
[`docs/nersc_setup.md`](docs/nersc_setup.md).

## Training

Training settings are named Python configs in `train_configs.py`; see
`screamcast/config.py` for the full config reference. Add an experiment
(optionally branching off an existing one with `dataclasses.replace()`), then
launch it directly in an interactive GPU session.

Single GPU:

    python train.py <config_name>

The released STRATA v1 checkpoint is the 4-step finetune of the Pixel DiT
with the bilinear depthwise-GeLU projection:
`pixeldit_sem1024d24l_pix128d4l_bilineardwgeluproject_unfreeze_3src_const1em5_t128_4stepft`.
Run `torchrun` inside an interactive GPU session, not from a login node.
That config also needs `sdy1` and `sdy2` under `ZARR_ROOT`, under the
directory names `_ZARR_SDY1` and `_ZARR_SDY2` in `train_configs.py`
(`sdy1.ne1024pg2_...out10min-4lev.cubesphere.zarr` and the matching `sdy2`
name), not the Hugging Face repository names. A forecast does not.

Multiple GPUs on one node:

    torchrun --nproc_per_node=<num_gpus> train.py pixeldit_sem1024d24l_pix128d4l_bilineardwgeluproject_unfreeze_3src_const1em5_t128_4stepft

Checkpoints (`best.pth` / `latest.pth`) and logs are written to the config's
`rundir` (default `output/`). `train.py` uses PyTorch Lightning Fabric and
auto-detects the launch environment, so the same entry point also runs under
SLURM `srun` (via `SLURM_*` env vars) for multi-node jobs.

## Global rollout (cubed sphere)

Global rollouts run `scripts/ace/run_screamcast_nudged.py` under SLURM.
[`slurm/submit_inference.sh`](slurm/submit_inference.sh) is a public example
launcher — edit the checkpoint/output paths at the top and submit it with
`sbatch`. Run `python3 scripts/ace/run_screamcast_nudged.py --help` for the full
CLI (checkpoint, number of steps, tile/halo size, omega filtering, output
levels, initial time, ...). `--output-levels` selects which vertical levels are
written to the output zarr.

## Local evaluation

For a quick rollout on a single tile, open
[`notebooks/run_screamcast.py`](notebooks/run_screamcast.py) as a
[marimo](https://marimo.io) notebook (`marimo edit notebooks/run_screamcast.py`).
It loads a checkpoint via `ScreamcastModel` and rolls out on one tile so you can
inspect predictions interactively.

## ACE → SCREAM finetuning

The optional ACE→SCREAM forecast-residual workflow is documented in
[`docs/ace2scream_finetuning.md`](docs/ace2scream_finetuning.md); its scripts
live under [`scripts/ace/`](scripts/ace/).

## Developing

GitHub Actions runs `make lint` on pushes to `main` and pull requests. To run the same
checks locally, install Python 3.12 and run `make install-linters` first.

    make lint     # SPDX license headers, black, and ruff
    pytest

## Disclaimer

This project will download and install additional third-party open-source
software. Review the license terms of those projects before use.
