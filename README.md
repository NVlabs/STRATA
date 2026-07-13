# Strata

Strata is a deep-learning weather emulator for the
[SCREAM](https://e3sm.org/scream/) global atmosphere model. It trains
transformer-based neural networks to emulate SCREAM atmospheric physics on
the cubed-sphere grid and supports multi-day global forecasting.

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
alone: a CUDA-tuned PyTorch build, NVIDIA DALI (data pipelines), Transformer
Engine, and a NATTEN build that matches the installed torch. The supported
base environment is the [NGC PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
`nvcr.io/nvidia/pytorch:26.01-py3` (anonymous pulls generally work; a free
NGC account may be required depending on NGC policy).

### Docker (recommended)

Build the training/inference image from the repository root:

    docker build -f docker/Dockerfile -t strata .

The image contains the complete environment, including a from-source NATTEN
build (set `CUDA_ARCH` in [`docker/build_natten.sh`](docker/build_natten.sh)
for non-Hopper GPUs) and PhysicsNeMo pinned to a commit that includes the
Strata models.

### Manual install

Inside an NGC PyTorch `26.01` container (torch, DALI, and Transformer Engine
ship with it):

    make install

Besides `requirements.txt`, this installs three packages that need special
handling — NATTEN from the per-torch wheel index at `https://whl.natten.org`
(the wheel must match the installed torch; NGC images need the source build
in `docker/build_natten.sh` instead), earth2grid from a pinned GitHub archive
(it is not on PyPI), and PhysicsNeMo from a pinned GitHub archive with
`--no-deps` (its `torch>=2.10.0` floor would otherwise replace a container's
pre-release torch build). See the comments in
[`docker/Dockerfile`](docker/Dockerfile) for the rationale behind each step.

### Data, checkpoints, and auxiliary files

Create a `.env` file at the repository root that points to your data and output
locations. Copy the template and fill in the values:

    cp envs/example.env .env

The key variables are `PROJECT_ROOT` (training and rollout outputs are written
here), `ZARR_ROOT` (the SCREAM zarr dataset), `AUX_DATA_ROOT` (auxiliary files,
below), and `WANDB_API_KEY` (experiment logging).

Place the SCREAM zarr dataset, a model checkpoint, and the auxiliary files under
the locations configured in `.env`. The auxiliary files used for training and
cubed-sphere inference are `latlon_ne1024pg2.nc`, `ne1024pg2_scrip.nc`,
`ne1024halo256pg2_scrip.nc`, `scream_vertical_coordinate.nc`, and (optionally,
for the rollout qv-fixer) `ps_mean_cubesphere_day14_r2.nc`.

> **Availability**: the SCREAM zarr dataset, trained model checkpoints, and
> the auxiliary files other than the shipped
> `data/scream_vertical_coordinate.nc` are not yet publicly distributed, so
> the training and rollout workflows below cannot currently run end to end
> outside NVIDIA. The environment build and the unit test suite (`pytest`)
> work without them.

## Training

Training settings are named Python configs in `train_configs.py`; see
`screamcast/config.py` for the full config reference. Add an experiment
(optionally branching off an existing one with `dataclasses.replace()`), then
launch it directly in an interactive GPU session.

Single GPU:

    python train.py <config_name>

Multiple GPUs on one node:

    torchrun --nproc_per_node=<num_gpus> train.py <config_name>

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

    make lint     # SPDX license headers, black, and ruff
    pytest

## Disclaimer

This project will download and install additional third-party open-source
software. Review the license terms of those projects before use.
