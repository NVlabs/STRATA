# Strata

Strata is a deep-learning weather emulator for the
[SCREAM](https://e3sm.org/scream/) global atmosphere model. It trains
transformer-based neural networks (DiT3D / PixelDiT) to emulate SCREAM
atmospheric physics on the cubed-sphere grid and supports multi-day global
forecasting.

The public release is named *Strata*, but the Python package, command-line
scripts, and environment variables keep the project's original name: you import
`screamcast`, run the `screamcast`-prefixed entry points, and configure
`SCREAM_*` variables. *Strata* and *screamcast* refer to the same project.

> This code is provided for research and development purposes only.

## Setup

Install the dependencies that are not already in the base PyTorch image:

    make install

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
