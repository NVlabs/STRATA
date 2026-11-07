# Strata

[![Paper](https://img.shields.io/badge/Paper-NeurIPS%202026-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2606.31248)
[![Project page](https://img.shields.io/badge/Project-page-76b900?logo=nvidia&logoColor=white)](https://research.nvidia.com/labs/climate/strata/)
[![Model](https://img.shields.io/badge/Model-nvidia%2Fstrata-ffd21e?logo=huggingface)](https://huggingface.co/nvidia/strata)
[![Data](https://img.shields.io/badge/Data-SCREAM%20Zarr-ffd21e?logo=huggingface)](#data)
[![License](https://img.shields.io/badge/License-Apache%202.0%20%7C%20OpenMDW%20%7C%20CC%20BY%204.0-lightgrey)](#license)

Code for *Scaling Storm-Resolving Atmospheric AI Simulation to the Entire
Planet* (NeurIPS 2026, Main Track).

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
alone: a CUDA-tuned PyTorch build, NVIDIA DALI (data pipelines), and a NATTEN
build that matches the installed torch. The supported base environment is the
[NGC PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
`nvcr.io/nvidia/pytorch:26.09-py3`.

### Docker (recommended)

Build the training/inference image from the repository root:

    docker build -f docker/Dockerfile -t strata .

The image contains the complete environment, including NATTEN built from
source for Ada and Hopper GPUs (pass `--build-arg CUDA_ARCH=8.0` for A100; see
[`docker/build_natten.sh`](docker/build_natten.sh)) and PhysicsNeMo pinned to a
commit that includes the Strata models. It does not contain this repository;
mount it at run time.

### pip install (plain PyPI environment)

Outside NGC containers the package also installs as a normal Python project.
NATTEN's wheel index lets pip pick the CUDA wheel matching your torch build:

    pip install -e . -f https://whl.natten.org

Optional extras: `.[s3]` (S3 data access), `.[analysis]` (plotting),
`.[cu12]`/`.[cu13]` (DALI for your CUDA major version), `.[dev]` (tests),
`.[notebooks]` (marimo). This path installs the newest compatible versions
from PyPI; use the Docker image for the tested stack.

### Environment variables

The code reads its paths from environment variables. Export them, or put them
in a `.env` file at the repository root (`cp envs/example.env .env`), which is
loaded when present. Which ones you need depends on the task:

| Task | Variables |
|---|---|
| Forecast with the released checkpoint | `AUX_DATA_ROOT` (auxiliary files, below) and `SCREAM_MAIN_ZARR_PATH` (the sdecadal store, for initial conditions) |
| Training | also `PROJECT_ROOT` (run directories, and the checkpoints that chained configs resume from), `ZARR_ROOT` (directory or `s3://` prefix holding the stores), and `SCREAM_MAIN_ZARR_PATH` and `SCREAM_AUX_ZARR_PATH`, both set to the local sdecadal store, which the validation loader reads |
| Weights & Biases logging | `WANDB_API_KEY`, optionally `WANDB_PROJECT` and `WANDB_ENTITY`; without the key, training skips wandb |
| `s3://` store paths | `SCREAM_ZARR_PROFILE`, the name of a remote in `~/.config/rclone/rclone.conf` that holds the S3 endpoint and keys |

### Data

The raw SCREAM output used for training and evaluation is at
<https://portal.nersc.gov/archive/home/n/ndk/www/STRATA2026> (CC BY 4.0, see
[License](#license)). A training-ready version in Zarr format is on Hugging
Face, one repository per simulation:

| Repository | Simulation | Time steps (10 min) | Size |
|---|---|---|---|
| [nvidia/STRATA-SCREAM-sdecadal](https://huggingface.co/datasets/nvidia/STRATA-SCREAM-sdecadal) | sdecadal, 14 days (transient historical, F20TR) | 2017 | 36.8 TB |
| [nvidia/STRATA-SCREAM-sdy1](https://huggingface.co/datasets/nvidia/STRATA-SCREAM-sdy1) | DYAMOND1, 7 days (F2010) | 1009 | 19.6 TB |
| [nvidia/STRATA-SCREAM-sdy2](https://huggingface.co/datasets/nvidia/STRATA-SCREAM-sdy2) | DYAMOND2, 7 days (F2010) | 1009 | 19.7 TB |

Each repository is a single Zarr v3 store on the ne1024pg2 grid: arrays are
`(time, lev, ncol)` with 32 model levels. The stores have no coordinate
arrays; the code attaches them. A store can be read directly over HTTP, which
is convenient for exploration:

```python
import xarray as xr

ds = xr.open_zarr("hf://datasets/nvidia/STRATA-SCREAM-sdy1")
ds["T_2m"].isel(time=500, ncol=slice(0, 100)).values  # streams just the chunks it needs
```

For training, download the stores (`hf download <repo> --repo-type dataset
--local-dir <dir>`); HTTP streaming is too slow for the dataloader. Place
each store under `ZARR_ROOT` with the directory name the training configs
expect (`_ZARR_SDECADAL` / `_ZARR_SDY1` / `_ZARR_SDY2` in `train_configs.py`):

| Repository | Directory under `ZARR_ROOT` |
|---|---|
| `nvidia/STRATA-SCREAM-sdecadal` | `sdecadal.ne1024pg2_ne1024pg2.F20TR-SCREAMv1.c10-sep11.out10min.cubesphere.zarr` |
| `nvidia/STRATA-SCREAM-sdy1` | `sdy1.ne1024pg2_ne1024pg2.F2010-SCREAMv1.c10-sep11-f602da2b98.out10min-4lev.cubesphere.zarr` |
| `nvidia/STRATA-SCREAM-sdy2` | `sdy2.ne1024pg2_ne1024pg2.F2010-SCREAMv1.c10-sep11-f602da2b98.out10min-4lev.cubesphere.zarr` |

`ZARR_ROOT` cannot be an `hf://` prefix. Download whole stores: a chunk that
is missing from a partial download reads back as zeros, not as an error.

### Auxiliary files (`AUX_DATA_ROOT`)

Inference and training look up auxiliary files by name in one directory,
`AUX_DATA_ROOT`. Some come from the model repository; generate the rest with
[`data_prep/scrip_generation/`](data_prep/scrip_generation/):

| File | Source | Used for |
|---|---|---|
| `checkpoints/strata_v1.pth` | [nvidia/strata](https://huggingface.co/nvidia/strata) | weights, normalization statistics, training config |
| `phis.npy`, `sgh30.npy`, `landfrac.npy` | [nvidia/strata](https://huggingface.co/nvidia/strata) | static forcing fields (inference) |
| `latlon_ne1024pg2.nc` | generated by `generate_all.sh` | model lat/lon conditioning (inference and training) |
| `ne1024pg2_scrip.nc` (2.3 GB) | generated by `generate_all.sh` | cell areas for regional averages, regridding |
| `ne1024halo256pg2_scrip.nc` (5.2 GB) | generated by `generate_all.sh` | halo exchange between cubed-sphere tiles |
| `scream_vertical_coordinate.nc` | copied from `data/` in this repository | hybrid level coefficients |

From the repository root:

```bash
hf download nvidia/strata --local-dir /path/to/strata_v1
export AUX_DATA_ROOT=/path/to/strata_v1
bash data_prep/scrip_generation/generate_all.sh --output-dir $AUX_DATA_ROOT
cp data/scream_vertical_coordinate.nc $AUX_DATA_ROOT/
```

`generate_all.sh` takes about 20 minutes on a 64-core node and leaves
intermediate meshes (`*.g`, about 7 GB) that can be deleted.

## Running the STRATA v1 checkpoint

The forecast reads only its initial condition from the sdecadal store, so the
store can stay remote (`SCREAM_MAIN_ZARR_PATH=hf://datasets/nvidia/STRATA-SCREAM-sdecadal`).
To read it locally instead, download it (36.8 TB) and point
`SCREAM_MAIN_ZARR_PATH` at the copy:

```bash
hf download nvidia/STRATA-SCREAM-sdecadal --repo-type dataset \
  --local-dir $ZARR_ROOT/sdecadal.ne1024pg2_ne1024pg2.F20TR-SCREAMv1.c10-sep11.out10min.cubesphere.zarr
export SCREAM_MAIN_ZARR_PATH=$ZARR_ROOT/sdecadal.ne1024pg2_ne1024pg2.F20TR-SCREAMv1.c10-sep11.out10min.cubesphere.zarr
```

The code labels sdecadal time index 0 as `2020-10-01T00:00`, so valid initial
times run from `2020-10-01T00:00` to `2020-10-15T00:00` in 10-minute steps.
The simulation used October 2020 sea-surface temperatures but started from a
1994 atmospheric initial condition, so its weather does not follow October 2020
observations: evaluate forecasts against the SCREAM output, not against
observations or reanalysis. The model was trained on 2020-10-02 to 2020-10-10
of this store (plus sdy1 and sdy2), so evaluate from 2020-10-11 onwards.
Initial conditions come from sdecadal only.

### Forecast

The forecast runs under `torchrun` with one process per GPU; the commands
below use 4 × H100 80 GB. Run them from the repository root in an environment
from [Setup](#setup). The Docker image does not contain this repository, so
mount it, `AUX_DATA_ROOT` and the output directory into the container; on a
cluster, use its own container setup for interactive or batch jobs (for
example Pyxis/Enroot, Apptainer or Shifter):

```bash
export PYTHONPATH=$PWD   # the script imports data_catalog, which is not in the installed package
export AUX_DATA_ROOT=/path/to/strata_v1
export SCREAM_MAIN_ZARR_PATH=hf://datasets/nvidia/STRATA-SCREAM-sdecadal
export OUTPUT=/path/to/output/strata_v1_24h.zarr
torchrun --nproc_per_node=4 scripts/ace/run_screamcast_nudged.py \
  --run-name strata_v1_24h \
  --checkpoint $AUX_DATA_ROOT/checkpoints/strata_v1.pth \
  --forecast-model screamcast \
  --correction none \
  --n-steps 144 \
  --tile-size 128 \
  --halo-width 16 \
  --halo-adjoint \
  --no-halo-adjoint-cross-face \
  --tile-jitter 4 \
  --tile-jitter-mode cycle16 \
  --sht-omega-lmax 3 \
  --inference-batch-size 1 \
  --output-variables T_2m ps U V PotentialTemperature qv omega precip_liq_surf_mass_flux \
  --output-levels 13,18,24,25,29,30 \
  --initial-time 2020-10-13T00:00:00 \
  $OUTPUT
```

These are the recommended settings for this checkpoint: tile jitter on (4 px,
`cycle16`), no halo adjoint across cube-face edges, and omega filtered with
SHT lmax 3. For a quick check, run `--n-steps 2` first; a finished run logs
`Forecast done: 2 steps`. Loading the model and the initial condition takes a
few minutes, then each 10-minute step takes about 1.5–2.5 minutes on
4 × H100. The network is compiled with `torch.compile`, which is about a third
faster than eager after the first step (`TORCH_COMPILE_DISABLE=1` runs it
eagerly); Dynamo's `recompile_limit` warning during the first step is
harmless.

On 4 GPUs keep `--inference-batch-size 1`: batch 4 runs out of memory on
4 × 80 GB. Each written step holds every selected channel on the full grid
(about 100 MB per channel before compression), so choose `--output-variables`
and `--output-levels` deliberately.

Each step logs the global-mean liquid and ice precipitation in mm/day. Ice
precipitation reads `0.000` with this checkpoint (a known limitation); it is a
diagnostic output and does not feed back into the forecast.

### Forecast output

The forecast is written to the Zarr store `$OUTPUT`, and area-weighted
regional means (global, land, ocean, tropics, extratropics) of every output
channel to `${OUTPUT}_regional_averages.nc` (`--no-zarr-output` writes only
the latter). Arrays in the store are `(time, step, face, x, y)` for 2-D fields
and `(time, step, level, face, x, y)` for 3-D fields, on the six
2048 × 2048 cube faces:

- `time`: initial time; `step`: lead time in seconds (`600` is the first
  10-minute step);
- `level`: SCREAM model-level index (8–31, from the top), not pressure;
  `hyam` / `hybm` give the hybrid coefficients, and `--output-levels`
  selects levels by these indices;
- `lat`, `lon`: `(face, x, y)` cell centres in degrees.

Variable units are SCREAM's: precipitation fluxes are in m/s.
`--output-steps N` writes every N-th step, at leads N, 2N, … × 10 minutes
(`--output-steps 6` gives hourly output); `--n-steps` must be a multiple of N.

### Running under SLURM

[`slurm/submit_inference.sh`](slurm/submit_inference.sh) is an example
`sbatch` launcher for the same forecast on 2 nodes × 8 GPUs, where
`--inference-batch-size 4` fits: edit the checkpoint and output paths at the
top, and submit it from the repository root (it sets `PYTHONPATH` from
there). The path variables can be exported before `sbatch` or come from
`.env`. On Perlmutter, `perlmutter_scripts/strata_v1_rollout.sh` runs a 2-step
check with the same model settings (eager, regional means only; setup in
[`docs/nersc_setup.md`](docs/nersc_setup.md)); for a full forecast, raise
`--n-steps` and drop `--no-zarr-output`.
`python3 scripts/ace/run_screamcast_nudged.py --help` lists all options.

## Training

Training settings are named Python configs in `train_configs.py`; see
`screamcast/config.py` for the full config reference. Add an experiment
(optionally branching off an existing one with `dataclasses.replace()`), then
launch it on a GPU node, not a login node:

    python train.py <config_name>                                  # single GPU
    torchrun --nproc_per_node=<num_gpus> train.py <config_name>    # one node, several GPUs

For multi-node jobs, run the same entry point under SLURM `srun`. For queued
SLURM jobs, `experiment_manager.py submit --name <run> train.py <config_name>`
copies the code to `$PROJECT_ROOT/<run>` and submits it with `sbatch` through
the script named by `SUBMIT_SCRIPT`; see
[`docs/nersc_setup.md`](docs/nersc_setup.md) for a Perlmutter example.
Checkpoints (`best.pth` / `latest.pth`) and logs are written to the config's
`rundir` (default `output/`).

The released checkpoint was trained with the config
`pixeldit_sem1024d24l_pix128d4l_bilineardwgeluproject_unfreeze_3src_const1em5_t128_4stepft`,
which reads all three stores under the directory names in the table above. A
recommended training recipe will be added in a future update.

To fine-tune from the released `strata_v1.pth` (via `experiment.resume_from`),
turn off `torch.compile` (`compute.do_torch_compile=False`) for now: with
compilation on, its weights are not loaded, and the log does not flag this.

## Local evaluation

For a quick rollout on a single tile, open
[`notebooks/run_screamcast.py`](notebooks/run_screamcast.py) as a
[marimo](https://marimo.io) notebook (`marimo edit notebooks/run_screamcast.py`).
It loads a checkpoint via `ScreamcastModel` and rolls out on one tile so you can
inspect predictions interactively. Set `checkpoint_path` in the notebook to
`$AUX_DATA_ROOT/checkpoints/strata_v1.pth`, and set the same environment
variables as for the forecast above.

## ACE → SCREAM finetuning

The optional ACE→SCREAM forecast-residual workflow is documented in
[`docs/ace2scream_finetuning.md`](docs/ace2scream_finetuning.md); its scripts
live under [`scripts/ace/`](scripts/ace/).

## Developing

GitHub Actions runs `make lint` on pushes to `main` and pull requests. To run the same
checks locally, install Python 3.12 and run `make install-linters` first.

    make lint     # SPDX license headers, black, and ruff
    pytest

## License

- Code in this repository: Apache License 2.0 ([`LICENSE.txt`](LICENSE.txt)).
- Model weights ([nvidia/strata](https://huggingface.co/nvidia/strata)):
  [OpenMDW License Agreement 1.1](https://github.com/OpenMDW/OpenMDW/blob/main/1.1/LICENSE.OpenMDW-1.1);
  its static fields `phis.npy`, `sgh30.npy` and `landfrac.npy` are derived from
  the SCREAM data and carry CC BY 4.0 attribution (see `ATTRIBUTION.md` there).
- SCREAM data (raw output and the `nvidia/STRATA-SCREAM-*` Zarr stores):
  CC BY 4.0, as E3SM model output (see the
  [E3SM data documentation](https://docs.e3sm.org/e3sm_data_docs/_build/html/index.html)).

## Disclaimer

This project will download and install additional third-party open-source
software. Review the license terms of those projects before use.
