# STRATA

[![Paper](https://img.shields.io/badge/Paper-NeurIPS%202026-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2606.31248)
[![Project page](https://img.shields.io/badge/Project-page-76b900?logo=nvidia&logoColor=white)](https://research.nvidia.com/labs/climate/strata/)
[![Model](https://img.shields.io/badge/Model-nvidia%2Fstrata-ffd21e?logo=huggingface)](https://huggingface.co/nvidia/strata)
[![Data](https://img.shields.io/badge/Data-SCREAM%20Zarr-ffd21e?logo=huggingface)](#data)
[![License](https://img.shields.io/badge/License-Apache%202.0%20%7C%20OpenMDW%20%7C%20CC%20BY%204.0-lightgrey)](#license)

Code for *Scaling Storm-Resolving Atmospheric AI Simulation to the Entire
Planet* (NeurIPS 2026, Main Track).

STRATA is a deep-learning weather emulator for the
[SCREAM](https://e3sm.org/scream/) global atmosphere model. It trains
transformer-based neural networks to emulate SCREAM atmospheric physics on
the cubed-sphere grid and supports multi-day global forecasting.

The STRATA architecture (a two-stage 3D neighborhood-attention transformer
with stereographic rotary position embeddings) lives in
[NVIDIA PhysicsNeMo](https://github.com/NVIDIA/physicsnemo) as
`physicsnemo.experimental.models.strata`. This repository adds the
SCREAM-specific pieces around it: spherical tile geometry, tile-local wind
rotation, data pipelines, training configs, rollout/inference tooling
(see `screamcast/strata_wrappers.py`), and a pipeline that builds SCREAM
initial conditions from ERA5 or the ECMWF HRES analysis
(see [Forecasts from ERA5 or ECMWF HRES](#forecasts-from-era5-or-ecmwf-hres)).

The public release is named *STRATA*, but the Python package, command-line
scripts, and environment variables keep the project's original name: you import
`screamcast`, run the `screamcast`-prefixed entry points, and configure
`SCREAM_*` variables. *STRATA* and *screamcast* refer to the same project.

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
commit that includes the STRATA models. It does not contain this repository;
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
the latter; off by default for a regional window). Arrays in the store are `(time, step, face, x, y)` for 2-D fields
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

## Forecasts from ERA5 or ECMWF HRES

STRATA can also forecast real weather. At start-up the run builds a SCREAM initial
condition from ERA5 or the ECMWF HRES analysis; it can run on the globe or on a
regional window; and it can run freely or keep its large scales on a driver forecast
(*replay*). Nothing has to be prepared per case: four choices and one command make a
run.

### One-time setup

The initial condition follows the E3SM [HICCUP](https://github.com/E3SM-Project/HICCUP)
steps (conservative remap, surface pressure adjusted to the SCREAM topography, vertical
remap to L128); `screamcast/initial_conditions/fast_ic.py` is a torch port of them
(HICCUP is BSD-3-licensed; its notice is kept in the file) that matched HICCUP bit for
bit on ten cases apart from last-bit differences, and builds one in about 80 s on one
GPU. It needs, once:

- the SCREAM `phis`, `sgh30` and `landfrac` arrays from `nvidia/STRATA-SCREAM-sdy1`
  (about 60 MB);
- the L128 coordinate file from HICCUP (8 KB);
- the ERA5 to ne1024pg2 remap map (3.2 GB): `build_map.py` builds it in about 30 min on
  72 cores and needs conda-forge `nco` and `tempest-remap`.

[`docs/initial_conditions.md`](docs/initial_conditions.md#inputs) has the commands.

### 1. Date and data source

A JSON spec names the source and the one-time inputs:

```json
{"source": "hres", "init": "2024-09-25T12", "cache": "/path/to/grib-cache",
 "map": "/path/to/map_721x1440_to_ne1024pg2.nc", "phis": "/path/to/scream_phis.zarr",
 "vert": "/path/to/vert_coord_E3SM_L128.nc",
 "latlon": "/path/to/latlon_ne1024pg2.nc", "scream_ref": "/path/to/scream_static"}
```

| `source` | Dates | Initial condition | Driver at +6 h, +12 h, ... |
|---|---|---|---|
| `hres` | 00 and 12 UTC forecasts from 2024-03-05 12 UTC (ECMWF open data) | the HRES analysis at `init` (lead 0 of the forecast) | the same forecast's +6 h, +12 h, ... leads |
| `wb2_hres` | 00 and 12 UTC forecasts, 2016-01-01 to 2023-01-10 12 UTC (WeatherBench2) | the same | the same |
| `era5` | any hour (ARCO ERA5); give `--initial-time` | ERA5 at that hour | ERA5 at those hours |

An HRES run starts at `init` and uses nothing known after it: it is a forecast, the HRES
forecast downscaled by STRATA. An ERA5 driver is reanalysis, so a replay to it is a
hindcast pulled toward the observed weather. `cache` (`hres` only) is where the GRIB
files are kept; if the AWS bucket answers HTTP 503, add
`"opendata_root": "https://storage.googleapis.com/ecmwf-open-data"`.

### 2. Domain: the globe or a window

- **Globe**: all 1,536 tiles. Add `--sht-omega-lmax 3`. A 24 h run takes about 1 to
  1.5 h on two nodes of 8 H100 GPUs (87 min replayed).
- **Window**: `--region-latlon LAT0,LAT1,LON0,LON1` predicts only the smallest window of
  128-px tiles that holds the box with a tile to spare; 24 h take 4 to 8 minutes on one
  node of 8 H100 GPUs (4 x 4 to 8 x 8 tiles, about 2,500 to 5,000 km). The box runs
  east from `LON0` to `LON1`; write a southern box with `=`
  (`--region-latlon=-40,-20,140,160`). The window's edge is a one-tile ring that follows
  the driver's large-scale change, so a window always needs a driver
  (`--driver-onthefly`). `python -m screamcast.regional --latlon-box 8,18,-91,-79
  --world-size 8 --latlon $AUX_DATA_ROOT/latlon_ne1024pg2.nc` prints the window
  (`--region face,i0,i1,j0,j1`) first. Windows can cross cube-face edges and touch cube
  corners.

### 3. Free rollout or replay

- **Free**: after the initial condition STRATA runs on its own (in a window, only the
  ring is driven).
- **Replay** (`--replay`): every 6 hours the scales above about 250 km are pulled to the
  driver and the scales below stay STRATA's. The difference between the driver and the
  run, averaged on 40 km blocks and low-passed on the sphere, is added over the 6 hours
  centred on that time (an incremental analysis update). A 24 h run is replayed at +6,
  +12 and +18 h; run 27 h (`--n-steps 162`) to replay at +24 h too. Tuning:
  `--replay-every-hours` (6), `--replay-ell-half` (160, the scale where the increment is
  halved, about 250 km), `--replay-vars` (default all seven model inputs) and
  `--replay-level-ramp 800,900` (leave the boundary layer free).

| | Globe | Window |
|---|---|---|
| Free | `--ic-onthefly S` | `--ic-onthefly S --driver-onthefly S --region-latlon ...` |
| Replay | `--ic-onthefly S --driver-onthefly S --replay` | `--ic-onthefly S --driver-onthefly S --region-latlon ... --replay` |

`S` is the spec; the driver is usually the same spec as the initial condition. Every run
also takes `--driver-target-dir DIR`, where the initial condition (about 9 GB) and the
driver's 8 x 8-block means are written; a rerun reuses them. So far (2026-11) replayed
windows and a free global run have run this way; global replayed runs have run from
driver stores only.

### 4. Run

A replayed 24 h window over Hurricane Helene from the HRES forecast of 2024-09-25
12 UTC (the spec above as `hres_helene.json`; `--region 3,10,14,7,11` is a 4 x 4-tile
window over the Gulf of Mexico), on one node of 8 GPUs:

```bash
torchrun --nproc_per_node=8 scripts/ace/run_screamcast_nudged.py \
  --run-name helene_replay \
  --checkpoint $AUX_DATA_ROOT/checkpoints/strata_v1.pth \
  --ic-onthefly hres_helene.json \
  --driver-onthefly hres_helene.json --driver-target-dir "$WORK/targets/helene" \
  --region 3,10,14,7,11 --replay \
  --forecast-model screamcast --correction none --n-steps 144 --output-steps 6 \
  --tile-size 128 --halo-width 16 --halo-adjoint --no-halo-adjoint-cross-face \
  --tile-jitter 4 --tile-jitter-mode cycle16 --inference-batch-size 1 \
  --output-variables T_2m ps U V qv omega precip_liq_surf_mass_flux \
  --output-levels 19,24,31 \
  $OUTPUT
```

It takes about 5 minutes, start-up and the initial condition included. For the other
cells of the table, change the flags it names; on the globe add `--sht-omega-lmax 3` and
run on more GPUs (`srun --nodes=2 --ntasks-per-node=8 python ...`). The output is laid
out as in [Forecast output](#forecast-output), with NaN outside a window (whose regional
means are off unless `--regional-averages` is given); the log has one line per analysis
with the size of the increment.

Building full-resolution stores instead (`--ic-store`, `--driver-stores`) is for
sources the run cannot read itself, such as the HRES analyses every 6 h of 2016-2022 in
WeatherBench2's `hres_t0` (`hres_ic.py`); see the docs below. The data keep their own
licences: credit ECMWF or Copernicus as [License](#license) says.

More in `docs/`: [initial_conditions.md](docs/initial_conditions.md) (inputs, the
HICCUP port and its validation), [regional_window.md](docs/regional_window.md)
(choosing a window, the boundary) and [replay.md](docs/replay.md) (the method, the
on-the-fly build, timings, limitations). No forecast skill has been measured yet.

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
which reads all three stores under the directory names in the table above.

### Recommended schedule (production checkpoint)

The 5 km production recipe trains in four stages; each resumes the previous stage's
`best.pth` (`experiment.resume_from`). "Images" counts training tiles (batch size is 1 per
GPU, so one step on 2 nodes / 16 GPUs is 16 images).

We recommend training with the precipitation ReLU off (`pipeline.do_precip_relu=False`)
and clipping precipitation at inference instead: with the ReLU inside the forward pass a
precipitation head can saturate below zero during training (zero output, zero gradient)
and never recover.

1. **Pretrain on 64×64 tiles** at a learning rate of 1e-4 (5k-step warm-up, then
   constant) for about 10M images, or until convergence. The production run reached
   10M images in total (started on 1 node, moved to 2 nodes at the same learning rate);
   starting on 2 nodes should need about the same number of images.
2. **Anneal on 64×64 tiles**: one cosine from 1e-4 down to 1e-6 over the next ~3M images
   (200k steps on 2 nodes), keeping the optimizer state. In `train.py` that is
   `scheduler_type="dampened_cosine_with_hard_restarts"` with `cosine_n_cycles=1`,
   `warmup_steps=0` (otherwise the stage-1 warm-up runs again) and `cosine_t_max=10`
   validations × `validate_steps=20000`; its floor is `min_ratio=0.01` × lr = 1e-6. The LR
   drops to 0 after those 200k steps, so stop there. (`scheduler_type="cosine"` only
   decays to 0.1 × lr = 1e-5.)
3. **Fine-tune on 128×128 tiles** at a constant 1e-5 (2k-step warm-up) with a fresh
   optimizer. Better: fine-tune directly on tile + 2×halo, the padded tile the rollout
   feeds the model. This converges within about 0.5-1M images.
4. **Multi-step fine-tuning** once stage 3 has converged: 2-step, then 4-step, both at a
   constant 1e-5. `add_multistep_finetune_configs("<stage-3 config>")` in
   `train_configs.py` defines `<stage-3 config>_2stepft` and `_4stepft`; run the 2-step
   config, then the 4-step one (it resumes the 2-step run's `best.pth`).

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
- Analysis and forecast data the code downloads when it runs are not part of this
  repository and keep their own licences. Credit them in work that uses them,
  including initial conditions, replay targets, forecasts and figures made from them:
  - ECMWF IFS HRES (ECMWF open data, and WeatherBench2's `hres` and `hres_t0`
    stores): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) under the
    [ECMWF terms of use](https://apps.ecmwf.int/datasets/licences/general/), which
    ask for
    - the copyright statement "© [year] European Centre for Medium-Range Weather
      Forecasts (ECMWF)" with the source www.ecmwf.int (for a service built on the
      data: "This service is based on data and products of the European Centre for
      Medium-Range Weather Forecasts (ECMWF)");
    - the licence statement "This data is published under a Creative Commons
      Attribution 4.0 International (CC BY 4.0).
      https://creativecommons.org/licenses/by/4.0/";
    - the disclaimer "ECMWF does not accept any liability whatsoever for any error or
      omission in the data, their availability, or for any loss or damage arising from
      their use.";
    - a note that the data were modified: a STRATA initial condition or replay target
      is HRES remapped to the SCREAM grid, and a forecast from it is model output.
  - ERA5 (from the ARCO-ERA5 store): the
    [Copernicus licence](https://apps.ecmwf.int/datasets/licences/copernicus/), which
    asks for "Contains modified Copernicus Climate Change Service information [year]"
    and a statement that neither the European Commission nor ECMWF is responsible for
    any use made of the Copernicus information or data it contains.

## Disclaimer

This project will download and install additional third-party open-source
software. Review the license terms of those projects before use.
