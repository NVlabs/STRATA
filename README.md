# Strata

Strata is a deep-learning weather emulator for the
[SCREAM](https://e3sm.org/scream/) global atmosphere model. It trains
transformer-based neural networks to emulate SCREAM atmospheric physics on
the cubed-sphere grid and supports multi-day global forecasting.

- 📄 **Paper:** [Scaling Storm-Resolving Atmospheric AI Simulation to the Entire Planet](https://arxiv.org/html/2606.31248v1) — Accepted at **NeurIPS 2026, Main Track**.
- 🌐 **Project page:** <https://research.nvidia.com/labs/climate/strata/>
- 🤗 **Model weights:** [nvidia/strata](https://huggingface.co/nvidia/strata)
- 🤗 **Training data (Zarr):** [nvidia/STRATA-SCREAM-sdecadal](https://huggingface.co/datasets/nvidia/STRATA-SCREAM-sdecadal), [nvidia/STRATA-SCREAM-sdy1](https://huggingface.co/datasets/nvidia/STRATA-SCREAM-sdy1), [nvidia/STRATA-SCREAM-sdy2](https://huggingface.co/datasets/nvidia/STRATA-SCREAM-sdy2)

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
build for Ada and Hopper GPUs by default (pass `--build-arg CUDA_ARCH=...`
for others, e.g. `8.0` on A100; see [`docker/build_natten.sh`](docker/build_natten.sh))
and PhysicsNeMo pinned to a commit that includes the Strata models.

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

#### Availability

The STRATA code, including the model architecture and training and inference
pipelines, is available in this repository, and the trained model weights at
[huggingface.co/nvidia/strata](https://huggingface.co/nvidia/strata). The raw
SCREAM output used for training and evaluation is available at
<https://portal.nersc.gov/archive/home/n/ndk/www/STRATA2026>. As E3SM model
output, it is distributed under the
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
license (see the
[E3SM data documentation](https://docs.e3sm.org/e3sm_data_docs/_build/html/index.html)).
A training-ready version in Zarr format is available on Hugging Face, one
repository per simulation:

| Repository | Simulation | Time steps (10 min) | Size |
|---|---|---|---|
| [nvidia/STRATA-SCREAM-sdecadal](https://huggingface.co/datasets/nvidia/STRATA-SCREAM-sdecadal) | sdecadal, 14 days (transient historical, F20TR) | 2017 | 36.8 TB |
| [nvidia/STRATA-SCREAM-sdy1](https://huggingface.co/datasets/nvidia/STRATA-SCREAM-sdy1) | DYAMOND1, 7 days (F2010) | 1009 | 19.6 TB |
| [nvidia/STRATA-SCREAM-sdy2](https://huggingface.co/datasets/nvidia/STRATA-SCREAM-sdy2) | DYAMOND2, 7 days (F2010) | 1009 | 19.7 TB |

Each repository is a single Zarr v3 store on the ne1024pg2 grid: arrays are
`(time, lev, ncol)` with 32 model levels and 25,165,824 columns in SCREAM
column order, one shard per variable and time step. The stores carry no
`time`, `lev` or lat/lon coordinate arrays; the code attaches them. For
sdecadal, `data_catalog.py` labels time index 0 as `2020-10-01T00:00`, so
`--initial-time 2020-10-13T00:00:00` below is time index 1728 (the model card
lists the simulated dates).

A store can be read directly over HTTP via `huggingface_hub`'s fsspec
filesystem (installed in the Docker image), which is convenient for
exploration without downloading tens of TB:

```python
import xarray as xr

ds = xr.open_zarr("hf://datasets/nvidia/STRATA-SCREAM-sdy1")
ds["T_2m"].isel(time=500, ncol=slice(0, 100)).values  # streams just the chunks it needs
```

For training, download the stores instead (`hf download <repo> --repo-type
dataset --local-dir <dir>`): the dataloader reads tiles at a rate that HTTP
streaming cannot sustain. Place each store under `ZARR_ROOT` with the
directory name the training configs expect (`_ZARR_SDECADAL` / `_ZARR_SDY1` /
`_ZARR_SDY2` in `train_configs.py`), not the Hugging Face repository name:

| Repository | Directory under `ZARR_ROOT` |
|---|---|
| `nvidia/STRATA-SCREAM-sdecadal` | `sdecadal.ne1024pg2_ne1024pg2.F20TR-SCREAMv1.c10-sep11.out10min.cubesphere.zarr` |
| `nvidia/STRATA-SCREAM-sdy1` | `sdy1.ne1024pg2_ne1024pg2.F2010-SCREAMv1.c10-sep11-f602da2b98.out10min-4lev.cubesphere.zarr` |
| `nvidia/STRATA-SCREAM-sdy2` | `sdy2.ne1024pg2_ne1024pg2.F2010-SCREAMv1.c10-sep11-f602da2b98.out10min-4lev.cubesphere.zarr` |

(Setting `ZARR_ROOT` itself to an `hf://` prefix does not work, because the
configs join it with these directory names.) Download whole stores: a chunk
that is missing from a partial download reads back as zeros, not as an error.

#### Auxiliary files (`AUX_DATA_ROOT`)

Inference and training look up auxiliary files by name in one directory,
`AUX_DATA_ROOT`. The model repository provides some of them; generate the rest:

| File | Source | Used for |
|---|---|---|
| `checkpoints/strata_v1.pth` | [nvidia/strata](https://huggingface.co/nvidia/strata) | weights, normalization statistics, training config |
| `phis.npy`, `sgh30.npy`, `landfrac.npy` | [nvidia/strata](https://huggingface.co/nvidia/strata) | static forcing fields (inference) |
| `latlon_ne1024pg2.nc` | generated by `generate_all.sh` | model lat/lon conditioning (inference and training) |
| `ne1024pg2_scrip.nc` (2.3 GB) | generated by `generate_all.sh` | cell areas for regional averages, regridding |
| `ne1024halo256pg2_scrip.nc` (5.2 GB) | generated by `generate_all.sh` | halo exchange between cubed-sphere tiles |
| `scream_vertical_coordinate.nc` | copied from `data/` in this repository | hybrid level coefficients |

The model repository also holds the normalization statistics as `.npy`
files, `channels.json` and the model card; the code reads the statistics from
the checkpoint. The three grid files are generated from scratch (pure numpy +
netCDF4) by [`data_prep/scrip_generation/`](data_prep/scrip_generation/); the
derived `latlon_ne1024pg2.nc` is bit-identical to the file the released model
was trained with. From the repository root:

```bash
hf download nvidia/strata --local-dir /path/to/strata_v1
export AUX_DATA_ROOT=/path/to/strata_v1
bash data_prep/scrip_generation/generate_all.sh --output-dir $AUX_DATA_ROOT
cp data/scream_vertical_coordinate.nc $AUX_DATA_ROOT/
```

`generate_all.sh` took about 20 minutes on a 64-core node. It also leaves
its intermediate Exodus meshes (`*.g`, about 7 GB) in the output directory;
they are not read afterwards and can be deleted.

A `.env` file is not required if you export `AUX_DATA_ROOT` and
`SCREAM_MAIN_ZARR_PATH`.

## Running the STRATA v1 checkpoint

Assemble `AUX_DATA_ROOT` as above. The forecast reads its initial condition
(and the static land fraction for regional averages) from the sdecadal store,
which can stay remote:

```bash
export SCREAM_MAIN_ZARR_PATH=hf://datasets/nvidia/STRATA-SCREAM-sdecadal
```

`SCREAM_MAIN_ZARR_PATH` also accepts an `s3://` path (with
`SCREAM_ZARR_PROFILE` set to that rclone remote) or a local copy of the store.
With `--correction none` only the initial state is read; `--correction data`
also reads the true state at the end of each nudging window. Download a local
copy (36.8 TB) only if remote throughput is the bottleneck:

```bash
hf download nvidia/STRATA-SCREAM-sdecadal --repo-type dataset \
  --local-dir /path/to/sdecadal.ne1024pg2_ne1024pg2.F20TR-SCREAMv1.c10-sep11.out10min.cubesphere.zarr
export SCREAM_MAIN_ZARR_PATH=/path/to/sdecadal.ne1024pg2_ne1024pg2.F20TR-SCREAMv1.c10-sep11.out10min.cubesphere.zarr
```

Valid initial times are the sdecadal time steps, `2020-10-01T00:00` to
`2020-10-15T00:00` in the code's labelling. The model was trained on
2020-10-02 to 2020-10-10 of this store (plus sdy1 and sdy2), so evaluate from
2020-10-11 onwards. The rollout script reads initial conditions from sdecadal
only (`data_catalog.py`); starting from sdy1 or sdy2 needs a catalog entry
with that store's path and start time.

### Smoke test (2 steps)

The forecast runs under `torchrun` with one process per GPU; the commands
below use 4 GPUs (tested on 4 × H100 80 GB). Build `docker/Dockerfile` for
your GPU (`--build-arg CUDA_ARCH=8.0` on A100; the default covers Ada and
H100). PhysicsNeMo is installed with `--no-deps` in that image; its runtime
imports (`treelib`, `gitpython`, `hydra-core`, `termcolor`) are installed
explicitly. The image does not contain this repository: mount it and work
from its root, e.g.

```bash
docker run --rm -it --gpus all --ipc=host \
  -v $PWD:/workspace/strata -v /path/to/strata_v1:/path/to/strata_v1 \
  -w /workspace/strata strata bash
```

(also mount the output directory, and pass `HF_TOKEN` with `-e` if you use
one; anonymous access to the public repositories works). Inside, put the
repository root on `PYTHONPATH`: the script imports the top-level
`data_catalog` module, which is not part of the installed package.
`TORCH_COMPILE_DISABLE=1` runs the network eagerly; see the note on
`torch.compile` below.

```bash
export PYTHONPATH=$PWD
export AUX_DATA_ROOT=/path/to/strata_v1
export SCREAM_MAIN_ZARR_PATH=hf://datasets/nvidia/STRATA-SCREAM-sdecadal
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

A finished run exits 0 and logs `run_scream step 2/2` and
`Forecast done: 2 steps = 0.3 h`; with `--no-zarr-output` it writes only the
regional averages, to `${OUTPUT}_regional_averages.nc`. On 4 × H100 it took
8 minutes: 3.5 minutes to load the model and fetch the initial condition over
`hf://`, then about 130 s per step with these settings.

The script compiles the network with `torch.compile` unless
`TORCH_COMPILE_DISABLE=1` is set. For forecasts longer than a few steps,
leave it unset: on the same 4 × H100 the first step took 113 s (including
compilation) and later steps 87 s, against about 130 s per step eager, and the
global means agreed with the eager run to floating-point precision. Dynamo
logs a `recompile_limit` warning during the first step; it is harmless.

Each step also logs `global mean precip: liq=0.002 mm/day ice=0.000 mm/day`.
The liquid value is 1000 times too small: the precipitation outputs are
water-depth rates in m/s, and the log converts them as if they were
kg m⁻² s⁻¹, so the number is in m/day. The actual global mean is about
2.6 mm/day. Ice precip `0.000` is a known bug of this checkpoint, not normal
behavior: ice precip is a diagnostic, and training applied a ReLU
non-negative constraint (`do_precip_relu`), so ice precip was not learned
well. It should not affect the other state variables.

### 24 h forecast

These settings turn off tile jitter and the halo adjoint across cube-face
edges, and filter omega with SHT lmax 3. A 1 h forecast with them, from the
public files, agreed with our reference rollout of this checkpoint (RMSE
against SCREAM equal to three significant figures):

```bash
unset TORCH_COMPILE_DISABLE   # compile pays off over 144 steps
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
  --tile-jitter 0 \
  --sht-omega-lmax 3 \
  --inference-batch-size 1 \
  --output-variables T_2m ps U V PotentialTemperature qv omega precip_liq_surf_mass_flux \
  --output-levels 13,18,25,30 \
  --initial-time 2020-10-13T00:00:00 \
  $OUTPUT.zarr
```

On 4 GPUs keep `--inference-batch-size 1`: batch 4 ran out of memory on
4 × H100 80 GB (it fits when the tiles are spread over more GPUs, as in
[`slurm/submit_inference.sh`](slurm/submit_inference.sh), which uses 16).
Eager, these settings took about 140 s per step on 4 × H100, so 144 steps
take several hours; leave `torch.compile` on and use more GPUs to shorten
that. Each written step holds every selected channel on the full grid
(about 100 MB per channel before compression), so choose
`--output-variables` and `--output-levels` deliberately.

### Forecast output

Without `--no-zarr-output`, the forecast is written to the Zarr store
`$OUTPUT`, and area-weighted regional means (global, land, ocean, tropics,
extratropics) of every output channel to `${OUTPUT}_regional_averages.nc`.
Arrays in the store are `(time, step, face, x, y)` for 2-D fields and
`(time, step, level, face, x, y)` for 3-D fields, on the six 2048 × 2048
cube faces:

- `time`: initial time; `step`: lead time in seconds (`600` is the first
  10-minute step);
- `level`: SCREAM model-level index (8–31, from the top), not pressure;
  `hyam` / `hybm` give the hybrid coefficients, and `--output-levels`
  selects levels by these indices;
- `lat`, `lon`: `(face, x, y)` cell centres in degrees;
- `--output-variables` selects variables, e.g. `T_2m U V qv ps`.

Variable units are SCREAM's: precipitation fluxes are in m/s.

> **Known issue — keep `--output-steps 1` (the default).** With
> `--output-steps N > 1` the writer saves the states after steps 1, N+1,
> 2N+1, … but labels them with leads N, 2N, 3N, … steps, so every saved
> frame is (N−1) × 10 minutes earlier than its `step` label says. Write
> every step and subsample afterwards until this is fixed.

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
`sbatch`. It needs the same environment as the forecast above
(`PYTHONPATH`, `AUX_DATA_ROOT`, `SCREAM_MAIN_ZARR_PATH`), exported in the
job or loaded from `.env`. Run `python3 scripts/ace/run_screamcast_nudged.py --help` for the full
CLI (checkpoint, number of steps, tile/halo size, omega filtering, output
levels, initial time, ...). `--output-levels` selects which vertical levels are
written to the output zarr.

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
  CC BY 4.0, as E3SM model output.

## Disclaimer

This project will download and install additional third-party open-source
software. Review the license terms of those projects before use.
