# ACE→SCREAM Fine-Tuning

This repository now keeps only the Rev 2 forecast-residual workflow.

## Rev 2 workflow

The current workflow starts from the native forecast zarr produced by
`scripts/ace/run_all_tendencies.sh`, regrids that forecast once onto the ACE lat/lon grid,
and then builds paired forecast/truth samples for training. The active
orchestration path is `scripts/ace/run_pipeline.py`.

### Step 1: build the forecast zarr

```bash
bash scripts/ace/run_all_tendencies.sh
```

The native forecast zarr written by `scripts/ace/build_screamcast_forecast.py`
can be inspected with `scripts/plotting/plot_screamcast_tile.py` for quick
tile-level checks before regridding.

### Step 2: set the paths in `scripts/ace/run_pipeline.py`

Update these variables in `scripts/ace/run_pipeline.py`:
- `base` — directory containing `6hour_forecasts.zarr`
- `truth_data` — ACE-grid truth zarr produced by `scripts/ace/regrid_zarr.py`
- `forecast_data` — output path for the ACE-grid forecast zarr
- `rev2_data` — output path for the paired netCDF

### Step 3: build the ACE-grid forecast zarr and training pairs

```bash
python scripts/ace/run_pipeline.py
```

The checked-in `scripts/ace/run_pipeline.py` currently:
- regrids only `step=36` via `selection={"step": slice(35, 36)}`
- builds `ace_rev2_pairs.step36.nc` with `forecast_step=36`
- filters to forecasts with `min_init_time="2020-10-02T00:00:00"`

`scripts/ace/build_ace_forecast_pairs.py` expects an already regridded forecast zarr plus the
ACE-grid truth zarr. It uses the stored `step` coordinate to interpret the
physical forecast lead, so a single stored `+6h` slice still works with
`forecast_step=36`.

The resulting netCDF contains:
- `time_input` `(time,)` — input valid time (`forecast_valid_time - lead_time`)
- `time` `(time,)` — forecast valid time
- `scream_input_state` `(time, scream_channel, lat, lon)` — truth/input state at `time_input`
- `forecast_state` `(time, scream_channel, lat, lon)` — forecast at `time`
- `truth_state` `(time, scream_channel, lat, lon)` — truth at `time`

### Step 4: fine-tune on the preprocessed pairs

```bash
python scripts/ace/finetune_ace2scream_sfno.py \
    --data ace_rev2_pairs.step36.nc \
    --output ace2scream_sfno_rev2.pt \
    --epochs 20 \
    --batch-size 1 \
    --device cuda
```

Rev 2 trains on the model's forecast-output variables. It uses `ps` directly
from the forecast output, and derives `qv_2m`, `u10m`, and `v10m` from the
lowest available model level when constructing ACE inputs.

Minimal smoke test:

```bash
python scripts/ace/finetune_ace2scream_sfno.py \
    --data ace_rev2_pairs.step36.nc \
    --output /tmp/ace2scream_sfno_rev2_smoketest.pt \
    --epochs 1 \
    --batch-size 1 \
    --device cuda \
    --save-every 1
```

## Local nudged forecast comparison

`scripts/ace/run_screamcast_nudged.py` is a local analysis script for
comparing:
- an unnudged SCREAMcast forecast
- a nudged forecast using either ACE residual corrections or data-derived corrections
- the corresponding truth tile

It is useful for qualitative debugging of the ACE correction workflow before
running larger coupled experiments.

Example: ACE-based nudging on a single face/tile

```bash
root=/path/to/project-data/screamcast
ace_checkpoint=$root/inferences/pixeldit_sem1024d24l_pix128d4l_2stepft/sdecadal.ne1024pg2_ne1024pg2.F20TR-SCREAMv1.c10-sep11/finetune_rev2/ace2scream_sfno_rev2_100ep.pt
python scripts/ace/run_screamcast_nudged.py \
    --run-name pixeldit_sem1024d24l_pix128d4l_2stepft \
    --correction ace \
    --ace-checkpoint ${ace_checkpoint} \
    --face 0 \
    --initial-time 2020-10-13T00:00:00 \
    --n-steps 36 \
    /tmp/screamcast_nudged_ace.pth
```

Example: data-based nudging

```bash
python scripts/ace/run_screamcast_nudged.py \
    --run-name pixeldit_sem1024d24l_pix128d4l_2stepft \
    --correction data \
    --coarsen 32 \
    --n-steps 36 \
    --n-steps-outer 1 \
    /tmp/screamcast_nudged_data.pth
```

Notes:
- `--checkpoint` overrides the checkpoint derived from `--run-name`.
- `--coarsen` is only used with the data-based correction, and indicates the amount of grid points 
    that should be pooled-then-interpolated when computing the correction.
- `--nudge-only-wind` is enabled by default; use `--no-nudge-only-wind` to
  apply corrections to all channels.
- Output prediction and truth at the specified lead time are written to the specified output
