# Replay to a driver forecast

A free STRATA rollout from a real initial condition keeps its fine scales realistic,
but its large scales drift away from the real weather within a day. Replay keeps the
large scales on a driver forecast, for example ECMWF HRES, and leaves the scales below
a few hundred kilometres to STRATA. STRATA then downscales the forecast dynamically to
its 5 km, 10-minute grid. It works for global runs and for regional windows
([regional_window.md](regional_window.md)).

## How it works

Every `--replay-every-hours` (default 6) hours, at a time `t` where a driver state
`D(t)` exists, the run is compared with the driver on 8 x 8-pixel blocks (about 40 km):

```
delta = blockmean(D(t)) - blockmean(x(t))
```

`x(t)` is the run's own forecast to `t`. The increment is low-passed on the sphere with
a Gaussian spherical-harmonic taper whose gain is 1/2 at degree `--replay-ell-half`
(default 160, a wavelength of about 250 km) and 1 at degree 0, truncated where the gain
drops below 1 %. The run is then rewound to `t - 3 h` and run again to `t + 3 h`, with
`delta / 36` added after each of the 36 steps (an incremental analysis update). Half of
the increment is in by `t`. Spreading it over the window keeps the model from a shock,
and scales below the taper stay STRATA's own throughout.

- The replayed inputs are `U`, `V`, `PotentialTemperature`, `qv`, `z_mid`, `omega` and
  `T_2m` on all model levels (`--replay-vars`). `strata_v1` predicts `ps` as a diagnostic,
  so it is not replayed.
- An analysis at `t` needs a driver at `t` and `t + every/2` within the run. A 24 h run
  with drivers every 6 h to +24 h is replayed at +6, +12 and +18 h; the last 3 hours are
  a free forecast. Build drivers to +30 h and run 27 h or more to replay at +24 h too.
- Only the final trajectory is written: every output hour once, in order. The predictor
  windows that are re-run are not written.
- Each analysis re-runs 3 hours, so a 24 h run replayed at +6, +12 and +18 h costs 33
  hours of steps.
- With `--sht-omega-lmax N` (the omega high-pass of global runs) the omega increment
  keeps none of the degrees the high-pass removes.
- In a regional window the analysis covers the window's blocks only: the increment is
  the low-passed window increment and zero outside, so it is weaker within a few hundred
  kilometres of the window edge. The window's edge is the anchored ring, from the same
  driver stores.

Each rank analyses its own blocks and the spherical-harmonic coefficients are
all-reduced (`lmax` 413 at the default taper), then each rank evaluates them on its own
pixels.

## Run

Build the driver stores from an HRES forecast as for a window (the lead-0 store is the
initial condition; see
[regional_window.md](regional_window.md#drive-a-window-with-an-ecmwf-hres-forecast)):

```bash
python -m screamcast.initial_conditions.hres_forecast \
  --init 2024-09-25T12 --leads 0 6 12 18 24 --tag-prefix helene \
  --out-dir "$WORK/ics" --cache "$WORK/grib" \
  --opendata-root https://storage.googleapis.com/ecmwf-open-data \
  --map "$WORK/map/map_721x1440_to_ne1024pg2.nc" --phis "$WORK/scream_phis.zarr" \
  --latlon "$AUX_DATA_ROOT/latlon_ne1024pg2.nc" --scream-ref "$WORK/scream_static" \
  --vert "$WORK/vert_coord_E3SM_L128.nc"
```

A replayed window adds `--replay` to the window command:

```bash
torchrun --nproc_per_node=8 scripts/ace/run_screamcast_nudged.py \
  --run-name helene_window_replay \
  --checkpoint $AUX_DATA_ROOT/checkpoints/strata_v1.pth \
  --ic-store "$WORK/ics/scream_hiccup_ne1024pg2_helene_f000.zarr" \
  --driver-stores "$WORK"/ics/scream_hiccup_ne1024pg2_helene_f0*.zarr \
  --region 3,10,14,7,11 --replay \
  --forecast-model screamcast --correction none --n-steps 144 --output-steps 6 \
  --tile-size 128 --halo-width 16 --halo-adjoint --no-halo-adjoint-cross-face \
  --tile-jitter 4 --tile-jitter-mode cycle16 --inference-batch-size 1 \
  --output-variables T_2m ps U V qv omega precip_liq_surf_mass_flux \
  --output-levels 19,24,31 \
  $OUTPUT
```

A replayed global run is the README's global command with `--ic-store`,
`--driver-stores` and `--replay`. It needs more GPUs; on two nodes, one task per GPU:

```bash
srun --nodes=2 --ntasks-per-node=8 python scripts/ace/run_screamcast_nudged.py \
  --run-name helene_global_replay \
  --checkpoint $AUX_DATA_ROOT/checkpoints/strata_v1.pth \
  --ic-store "$WORK/ics/scream_hiccup_ne1024pg2_helene_f000.zarr" \
  --driver-stores "$WORK"/ics/scream_hiccup_ne1024pg2_helene_f0*.zarr \
  --replay --sht-omega-lmax 3 \
  --forecast-model screamcast --correction none --n-steps 144 --output-steps 6 \
  --tile-size 128 --halo-width 16 --halo-adjoint --no-halo-adjoint-cross-face \
  --tile-jitter 4 --tile-jitter-mode cycle16 --inference-batch-size 1 \
  --output-variables T_2m ps U V qv omega precip_liq_surf_mass_flux \
  --output-levels 19,24,31 \
  $OUTPUT
```

The log has one line per analysis and variable with the root-mean-square of the block
increment and of its low-passed part (`replay +6 h U_24: increment rms ...`).

Without `--driver-stores` and without `--ic-store`, the driver is the SCREAM run in
`SCREAM_MAIN_ZARR_PATH` every `--driver-every-hours`: a replay toward a known truth.

## Resources

RESOURCES_TODO

## Limitations

- No skill has been measured. With replay the track and the large-scale intensity of a
  storm follow the driver by construction; what STRATA adds is the structure below the
  taper scale (rain bands, eyewall, convective cells).
- The increment is computed every 6 hours from 6-hourly drivers; between analyses the
  run is free.
- `--correction` nudging and `--replay` do not combine.
