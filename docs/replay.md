# Replay to a driver forecast

A free STRATA rollout from a real initial condition keeps its fine scales realistic,
but its large scales drift away from the real weather within a day. Replay keeps the
large scales on a driver forecast, for example ECMWF HRES, and leaves the scales below
a few hundred kilometres to STRATA. STRATA then downscales the forecast dynamically to
its 5 km, 10-minute grid. It works for global runs and for regional windows
([regional_window.md](regional_window.md)).

## Run

The README's [Forecasts from ERA5 or ECMWF HRES](../README.md#forecasts-from-era5-or-ecmwf-hres) has the spec
and the command: a replayed run adds `--replay` and `--driver-onthefly SPEC.json` (with
`--driver-target-dir DIR`), on the globe or on a window, and usually builds its initial
condition from the same spec (`--ic-onthefly`). The log has one line per analysis and
variable with the root-mean-square of the block increment and of its low-passed part
(`replay +6 h U_24: increment rms ...`). Global runs with drivers built on the fly have
not been run yet; the global timing in [Resources](#resources) is from driver stores.

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
  `T_2m` on all model levels. `--replay-vars` chooses among them: for example
  `--replay-vars U,V,PotentialTemperature,qv,z_mid` leaves `omega` and `T_2m` to STRATA,
  and a variable left out gets no increment at any level (`--replay-level-ramp` weights
  the levels of the ones kept). These seven are all of `strata_v1`'s inputs; `ps` and
  precipitation are diagnostics, outputs only, so they cannot be replayed. The choice
  does not apply to a window's anchored ring, which is a boundary condition: it follows
  the driver in every input variable.
- An analysis at `t` needs a driver at `t` and `t + every/2` within the run. A 24 h run
  with drivers every 6 h to +24 h is replayed at +6, +12 and +18 h; the last 3 hours are
  a free forecast. Run 27 h or more (`--n-steps 162`) to replay at +24 h too (from
  stores, build them to +30 h).
- Only the final trajectory is written: every output hour once, in order. The predictor
  windows that are re-run are not written.
- Each analysis re-runs 3 hours, so a 24 h run replayed at +6, +12 and +18 h costs 33
  hours of steps.
- With `--sht-omega-lmax N` (the omega high-pass of global runs, which removes degrees
  below N) the omega increment has no degree below N either.
- In a regional window the analysis covers the window's blocks only and the increment
  is zero outside. The low-pass alone would give a cell near the window edge only the
  share of the kernel that lies inside (half on the edge, 84 % about 47 km in at the
  default taper), so the low-passed increment is divided by the low-passed window
  indicator, `K*(m delta) / K*(m)` (`--replay-region-edge normalized`, the default): a
  uniform difference is added whole up to the edge, and away from the edge the
  increment is the global one. `--replay-region-edge zero` keeps the share. The
  window's ring is the anchored boundary, from the same driver.
- `--replay-level-ramp PFULL,PZERO` (e.g. `800,900`) leaves the boundary layer to
  STRATA: the increment of each level is weighted 1 at reference pressures up to PFULL
  hPa, 0 from PZERO hPa down, linear between. The reference pressure is
  `(hyam + hybm) * 1000 hPa` of the model levels. Surface channels are not weighted;
  leave `T_2m` out of `--replay-vars` to leave the surface free too.
- The output stores have a `history` attribute with the time and the exact command
  of the run.

Each rank analyses its own blocks and the spherical-harmonic coefficients are
all-reduced (`lmax` 414 at the default taper), then each rank evaluates them on its own
pixels.

## Drivers built on the fly

The replay uses the driver only as 8 x 8-block means at its analysis hours, and the
anchored ring of a window only as the 8 x 8-block means of the window and its ring at the
driver hours (its `smooth`, `--boundary-block 8`). So `--driver-onthefly SPEC.json
--driver-target-dir DIR` builds only those: it starts a producer process on rank 0's GPU
(`screamcast/replay_targets.py`). For each hour the run needs, the producer fetches the
source state, runs `fast_ic.build_ic` on it 2^21 columns at a time, and writes the block
means to `DIR/blocks_<YYYYMMDDHH>.zarr` (about 600 MB for the globe, 6 MB for a 4 x 4-tile
window, 14 MB with its ring). A window's producer builds only the window's columns, and
its ring's when the ring is anchored. Each rank reads the blocks it needs on a background
thread. The fine state never goes to disk.

- With `--replay` the targets are at the analysis hours; with an anchored ring also at
  every `--driver-every-hours` (default 6) from 0 to the first at or after the last step.
  A window can use them without `--replay` too: then they drive the ring alone.
- The ring takes every input variable from the targets; the replay takes
  `--replay-vars`.
- `--driver-producer gated` (default) builds the targets the run needs before its first
  analysis at once (the ring's +0 and +6 h, the first analysis) and each later one when
  the run begins the last analysis before it needs that target; rank 0 empties its CUDA
  cache first, so the build does not meet STRATA's peak memory of start-up and
  compilation. A window without `--replay` has no analyses and is not gated. `ahead`
  builds every target as early as it can. A rank that needs a target not yet built asks
  for it, and a gated producer then builds it without waiting for its gate. The next
  source state is fetched while the GPU builds the current one. On the 4 x 4-tile Helene
  window (H100, 2026-11-19) a target took about 7 s (2.6 s to read the cached GRIB, 2 s
  on the GPU).
- A target is written under `.partial<pid>` and renamed, so a rank never reads half a
  store. Each run has its own directory `DIR/runs/<run-name>-<time>-<pid>`: there the
  producer writes `FAILED` (with the traceback) or `DONE` and its `producer.log`, and
  the run its markers, so runs that share `DIR` do not disturb each other. A rank that
  waits longer than `--driver-target-timeout` seconds (default 3600) for a target fails.
- The directory must be on a file system every rank sees. Each target records its
  origin: the source (the ERA5 store, or the HRES forecast's init) and the absolute
  paths of `map`, `phis` and `vert`. A target already there is used only if it has this
  run's origin and holds its channels and tiles, so a rerun builds nothing; any other
  target with the name of one the run needs (say, from another HRES forecast) stops the
  run, in the producer and in every rank, before it is used.
- Relative paths, in the options and in the spec, are taken from the run's working
  directory, also by the producer.
- `python -m screamcast.replay_targets produce --spec SPEC.json --out-dir DIR
  --run-init 2024-09-25T12 --hours 6 12 18 [--region R --tile-size 128 [--ring]]`
  builds targets ahead of a run, for example on a node that can reach the source when
  the compute nodes cannot; the run then reuses them.

### The initial condition

`--ic-onthefly SPEC.json` (with `--driver-target-dir DIR`): at start-up, before the model
loads, rank 0 builds the full initial condition from the spec's source with
`fast_ic.build_ic` (the steps of `fast_ic`, `hres_ic` and `hres_forecast`) into
`DIR/ic_<YYYYMMDDHH>.zarr` (about 9 GB). The spec may be the driver's; it also needs the
statics `latlon` (the `lat`/`lon` NetCDF) and `scream_ref` (the directory with
`sgh30.zarr` and `landfrac.zarr`), as `fast_ic` does. The initial time is
`--initial-time`, by default the forecast's `init` for `hres` and `wb2_hres`; `era5`
needs `--initial-time`.

- Like a target, the store is written under `.partial<pid>` and renamed, and records
  its origin (the targets' origin, its time, and the absolute paths of `latlon` and
  `scream_ref`). A store already there is used if it has this run's origin, so a rerun
  builds nothing; one with another origin stops the run.
- The other ranks wait for rank 0's `IC_READY` (the store's path) or `FAILED` (with
  the traceback) in the run's directory, up to `--driver-target-timeout` seconds. They
  do not wait in a collective, so a build longer than the NCCL timeout does not fail
  the run.
- A `fast_ic` build of the globe takes about 80 s on one GPU, plus reading the source
  state.

### Sources

**Forecast or analysis.** `"hres"` and `"wb2_hres"` read one HRES *forecast*, the one
started at `init`. Its lead 0 is the HRES analysis at `init`, and that is the initial
condition. The driver at +6 h, +12 h, ... is that forecast's +6 h, +12 h, ... lead, not
the HRES analysis at those times. A replayed run from them therefore uses nothing
known after `init`: it is a forecast, the HRES forecast downscaled by STRATA.
`"era5"` is a reanalysis at every hour: replaying to it pulls the run toward the
observed weather at each analysis time, so it is a hindcast. An `--initial-time` later
than `init` starts from that forecast lead, not from an analysis.

`"source": "era5"` reads ERA5 from the public ARCO store (`era5_zarr` to use another),
which has every hour. `"hres"` reads the ECMWF HRES forecast `init` from the open-data
bucket (`opendata_root` to use a mirror). The bucket holds the 00 and 12 UTC forecasts,
3-hourly to +144 h and 6-hourly to +240 h; the 06 and 18 UTC runs are not there. Files
with every field `build_ic` needs start with the 2024-03-05 12 UTC forecast: the files
from 2024-02-29 to 2024-03-05 00 UTC lack four pressure levels and `w`. Checked on
2026-11-19, when forecasts to 2026-11-18 were there. `"wb2_hres"` reads the HRES
forecast `init` from WeatherBench2's `hres` store (`wb2_zarr` to use a copy): the 00 and
12 UTC forecasts from 2016-01-01 to 2023-01-10 12 UTC, 6-hourly to +240 h, on the same
13 levels with `w` and surface pressure (checked on 2026-11-24). The analyses fall at
the hours the source has; the ring's driver hours must all be there. The 2023 HRES files
at 0.4 deg (9 levels, no `w`) are not a source.

The HRES data, from open data and from WeatherBench2, are ECMWF's under CC BY 4.0;
ERA5 is under the Copernicus licence. Both ask you to credit the source in work that
uses them, replayed runs included: the [README](../README.md#license) gives the
wording each asks for.

## Driver stores

A source the producer does not read (another model, the 2023 HRES files) can drive a run
as full-resolution stores instead: `--ic-store` and `--driver-stores` take the place of
`--ic-onthefly` and `--driver-onthefly`. `hres_forecast` builds one store per lead from
an HRES forecast (about 9 GB each; the lead-0 store is the initial condition; see
[regional_window.md](regional_window.md#driver-stores)):

```bash
python -m screamcast.initial_conditions.hres_forecast \
  --init 2024-09-25T12 --leads 0 6 12 18 24 --tag-prefix helene \
  --out-dir "$WORK/ics" --cache "$WORK/grib" \
  --opendata-root https://storage.googleapis.com/ecmwf-open-data \
  --map "$WORK/map/map_721x1440_to_ne1024pg2.nc" --phis "$WORK/scream_phis.zarr" \
  --latlon "$AUX_DATA_ROOT/latlon_ne1024pg2.nc" --scream-ref "$WORK/scream_static" \
  --vert "$WORK/vert_coord_E3SM_L128.nc"
```

```bash
torchrun --nproc_per_node=8 scripts/ace/run_screamcast_nudged.py \
  --run-name helene_window_replay_stores \
  --checkpoint $AUX_DATA_ROOT/checkpoints/strata_v1.pth \
  --ic-store "$WORK/ics/scream_hiccup_ne1024pg2_helene_f000.zarr" \
  --driver-stores "$WORK"/ics/scream_hiccup_ne1024pg2_helene_f0*.zarr \
  --region 3,10,14,7,11 --replay \
  ... (the other options of the README's command)
```

The initial condition and the drivers are chosen apart: `--ic-store` also combines with
`--driver-onthefly`, and `--ic-onthefly` with `--driver-stores`. With none of the four
options, the run starts from and is driven by the SCREAM run in
`SCREAM_MAIN_ZARR_PATH`, every `--driver-every-hours`: a replay toward a known truth.

On the Helene window (H100, 2026-11-19) the on-the-fly targets were bitwise equal to
the block means of the stores built from the same forecast. The run itself is not
bitwise reproducible on GPU (two runs from stores differ by 0.8 m/s in `U` at +24 h),
and the on-the-fly run differed from the stores run by as much, also with the anchored
ring built on the fly.

## Resources

Measured on one node of 8 NVIDIA H100 80 GB GPUs (2026-11-24) with `torch.compile`,
everything on the fly (initial condition, replay targets and anchored ring), 24 h
replayed at +6, +12 and +18 h with hourly output; the whole run, start-up included:

| Window | Driver | Tiles | Whole run |
|---|---|---|---|
| Helene, `3,10,14,7,11` | HRES open data, 2024-09-25 12 UTC | 16 | 5 min 15 s |
| Iota, `3,8,12,7,11` | WeatherBench2 HRES, 2020-11-16 12 UTC | 16 | 5 min 15 s |
| Kirk across a face edge, `3,10,15,12,20` | HRES open data, 2024-10-02 12 UTC | 40 | 8 min 9 s |
| Kirk at a cube corner, `3,12,20,12,16` | HRES open data, 2024-10-06 00 UTC | 32 | 7 min 18 s |

The Iota run was bitwise equal through +18 h to an earlier 48 h run of the same case from
driver stores; they part at +24 h, where the 48 h run is already in its +24 h analysis.

With driver stores (2026-11-17, Helene, 5 stores from +0 to +24 h):

| Step | 4 x 4-tile window, 1 node (8 GPUs) | Global, 2 nodes (16 GPUs) |
|---|---|---|
| Start-up: model, halo plan, initial condition and drivers | about 2 min 40 s | about 2 min 15 s |
| One model step | 0.5 s | about 21 s |
| One analysis (increment and its low-pass) | about 5 s | about 3.5 min |
| 24 h: 144 written steps + 54 re-run steps + 3 analyses | 2 min 23 s | 84 min |
| Whole run | 5 min 19 s | 87 min |

The free (not replayed) window takes about 4 minutes on the same node
([regional_window.md](regional_window.md#resources)). A global analysis is slower because
every rank evaluates about 172,000 spherical-harmonic modes on its 1.6 million pixels.

## Limitations

- No skill has been measured. With replay the track and the large-scale intensity of a
  storm follow the driver by construction; what STRATA adds is the structure below the
  taper scale (rain bands, eyewall, convective cells).
- The increment is computed every 6 hours from 6-hourly drivers; between analyses the
  run is free.
- `--correction` nudging and `--replay` do not combine.
- Drivers and initial conditions built on the fly come from ERA5, HRES open data (from
  2024-03-05 12 UTC) or WeatherBench2's HRES forecasts (2016-01-01 to 2023-01-10
  12 UTC) only; other sources (the 2023 HRES files at 0.4 deg, other models) need
  stores.
- Global runs with drivers built on the fly have not been run yet.
