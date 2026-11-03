# SCREAM-like initial conditions from ERA5 (and ECMWF HRES) with HICCUP

`initial_conditions/` builds EAMxx/SCREAM atmospheric initial conditions on
**ne1024pg2** (`ncol = 25165824`) with the **E3SM L128** hybrid vertical grid
from ERA5, in the 32-level Zarr v3 layout that the SCREAM rollout loaders read.
There are two pipelines:

- **`fast_ic.py` (use this).** A torch port of the HICCUP steps that runs on
  one GPU. It builds an IC for any ERA5 hour in about 80 s end to end, and
  further ICs in the same run take about 25 s each. Its output matches the
  HICCUP pipeline bit for bit apart from last-bit differences in a few `qv`
  values.
- **`run_hiccup_eamxx.py` + `hiccup_l128_to_zarr.py` (reference).** Calls the
  E3SM [HICCUP](https://github.com/E3SM-Project/HICCUP) functions directly
  through HICCUP's NetCDF multifile workflow. It takes 2.5-3 h per IC and is
  kept as the ground truth the fast path is validated against.

| Script | Role |
|---|---|
| `fast_ic.py` | ERA5 timestamp(s) to Zarr IC(s) on one GPU (see [Fast GPU path](#fast-gpu-path)) |
| `run_hiccup_eamxx.py` | One timestamp: ERA5 extraction, HICCUP pipeline, PotentialTemperature and `z_mid`, combined NetCDF IC |
| `hiccup_l128_to_zarr.py` | NetCDF IC to a one-frame Zarr v3 store (32 levels, flat `ncol`, consolidated metadata) |
| `run_tc_batch.py` | Loops over `layout.CASES` (10 tropical-cyclone ICs): build, convert, upload to an rclone remote with a manifest |
| `build_map.py` | Builds (in parallel) or reuses the ERA5 721x1440 to ne1024pg2 `fv2fv_flx --lrg2sml` map |
| `layout.py` | Grid and level constants, Zarr variable lists, the TC case list, file-name helpers |
| `hiccup_time_calendar.patch` | Required local fix to HICCUP (see below) |
| `hres_ic.py` | The same HICCUP steps on the ECMWF HRES analysis (see [HRES initial conditions](#hres-initial-conditions)) |

All Python entry points are modules. Run them from the repository root with
`python -m initial_conditions.<name>`.

## Inputs

| Input | Notes |
|---|---|
| ERA5 | ARCO `gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3`, read over HTTPS. **37 pressure levels** (1-1000 hPa), 0.25 deg 721x1440, north-to-south, starting at Greenwich. Time index is hours since 1900-01-01. The HICCUP path extracts each timestamp to `<data-root>/data/ERA5.{atm,sfc}.<date>.<HH>.nc`. |
| SCREAM `phis`, `sgh30`, `landfrac` | The `phis`, `sgh30` and `landfrac` arrays of the [nvidia/STRATA-SCREAM-sdy1](https://huggingface.co/datasets/nvidia/STRATA-SCREAM-sdy1) dataset (about 60 MB together; see below). |
| Map | `map_721x1440_to_ne1024pg2.nc` (3.1 GB), `ncremap -a fv2fv_flx --lrg2sml` from the ERA5 grid to ne1024pg2, with a `.command.txt` sidecar next to it. Build it once with `build_map.py`, about 30 min on 72 cores (see [Map](#map)). `fast_ic` and `hres_ic` only read it. |
| ne1024pg2 SCRIP | `ne1024pg2_scrip.nc`, the destination grid for the map (`grid_size = 25165824`) |
| lat/lon | `latlon_ne1024pg2.nc`, NetCDF with `lat`/`lon` on `ncol`. They are written to the Zarr as stored (radians) and converted to degrees for `coszr`. |
| L128 coordinate | HICCUP `files_vert/vert_coord_E3SM_L128.nc` (default `--vert`) |

`ne1024pg2_scrip.nc` and `latlon_ne1024pg2.nc` are the auxiliary files the
README's setup section generates with
`bash data_prep/scrip_generation/generate_all.sh --output-dir <AUX_DATA_ROOT>`.

Download the three SCREAM static arrays once:

```bash
hf download nvidia/STRATA-SCREAM-sdy1 --repo-type dataset \
  --include 'phis/*' --include 'sgh30/*' --include 'landfrac/*' \
  --local-dir "$WORK/sdy1_static"
mkdir -p "$WORK/scream_static"
mv "$WORK/sdy1_static/phis" "$WORK/scream_phis.zarr"
mv "$WORK/sdy1_static/sgh30" "$WORK/scream_static/sgh30.zarr"
mv "$WORK/sdy1_static/landfrac" "$WORK/scream_static/landfrac.zarr"
```

Give one `--include` per pattern: the `hf` CLI reads further bare patterns as
file names.

### Map

`build_map.py` builds the map in parallel. It needs conda-forge `nco` and
`tempest-remap` (see [Environment](#environment)), and writes the ERA5 SCRIP
file with HICCUP's `ncremap -G` command if `--src-scrip` does not exist:

```bash
python -m initial_conditions.build_map \
  --dst-scrip "$AUX_DATA_ROOT/ne1024pg2_scrip.nc" \
  --src-scrip "$WORK/grid/scrip_ERA5_721x1440.nc" \
  --map "$WORK/map/map_721x1440_to_ne1024pg2.nc"
```

`ncremap -a fv2fv_flx --lrg2sml` runs `GenerateOverlapMesh` and
`GenerateOfflineMap` serially, which takes 13-18 h at this size. The map is
first order, so the weights of an ne1024pg2 column depend only on that
column's overlaps with the ERA5 cells. `build_map.py` runs the same two
commands on contiguous column ranges in a process pool (`--workers` defaults
to all cores, with two ranges per worker) and stacks the results. On 72 Grace
cores the map took 32 min; the two ranges that hold the poles take about three
times longer than the others and finish last. A range of 87k columns used
0.8 GB.

Against the serial ncremap map built on an x86 machine (NCO 5.3.6), the
parallel map has the same `row` and `col` (38,299,240 links in the same
order). 31,590 weights (0.08%) differ after rounding to float32; the largest
relative difference is 1.3e-5, on sliver overlaps. A smooth temperature-like
field remapped with the two maps differs by at most 1.8e-5 K. Two different
splits of the same columns gave weights within 7e-9 of each other and the same
float32 weights, so the differences come from the build machine, not from the
split.

An existing map is reused when its `.command.txt` sidecar is next to it
(`build_map.py` writes one), and a map without a sidecar is adopted after its
`n_a`/`n_b` sizes are checked. `run_hiccup_eamxx.py` reuses the map only when
the sidecar is there; otherwise it asks HICCUP to build a new map serially, so
run `build_map.py` first.

## Fast GPU path

`fast_ic.py` takes one or more UTC timestamps, reads those hours from the ARCO
ERA5 store and writes one Zarr IC per timestamp to
`<out-dir>/scream_hiccup_ne1024pg2_<tag>.zarr`. The tag defaults to
`YYYYMMDDHH`. It needs a CUDA GPU with about 50 GB of memory and the STRATA
Docker image (torch, zarr 3, netCDF4 and gcsfs). It does not need NCO,
TempestRemap or a HICCUP clone; it reads the map and static arrays from
[Inputs](#inputs).

```bash
docker run --rm --gpus all --ipc=host --user "$(id -u):$(id -g)" \
  -v "$PWD:$PWD" -v "$WORK:$WORK" -w "$PWD" -e PYTHONPATH="$PWD" \
  strata \
  python -m initial_conditions.fast_ic \
    --timestamp 2017-09-14T00 2017-09-18T12 \
    --out-dir "$WORK/ics" \
    --map "$WORK/map/map_721x1440_to_ne1024pg2.nc" \
    --phis "$WORK/scream_phis.zarr" \
    --latlon "$AUX_DATA_ROOT/latlon_ne1024pg2.nc" --scream-ref "$WORK/scream_static" \
    --vert "$WORK/vert_coord_E3SM_L128.nc"
```

- Run the container as yourself (`--user`), or the stores and caches it
  writes are owned by root. Mount `$AUX_DATA_ROOT` too if it is outside
  `$WORK`.
- If your torch build has no kernels compiled for your GPU, CUDA compiles
  them on first use. Point `CUDA_CACHE_PATH` at persistent storage to keep
  them between runs.
- `--vert` should be HICCUP's float64 `vert_coord_E3SM_L128.nc` (it is only
  read, so copy it out of a clone once). The repository's
  `data/scream_vertical_coordinate.nc` is a float32 copy of the same
  coordinate; it also works but shifts results by float32 rounding.
- `--era5-atm`/`--era5-sfc` read the `ERA5.{atm,sfc}.<date>.<HH>.nc` files
  that `run_hiccup_eamxx.py` extracts instead of ARCO; this is how the
  validation below isolates the computation from the download.
- The stores are consolidated when they are written.

### How it works

The computation follows HICCUP step by step, with the float32/float64 types and
operation order that HICCUP gets from numpy 2:

1. `remap_horizontal`: the ncremap map as a CSR matrix on the GPU, applied to
   all 188 ERA5 level-fields in one sparse matmul, accumulated in float64 and
   stored as float32 (as ncremap does).
2. `adjust_surface_pressure`: HICCUP's formula, using T on the 1000 hPa level
   and `del_phis = PHIS_ERA5 - PHIS_SCREAM`.
3. Bottom-up loop over the 128 target levels across all columns:
   `log_pressure_stencil` and `interp_level` reproduce HICCUP's
   `_interp_block`. The bracketing indices and weights depend only on
   pressure, so they are computed once per level and shared by T, qv, U, V
   and omega. Then come `remove_supersaturation`, the hydrostatic `z_mid`
   accumulation in `run_hiccup_eamxx`'s order, and PotentialTemperature. On
   the 46 pure pressure levels (above 146 hPa) the pressure is the same in
   every column, so those float64 logs and powers are evaluated once.
4. Each output level goes to a pool of writer threads as soon as it is final.
   They zstd-encode the chunk with the codec from the store's own metadata and
   write it to the key its chunk key encoding gives. zarr's own writer encodes
   these 100 MB chunks one at a time, which takes minutes. Each store is
   consolidated, and one chunk per array is read back through zarr.

In a batch, the map, statics and imports are loaded once, the next hour is
downloaded while the current one is computed, and each store finishes writing
while the next IC runs on the GPU.

### Validation against HICCUP

The fast path rebuilt the ten tropical-cyclone ICs in `layout.CASES` from
ARCO, and each was compared value by value with the HICCUP-pipeline store of
the same name. Across the ten:

- `ps`, `T_2m`, `U`, `V`, `omega`, `PotentialTemperature` and all statics are
  bitwise identical, except for a single last-bit difference in
  `PotentialTemperature`, `V` or `omega` in four of the cases.
- `qv` differs in about 0.006% of values, by at most 3.4e-7 relative (a few
  float32 ulps). The cause is float32 `exp` in Bolton's saturation formula:
  CUDA's `expf` and glibc's `expf`, which numpy calls, round differently.
- `z_mid` differs in about 4 values per 10 million, by one float32 ulp
  (4 mm at 37 km), through those `qv` values.

### Timing

On one NVIDIA GB300 (72 Grace cores), starting from ARCO:

| Step | One IC (2021-07-15T12) | Ten TC ICs in one run |
|---|---|---|
| Container start and imports (torch alone takes 11 s) | 27 s | 28 s |
| Map and statics, with the ERA5 download overlapped | 14 s | 14 s |
| GPU steps, with zstd writing overlapped | 21 s | 18-28 s per IC |
| Last store's remaining writes, consolidation and read-back | 15 s | 16 s |
| **Wall time** | **77 s** | **285 s (about 24 s per further IC)** |

The HICCUP path takes 2.5-3 h per IC on the same machine. 81 min of that is the
vertical remap, and 27 min goes to NetCDF combine and attribute cleanup.
Writing is CPU-bound: the 192 3D chunks are about 1,000 CPU-seconds of zstd
at the reference stores' compression level.

## HRES initial conditions

`hres_ic.py` runs the same `fast_ic.build_ic` HICCUP steps on the ECMWF HRES
0-h analysis from WeatherBench2 `hres_t0` (2016-2022, every 6 h, 0.25 deg, 13
levels from 50 to 1000 hPa) for the whole globe. It runs on one GPU, or with
`--device cpu` in about 5 min on 64 cores.

- **Grid.** `hres_t0` is on the ARCO ERA5 721x1440 grid, south to north. After
  flipping latitude the ERA5 `fv2fv_flx` map (`--map`) applies unchanged; no
  new map is built.
- **Surface geopotential.** `hres_t0` has none, and the ps adjustment needs it.
  It is taken from the HRES geopotential profile at the HRES surface pressure
  (log-pressure interpolation between the bracketing levels; below 1000 hPa the
  hypsometric equation with the 1000 hPa temperature), so it is consistent with
  the ps it adjusts.
- **Top levels are not physical.** HRES has nothing above 50 hPa, and HICCUP
  holds its 50 hPa values up to the model top. The five output levels above
  50 hPa (L128 3-19, 4.6-39 hPa) therefore carry 50 hPa values: the top output
  level's PotentialTemperature comes out about 133 K too low on average. STRATA
  reads L128 `[3::4][8:]` (levels 35-127; the top one is at 100.18 hPa), so its
  inputs never include them. Do not use these ICs for anything that reads the
  output levels above 100 hPa, and expect a full-column IC check to flag them.
- **Boundary layer.** HRES has 3 levels from 850 to 1000 hPa where ERA5 has 7,
  so its boundary layer is interpolated over coarser steps.
- **Coverage.** HRES for 2023 and later is not in WeatherBench2. ECMWF's open
  data for those years (0.4 deg, 9 levels, no omega) is too coarse to be worth
  building from.

```bash
python -m initial_conditions.hres_ic \
  --device cpu --timestamp 2022-09-26T12:00:00 \
  --tag 2022092612_ian_post_hres --out-dir "$WORK/ics" \
  --map "$WORK/map/map_721x1440_to_ne1024pg2.nc" --phis "$WORK/scream_phis.zarr" \
  --latlon "$AUX_DATA_ROOT/latlon_ne1024pg2.nc" --scream-ref "$WORK/scream_static" \
  --vert "$WORK/vert_coord_E3SM_L128.nc" --write-workers 32
```

The output is `scream_hiccup_ne1024pg2_<tag>.zarr` with the same layout as the
ERA5 IC. `ps`, `T_2m` and every 3D level are HRES; the statics and `coszr`
come from the same inputs as the ERA5 path.

The storm at t = 0 in the Ian ICs, from a TC tracker on the lowest model
level, against IBTrACS (1-min 10-m wind):

| Case | Source | Pmin (hPa) | Vmax (kt) | RMW (km) |
|---|---|---|---|---|
| Ian, 2022-09-26 12 UTC | IBTrACS | 981 | 70 | 28 |
| | ERA5 IC | 991 | 45 | 68 |
| | HRES IC | 987 | 47 | 48 |

## Environment

The fast path and `hres_ic` only need the STRATA Docker image. The HICCUP
reference path and the map build need:

- A conda environment with conda-forge `nco` (provides `ncremap`; the PyPI
  `nco` wrapper does not work) and `tempest-remap`, plus `xarray`, `dask`,
  `netcdf4`, `zarr>=3`, `cftime`, `scipy`, `psutil`. The validation ICs were
  built with nco 5.4.1, tempest-remap 2.2.0, xarray 2026.9.0, zarr 3.1.6 and
  Python 3.11.
- A HICCUP clone at commit `3dd04f3` with the calendar patch applied:

  ```bash
  git clone https://github.com/E3SM-Project/HICCUP.git && git -C HICCUP checkout 3dd04f3
  git -C HICCUP apply "$REPO/initial_conditions/hiccup_time_calendar.patch"
  ```

  Without the patch, `add_time_date_variables_multifile` fails on current
  xarray because it refuses to overwrite the `calendar` attribute that HICCUP
  sets on `time`. The patch moves that attribute into the write encoding.

- `rclone`, only for `run_tc_batch.py`'s uploads.

The scripts only import `numpy`, `xarray`, `zarr`, `netCDF4`, HICCUP and the
light `screamcast.history` / `screamcast.astronomy` modules, so they run in
the HICCUP environment without installing `screamcast`.

## Pipeline

`run_hiccup_eamxx.py` follows the HICCUP EAMxx multifile order:

1. Horizontal remap of every variable with the `fv2fv_flx` map
   (`remap_horizontal_multifile`), then `rename_vars_multifile` and
   `add_time_date_variables_multifile`. The variables are `T_mid`, `qv`, `U`,
   `V`, `ps`, `phis`, `omega` and `T_2m`. Cloud water, ice and ozone are dropped.
2. A check that the remapped ERA5 surface geopotential correlates with SCREAM
   `phis` (r >= 0.8). This catches flipped or shifted grids.
3. `surface_adjustment_multifile(adj_PS=True, adj_TS=False, adj_T_eam=False)`.
   This adjusts `ps` for `del_phis = PHIS_ERA5 - PHIS_SCREAM`. `phis` is then
   replaced by SCREAM's, bit for bit.
4. `remap_vertical_multifile` from the ERA5 pressure levels onto L128, using
   the adjusted `ps`. HICCUP interpolates linearly in log pressure. Outside
   the 1-1000 hPa source range it holds the end values constant. The
   exception is temperature below 1000 hPa, which follows a 6.5 K per 100 hPa
   lapse rate.
5. Optionally (`--save-qv-before-sat PATH`), `qv` at this point is kept as a
   separate NetCDF. Its history records the clip fraction.
6. `atmos_state_adjustment_multifile(adjust_sat=True, adjust_wtr=True)`.
   `remove_supersaturation` changes only `qv`: it clips `qv > qv_sat(T, p)` to
   saturation and sets `qv < 0` to 1e-9. Temperature is an input only.
7. Derived fields: `PotentialTemperature = T (P0/p)^(Rd/cp)` and hydrostatic
   `z_mid`. `z_mid` integrates the virtual temperature upward from `phis/g`
   over the hybrid interfaces, with Rd = 287.05, cp = 1004 and g = 9.80616.
8. `combine_files` into one float32 NetCDF with 3D fields stored as
   `(time, lev, ncol)`, followed by `clean_global_attributes`. The `history`
   attribute records the map command and the exact driver command.

`omega` is ERA5 `vertical_velocity`, already in Pa s-1. It is not derived from
w. It goes through the same horizontal and vertical remaps as the other 3D
fields. `T_2m` gets only the horizontal remap.

One timestamp takes about 2-3 h on one GB300 node. The L128 vertical remap
dominates, at roughly 15 min per 3D field with 8 dask workers
(`--vremap-workers`, `--vremap-chunk-mem-gb`, `--state-adj-workers`).

## Commands

Single IC (`--phis-zarr` defaults to `<data-root>/scream_phis.zarr`):

```bash
python -m initial_conditions.run_hiccup_eamxx \
  --timestamp 2016-08-01T00:00:00 \
  --map map/map_721x1440_to_ne1024pg2.nc \
  --dst-scrip "$AUX_DATA_ROOT/ne1024pg2_scrip.nc" \
  --data-root "$WORK" --hiccup-root "$WORK/HICCUP" \
  --out out/HICCUP.atm_era5.2016-08-01.00.ne1024pg2.L128.nc \
  --save-qv-before-sat out/HICCUP.atm_era5.2016-08-01.00.ne1024pg2.L128.qv_before_sat.nc

python -m initial_conditions.hiccup_l128_to_zarr \
  --ic out/HICCUP.atm_era5.2016-08-01.00.ne1024pg2.L128.nc \
  --out out/scream_hiccup_ne1024pg2_2016080100.zarr \
  --timestamp 2016-08-01T00:00:00 \
  --latlon "$AUX_DATA_ROOT/latlon_ne1024pg2.nc" --scream-ref "$WORK/scream_static" \
  --workers 16
```

`--save-qv-before-sat` hard-links HICCUP's temporary `qv` file, so the path
must be on the same filesystem as `--tmp-dir`.

Tropical-cyclone batch (skips tags whose Zarr is already on the remote and
deletes each local NetCDF after a verified upload; `--remote` is any rclone
destination prefix):

```bash
python -m initial_conditions.run_tc_batch \
  --work-root "$WORK" --hiccup-root "$WORK/HICCUP" \
  --map map/map_721x1440_to_ne1024pg2.nc \
  --dst-scrip "$AUX_DATA_ROOT/ne1024pg2_scrip.nc" \
  --latlon "$AUX_DATA_ROOT/latlon_ne1024pg2.nc" --scream-ref "$WORK/scream_static" \
  --remote REMOTE:PREFIX/era5_ic \
  --python "$CONDA_PREFIX/bin/python" [--tags 2017091400_maria_pre ...]
```

## Output layouts

The NetCDF IC is float32 with `ncol = 25165824` and `lev = 128`. It contains
`T_mid`, `qv`, `U`, `V`, `omega`, `PotentialTemperature` and `z_mid` as
`(time, lev, ncol)`; `ps`, `phis` and `T_2m` as `(time, ncol)`; and `hyam`,
`hybm`, `hyai`, `hybi` and `P0`.

The Zarr v3 IC has one frame and flat `ncol`. There is no ncol-to-face
reshape; that happens at rollout load time.

| Variables | Shape | Chunks |
|---|---|---|
| `PotentialTemperature`, `U`, `V`, `qv`, `omega`, `z_mid` | `(1, 32, ncol)`, levels `np.arange(128)[3::4]` | `(1, 1, ncol)` |
| `T_2m`, `ps`, `coszr` | `(1, ncol)` | `(1, ncol)` |
| `phis`, `lat`, `lon`, `sgh30`, `landfrac` | `(ncol,)` | `(ncol,)` |

Root attributes: `grid = ne1024pg2`, `initial_time`, `level_subset = 128[3::4]`,
`ic_source`, and `history` (the NetCDF history plus the converter command).
`coszr` comes from `screamcast.astronomy.calculate_cosine_zenith_direct` and is
0 on the night side.

**Consolidate after writing.** The converter calls
`zarr.consolidate_metadata(out)` after the last array write, because SCREAM
loaders open stores with `zarr.open_consolidated`. Any other workflow that
writes or patches one of these stores must do the same.
