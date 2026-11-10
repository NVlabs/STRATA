# STRATA initial conditions from ERA5 and ECMWF HRES

`screamcast/initial_conditions/` builds STRATA initial conditions on the SCREAM
**ne1024pg2** physics grid (`ncol = 25165824`) with the **E3SM L128** hybrid
vertical grid, from ERA5 or the ECMWF HRES analysis. Each IC is a one-frame
Zarr v3 store with flat `ncol` and 32 of the L128 levels (see
[Output layouts](#output-layouts)). These are STRATA inputs on the grid STRATA
is trained on, not initial conditions EAMxx can start from: EAMxx reads its ICs
on the np4 GLL grid.

The computation is a torch port of the E3SM
[HICCUP](https://github.com/E3SM-Project/HICCUP) steps (see
[License](#license)). It builds an IC for any ERA5 hour in about 80 s on one
GPU, and further ICs in the same run take about 25 s each. Its output matches
HICCUP's own bit for bit apart from last-bit differences in a few `qv` values
(see [Validation against HICCUP](#validation-against-hiccup)).

| Module | Role |
|---|---|
| `fast_ic.py` | The HICCUP steps on the GPU; ERA5 timestamp(s) to Zarr IC(s) (see [ERA5 initial conditions](#era5-initial-conditions)) |
| `hres_ic.py` | The same steps on the ECMWF HRES analysis (see [HRES initial conditions](#hres-initial-conditions)) |
| `build_map.py` | Builds (in parallel) or reuses the ERA5 721x1440 to ne1024pg2 `fv2fv_flx --lrg2sml` map |
| `ic_store.py` | lat/lon and static inputs, `coszr`, creating and checking a one-frame store |
| `layout.py` | Grid and level constants, Zarr variable lists, the store name |

Run them with `python -m screamcast.initial_conditions.<name>` once the package
is installed (see the README's Setup), or from the repository root.

## Inputs

| Input | Notes |
|---|---|
| ERA5 | ARCO `gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3`, read over HTTPS. **37 pressure levels** (1-1000 hPa), 0.25 deg 721x1440, north-to-south, starting at Greenwich. Time index is hours since 1900-01-01. |
| SCREAM `phis`, `sgh30`, `landfrac` | The `phis`, `sgh30` and `landfrac` arrays of the [nvidia/STRATA-SCREAM-sdy1](https://huggingface.co/datasets/nvidia/STRATA-SCREAM-sdy1) dataset (about 60 MB together; see below). |
| Map | `map_721x1440_to_ne1024pg2.nc` (3.2 GB), `ncremap -a fv2fv_flx --lrg2sml` from the ERA5 grid to ne1024pg2, with a `.command.txt` sidecar next to it. Build it once with `build_map.py`, about 30 min on 72 cores (see [Map](#map)). `fast_ic` and `hres_ic` only read it. |
| ne1024pg2 SCRIP | `ne1024pg2_scrip.nc`, the destination grid for the map (`grid_size = 25165824`) |
| lat/lon | `latlon_ne1024pg2.nc`, NetCDF with `lat`/`lon` on `ncol`. They are written to the Zarr as stored, and `coszr` is computed from them in degrees (converted first when the `lat` units start with `rad`; the scripts stop if the result is not a latitude). `generate_all.sh` writes degrees. |
| L128 coordinate | HICCUP `files_vert/vert_coord_E3SM_L128.nc` (8 KB; see below) |

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

The L128 coordinate file comes from the HICCUP repository (no clone needed):

```bash
curl -L -o "$WORK/vert_coord_E3SM_L128.nc" \
  https://raw.githubusercontent.com/E3SM-Project/HICCUP/0dec8bba33600625bd6109990bcb63e7930bb264/files_vert/vert_coord_E3SM_L128.nc
```

### Map

`build_map.py` builds the map in parallel. It needs conda-forge `nco` and
`tempest-remap` (see [Environment](#environment)), and writes the ERA5 SCRIP
file with HICCUP's `ncremap -G` grid description if `--src-scrip` does not
exist:

```bash
python -m screamcast.initial_conditions.build_map \
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
times longer than the others and finish last. The default ranges have about
175k columns; a range of half that size used 0.8 GB.

Against the serial ncremap map built on an x86 machine (NCO 5.3.6), the
parallel map has the same `row` and `col` (38,299,240 links in the same
order). 31,590 weights (0.08%) differ after rounding to float32; the largest
relative difference is 1.3e-5, on sliver overlaps. A smooth temperature-like
field remapped with the two maps differs by at most 1.8e-5 K. Two different
splits of the same columns gave weights within 7e-9 of each other and the same
float32 weights, so the differences come from the build machine, not from the
split. Each part sees only its own overlaps, so the source-side `area_a`
(summed overlap area) and `frac_a` are recomputed from all links after the
merge; they match the serial map to 2e-11 and 1e-15.

The parts are built in `<map name>.parts` next to the map (or under
`--tmp-dir`), which is removed after the merge. A rerun reuses the parts that
finished, as long as the grids are the same.

An existing map is reused when its `.command.txt` sidecar is next to it
(`build_map.py` writes one), and a map without a sidecar is adopted after its
`n_a`/`n_b` sizes are checked.

## ERA5 initial conditions

`fast_ic.py` takes one or more UTC timestamps, reads those hours from the ARCO
ERA5 store and writes one Zarr IC per timestamp to
`<out-dir>/scream_hiccup_ne1024pg2_<tag>.zarr`. The tag defaults to
`YYYYMMDDHH`. It needs a CUDA GPU with about 50 GB of memory and the STRATA
Docker image (torch, zarr 3, netCDF4 and gcsfs). It does not need NCO,
TempestRemap or HICCUP; it reads the map, static arrays and L128 coordinate
file from [Inputs](#inputs).

```bash
docker run --rm --gpus all --ipc=host --user "$(id -u):$(id -g)" \
  -v "$PWD:$PWD" -v "$WORK:$WORK" -w "$PWD" -e PYTHONPATH="$PWD" \
  strata \
  python -m screamcast.initial_conditions.fast_ic \
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
- `--vert` should be HICCUP's float64 `vert_coord_E3SM_L128.nc` (downloaded
  in [Inputs](#inputs)). The repository's
  `data/scream_vertical_coordinate.nc` is a float32 copy of the same
  coordinate; it also works but shifts results by float32 rounding.
- The stores are consolidated when they are written.

### How it works

The computation follows HICCUP's EAMxx steps one by one, with the
float32/float64 types and operation order that HICCUP gets from numpy 2:

1. `remap_horizontal`: the ncremap map as a CSR matrix on the GPU, applied to
   all 188 ERA5 level-fields in one sparse matmul, accumulated in float64 and
   stored as float32 (as ncremap does). The fields are `T`, `qv`, `U`, `V`,
   `omega`, `ps`, `phis` and `T_2m`; cloud water, ice and ozone are not used.
   `omega` is ERA5 `vertical_velocity`, already in Pa s-1, not derived from w.
2. `adjust_surface_pressure`: HICCUP's formula, using T on the 1000 hPa level
   and `del_phis = PHIS_ERA5 - PHIS_SCREAM`. `phis` is then SCREAM's, bit for
   bit.
3. Bottom-up loop over the 128 target levels across all columns:
   `log_pressure_stencil` and `interp_level` reproduce HICCUP's
   `_interp_block`: linear in log pressure, with the end values held outside
   the 1-1000 hPa source range, except temperature below 1000 hPa, which
   follows a 6.5 K per 100 hPa lapse rate. The bracketing indices and weights
   depend only on pressure, so they are computed once per level and shared by
   T, qv, U, V and omega. Then `remove_supersaturation` clips `qv` above
   saturation over liquid (Bolton) to saturation and sets negative `qv` to
   1e-9; temperature is an input only. Last come hydrostatic `z_mid`, which
   integrates the virtual temperature upward from `phis/g` over the hybrid
   interfaces, and `PotentialTemperature = T (P0/p)^(Rd/cp)`, with
   Rd = 287.05, cp = 1004 and g = 9.80616. On the 46 pure pressure levels
   (above 146 hPa) the pressure is the same in every column, so those float64
   logs and powers are evaluated once. `T_2m` gets only the horizontal remap.
4. Each output level goes to a pool of writer threads as soon as it is final.
   They zstd-encode the chunk with the codec from the store's own metadata and
   write it to the key its chunk key encoding gives. zarr's own writer encodes
   these 100 MB chunks one at a time, which takes minutes. Each store is
   consolidated, and one chunk per array is read back through zarr.

In a batch, the map, statics and imports are loaded once, the next hour is
downloaded while the current one is computed, and each store finishes writing
while the next IC runs on the GPU.

### Validation against HICCUP

`fast_ic.py` rebuilt ten tropical-cyclone ICs from ARCO (Maria 2017-09-14T00
and 09-18T12, Michael 2018-10-06T00 and 10-09T12, Dorian 2019-08-26T00 and
09-01T00, Ian 2022-09-22T00 and 09-26T12, Otis 2023-10-21T00 and 10-24T06).
Each was compared value by value with an IC built by running HICCUP itself
(commit 3dd04f3, its EAMxx NetCDF multifile workflow, with the same map, L128
file and SCREAM `phis`). Across the ten:

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

Running HICCUP itself took 2.5-3 h per IC on the same machine: 81 min of that
is the vertical remap, and 27 min goes to NetCDF combine and attribute
cleanup. Writing is CPU-bound: the 192 3D chunks are about 1,000 CPU-seconds
of zstd at the stores' compression level.

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

Run it in the STRATA Docker image as for `fast_ic` (it imports torch):

```bash
python -m screamcast.initial_conditions.hres_ic \
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

`fast_ic` and `hres_ic` only need the STRATA Docker image. `build_map` needs a
conda environment with conda-forge `nco` (provides `ncremap`; the PyPI `nco`
wrapper does not work) and `tempest-remap`, plus `numpy` and `netcdf4`. The
validation ICs were built in an environment with nco 5.4.1 and tempest-remap
2.2.0.

## Output layouts

The Zarr v3 IC has one frame and flat `ncol`. There is no ncol-to-face
reshape; that happens at rollout load time. `run_screamcast_nudged.py
--ic-store <store> --correction none` starts a forecast from it at the
store's `initial_time` (see the README).

| Variables | Shape | Chunks |
|---|---|---|
| `PotentialTemperature`, `U`, `V`, `qv`, `omega`, `z_mid` | `(1, 32, ncol)`, levels `np.arange(128)[3::4]` | `(1, 1, ncol)` |
| `T_2m`, `ps`, `coszr` | `(1, ncol)` | `(1, ncol)` |
| `phis`, `lat`, `lon`, `sgh30`, `landfrac` | `(ncol,)` | `(ncol,)` |

Root attributes: `grid = ne1024pg2`, `initial_time`, `level_subset = 128[3::4]`,
`ic_source`, and `history` (the command that wrote the store).
`coszr` comes from `screamcast.astronomy.calculate_cosine_zenith_direct` and is
0 on the night side.

**Consolidate after writing.** The writers call
`zarr.consolidate_metadata(out)` after the last chunk, because SCREAM loaders
open stores with `zarr.open_consolidated`. Any other workflow that writes or
patches one of these stores must do the same.

## License

`fast_ic.py` contains code ported from HICCUP, Copyright 2026, Lawrence
Livermore National Security, LLC, released under the 3-Clause BSD license.
The full HICCUP notice is kept at the top of `fast_ic.py`, as that license
requires. The L128 coordinate file downloaded above is part of HICCUP and is
covered by the same license.
