# Regional window rollouts

STRATA can run on a window of the globe instead of the whole sphere. Only the tiles
inside the window are predicted. The halo that the tiles at the window's edge need is
read from a one-tile *ring* around the window, and the ring is filled from outside
information every 10-minute step. Inside the window STRATA runs freely: nothing nudges
it.

A window of 8 x 8 tiles of 128 px (1,024 x 1,024 cells, about 5,000 km across) has 64
tiles instead of the globe's 1,536. On one node of 8 H100 GPUs it runs 24 hours in
about 8 minutes, and a 4 x 4-tile window (about 2,500 km) in about 4 minutes (see
[Resources](#resources)).

## Choose a window

`--region face,i0,i1,j0,j1` is a rectangle of tiles (half-open ranges, in units of
`--tile-size`) on the *plane* of cube face `face`:

- one face; or
- an equatorial face (0-3) joined to the next equatorial face to the east (`j1` beyond
  the face's tiles), or to the north polar face 5 (`i1` beyond the face's tiles).

The ring must stay on the same plane, so `i0, j0 >= 1` and the window ends at least one
tile before the plane's far edge. A window inside one equatorial face that is flush with
its north, east or west edge (for example `0,12,16,10,14` with 16 tiles per face side)
runs on the plane that joins the neighbour holding its ring: the north polar face, the
east neighbour, or the west neighbour (as the base of an east join). It can be written
in either frame, `0,2,6,0,4` or `3,2,6,16,20`. `screamcast.regional` finds a window around a point:

```bash
python -m screamcast.regional --lat 12 --lon 0 --tiles 8 \
  --latlon "$AUX_DATA_ROOT/latlon_ne1024pg2.nc"
```

```text
0,7,15,4,12
# 8 x 8 tiles, about 5018 km across; corners:
#   lat   -5.18  lon  337.52
#   lat   -5.18  lon   22.48
#   lat   37.15  lon  337.52
#   lat   37.15  lon   22.48
```

It tries the point's own face first, then the planes that join it to a neighbour, and
moves the window as little as needed to keep its ring on the plane.

### A window from a lat/lon box

`--region-latlon LAT0,LAT1,LON0,LON1` lets the runner choose the window. The box runs
east from `LON0` to `LON1`, so `170,-170` crosses the dateline. The window holds every
grid cell of the box with at least `--region-margin-tiles` whole tiles (default 1)
between them and the driven ring, keeps its ring on one plane, and has a multiple of
the world size of tiles; among those it has the fewest tiles, then the largest margin,
then the most centred box. The run logs the chosen `--region`, and the output records
both (`region`, `region_latlon`), so `--region` reproduces it. To see the window first:

```bash
python -m screamcast.regional --latlon-box 8,18,-91,-79 --world-size 8 \
  --latlon "$AUX_DATA_ROOT/latlon_ne1024pg2.nc"
```

```text
3,8,13,4,12
# 5 x 8 tiles (40), 1-tile margin, world size 8; corners:
#   lat    0.02  lon  247.52
#   lat    0.02  lon  292.48
#   lat   26.26  lon  247.52
#   lat   26.26  lon  292.48
```

A box that needs a ring across an equatorial face's south edge or a cube corner, or is
too large for one plane, is refused with the reason.

## The boundary

`--region-boundary anchored` (the default) fills the ring with the initial state plus
the large-scale change of a *driver* since the start:

    ring(t) = IC + (smooth(D(t)) - smooth(D(t0)))

- `D` is a sequence of states on the SCREAM grid at known valid times, given as
  one-frame stores with `--boundary-stores` (the same layout as `--ic-store`). Between
  two driver times the ring is linear in time.
- `smooth` is a block mean over `--boundary-block` x `--boundary-block` cells (default 8,
  about 40 km), upsampled back bilinearly. Only the driver's large-scale evolution
  enters the ring.
- At `t0` the ring equals the initial state, so the window starts without a jump at its
  edge. Fine structure locked to the terrain (surface pressure over mountains, 2 m
  temperature) is kept from the initial state; the driver's own version of it cancels
  in the difference.
- If no store is at the initial time, `D(t0)` is the initial condition itself.
- Without `--boundary-stores` and without `--ic-store`, the driver is the SCREAM run in
  `SCREAM_MAIN_ZARR_PATH` itself, every `--boundary-every-hours` (default 6). That is a
  test of the window against a known truth.

`--region-boundary replicate` extends the window's own edge pixels outward instead. No
outside information enters, so the edge drifts; use it as a baseline.

The ring needs every input channel in one halo call, so `--region` sets
`--halo-channel-chunk 0`. The omega spherical-harmonic high-pass (`--sht-omega-lmax`) is
a global operator and is switched off. `--pressure-levels` output is not supported yet.
The output store keeps the global layout; cells outside the window are NaN. Regional
averages are off by default with `--region`: a window often misses a whole region (a
window equatorward of `--tropics-lat-deg`, default 23.5, has no extratropics), and the averager refuses an
empty region at start-up. `--regional-averages` turns them on for a window that covers
every region; they are then averages over the window.

## Drive a window with an ECMWF HRES forecast

`screamcast.initial_conditions.hres_forecast` builds one store per lead of an HRES
forecast from ECMWF open data (2024-02 onward), with the same steps as `fast_ic` and
`hres_ic` (see [initial_conditions.md](initial_conditions.md) for the map, statics and
L128 file). The lead-0 store is the HRES analysis and serves as the initial condition:

```bash
python -m screamcast.initial_conditions.hres_forecast \
  --init 2025-08-15T12 --leads 0 6 12 18 24 --tag-prefix erin \
  --out-dir "$WORK/ics" --cache "$WORK/grib" \
  --map "$WORK/map/map_721x1440_to_ne1024pg2.nc" --phis "$WORK/scream_phis.zarr" \
  --latlon "$AUX_DATA_ROOT/latlon_ne1024pg2.nc" --scream-ref "$WORK/scream_static" \
  --vert "$WORK/vert_coord_E3SM_L128.nc"
```

If the AWS bucket answers HTTP 503 (`SlowDown`), read the same files from ECMWF's
Google Cloud mirror with `--opendata-root https://storage.googleapis.com/ecmwf-open-data`.

Then run the window. `--boundary-stores` takes the stores in any order; their
`initial_time` attributes are the valid times:

```bash
torchrun --nproc_per_node=8 scripts/ace/run_screamcast_nudged.py \
  --run-name erin_window \
  --checkpoint $AUX_DATA_ROOT/checkpoints/strata_v1.pth \
  --ic-store "$WORK/ics/scream_hiccup_ne1024pg2_erin_f000.zarr" \
  --region "$(python -m screamcast.regional --lat 25 --lon -70 --tiles 8 \
      --latlon $AUX_DATA_ROOT/latlon_ne1024pg2.nc | head -1)" \
  --boundary-stores "$WORK"/ics/scream_hiccup_ne1024pg2_erin_f0*.zarr \
  --forecast-model screamcast --correction none --n-steps 144 \
  --tile-size 128 --halo-width 16 --halo-adjoint --no-halo-adjoint-cross-face \
  --tile-jitter 4 --tile-jitter-mode cycle16 --inference-batch-size 1 \
  --output-variables T_2m ps U V qv omega precip_liq_surf_mass_flux \
  --output-levels 13,18,24,25,29,30 \
  $OUTPUT
```

The drivers must cover the run: the last step reads the ring at `(n_steps - 1) x 10`
minutes. Any other source works the same way: ERA5 hours from `fast_ic`, or HRES
analyses from `hres_ic` for a hindcast.

## Resources

Measured on one node of 8 NVIDIA H100 80 GB GPUs (2026-11-17), Hurricane Helene with
the HRES forecast of 2024-09-25 12 UTC as the driver (5 stores, +0 to +24 h). The
8 x 8-tile window is `3,9,17,5,13` (it reaches one tile row into the north polar face);
the 4 x 4-tile window is `3,10,14,7,11`, all on face 3. Building the 5 HRES stores
(`hres_forecast`, one GPU, downloads included) took 134 s.

| Step | 8 x 8 tiles (64) | 4 x 4 tiles (16) |
|---|---|---|
| Start-up: container, model, halo plan, initial condition and 5 drivers | about 3 min | about 2 min |
| 24 h (144 steps), eager (`TORCH_COMPILE_DISABLE=1`) | 6 min 45 s (2.8 s per step) | not measured |
| 24 h (144 steps), `torch.compile` | 4 min 28 s (1.9 s per step) | 73 s (0.5 s per step) |
| Compile | 32 s | 30 s |
| Whole run, `torch.compile` | 8 min 13 s | 4 min |

These windows also fit in one GPU's memory, but then all their tiles run on it: expect
several times the per-step times above.

Each driver time is loaded once at the start, as the initial condition is (the full
globe, then cut to the window and its ring), so start-up grows with the number of
drivers.

## How it works

- `TileTopology(active_tiles=...)` predicts only the window's tiles and spreads them over
  the ranks. `ring_tiles` is the one-tile ring around them.
- The halo exchange is the global one with its KNN source indices remapped onto the
  window + ring tiles, so with the true state in the ring every padded window tile is
  the global run's, bit for bit (`tests/test_regional.py`).
- The halo adjoint drops the contributions of predictions made over the ring, which is
  never predicted; points next to the window edge are normalised by the window's own
  weights, so a constant field still round-trips.
- With the global run's own states as the driver (no smoothing), the window reproduces
  the global run (`test_a_perfect_boundary_reproduces_the_global_run`).

## Limitations

- A window that runs freely inside has been checked for stability and for continuity
  at its edge on a few cases, not for forecast skill.
- At the window edge the large scale is the driver's, smoothed; inside it is STRATA's
  own. Expect a narrow seam where the two meet.
- A window cannot cross the south edge of an equatorial face, or join three faces. Its
  ring cannot either: a window flush with an equatorial face's south edge, with a polar
  face's edge, or with two edges of a face (a cube corner) is refused.
