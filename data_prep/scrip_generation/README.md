# SCRIP Grid File Generation for SCREAM ne1024pg2

This is a pure-Python reimplementation of the [TempestRemap](https://github.com/ClimateGlobalChange/tempestremap) C++ mesh generation workflow. It produces SCRIP-format NetCDF grid files for SCREAM's cubed-sphere grid and a halo-expanded variant used in CubeSphere inference.

## Background

SCREAM uses a cubed-sphere grid at ne1024pg2 resolution — 6 faces, each with 1024 spectral elements subdivided into a 2×2 GLL point layout, giving `6 × 2048 × 2048 = 25,165,824` cells. The SCRIP file encodes each cell's center coordinates, corner coordinates, and area, and is required by TempestRemap's offline regridding tools (e.g. to go from ne1024pg2 → lat-lon for visualization or verification).

The halo variant (`ne1024halo256pg2`) expands each face by 256 elements on each side (512 cells after pg2 subdivision), yielding `6 × 3072 × 3072` cells. This is used to pre-compute overlap weights that span face boundaries during CubeSphere inference.

## Files

- `generate_exodus_meshes.py` — builds the cubed-sphere grid geometry and saves it as intermediate mesh files in TempestRemap's exodus mesh format (`.g`), which store the grid's points and cells before conversion to the final SCRIP file
- `scrip_convert.py` — reads one of those intermediate `.g` files and creates the final SCRIP NetCDF file (`.nc`), including center and corner latitude/longitude, cell area, and mask values
- `derive_latlon.py` — extracts per-column cell-center lat/lon (float32 degrees over `ncol`) from the no-halo SCRIP file, producing the `latlon_ne1024pg2.nc` auxiliary file the training configs read (verified bit-identical to the file the shipped models were trained with)
- `generate_all.sh` — runs the full workflow end-to-end to produce all three target files automatically

## Dependencies

```bash
pip install -r data_prep/scrip_generation/requirements.txt
```

## Usage

From the repo root, run both grids end-to-end:

```bash
bash data_prep/scrip_generation/generate_all.sh --output-dir data
```

This writes `data/ne1024pg2_scrip.nc`, `data/latlon_ne1024pg2.nc`, and
`data/ne1024halo256pg2_scrip.nc`.

To run each step manually:

```bash
# no-halo grid
python data_prep/scrip_generation/generate_exodus_meshes.py --res 1024 --np 2 --halo 0 --out-dir data
python data_prep/scrip_generation/scrip_convert.py --in-exodus data/ne1024pg2.g

# halo grid
python data_prep/scrip_generation/generate_exodus_meshes.py --res 1024 --np 2 --halo 256 --out-dir data
python data_prep/scrip_generation/scrip_convert.py --in-exodus data/ne1024halo256pg2.g
```

## Validation

The reference `data/ne1024pg2_scrip.nc` was originally produced by the upstream TempestRemap C++ package. This reimplementation reproduces it to floating-point accuracy: grid areas, cell centers, and corner coordinates are bit-for-bit identical or differ at the < 1e-11° level. The only discrepancy is in `grid_corner_lon`: 8190 out of 25,165,824 × 4 corner values are stored as 0° vs 360° — equivalent representations of the same point on the prime meridian.

## Notes

- Large meshes (ne1024) are memory-intensive. If you run out of memory during SCRIP conversion, lower `--chunk-elems` in `scrip_convert.py` (default: 200,000 elements per chunk).
- When `--halo 0`, filenames omit the `halo0` suffix to match TempestRemap's naming convention.
