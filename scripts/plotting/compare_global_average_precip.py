#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PRECIP_TO_MM_DAY = 1000.0 * 86400.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare total precipitation from precomputed regional-average "
            "NetCDF outputs."
        )
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Path to a *_regional_averages.nc file. Pass once per series.",
    )
    parser.add_argument(
        "--label",
        action="append",
        required=True,
        help="Legend label for each input. Must match --input count.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output PNG path.",
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=("global",),
        help=(
            "Region names to plot. Use a single region for a one-panel plot or "
            "multiple names for a subplot grid."
        ),
    )
    parser.add_argument(
        "--ncols",
        type=int,
        default=2,
        help="Number of subplot columns when plotting multiple regions.",
    )
    return parser.parse_args()


def _total_precip_mm_day(ds: xr.Dataset, region: str) -> xr.DataArray:
    liq = ds["precip_liq_surf_mass_flux"].sel(region=region)
    ice = ds["precip_ice_surf_mass_flux"].sel(region=region)
    if "time" in liq.dims:
        liq = liq.isel(time=0, drop=True)
    if "time" in ice.dims:
        ice = ice.isel(time=0, drop=True)
    return (liq + ice) * PRECIP_TO_MM_DAY


def _validate_regions(ds: xr.Dataset, requested: list[str]) -> list[str]:
    available = [str(region) for region in ds["region"].values.tolist()]
    missing = sorted(set(requested) - set(available))
    if missing:
        raise KeyError(
            f"Requested region(s) not found: {', '.join(missing)}. "
            f"Available regions: {', '.join(available)}"
        )
    return requested


def main() -> None:
    args = parse_args()
    if len(args.input) != len(args.label):
        raise ValueError(
            f"Expected the same number of --input and --label values, got "
            f"{len(args.input)} and {len(args.label)}"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    regions = list(args.regions)

    first_ds = xr.open_dataset(Path(args.input[0]), engine="h5netcdf")
    try:
        regions = _validate_regions(first_ds, regions)
    finally:
        first_ds.close()

    nregions = len(regions)
    if nregions == 1:
        fig, axes = plt.subplots(figsize=(8, 4.5), dpi=160)
        axes_list = [axes]
    else:
        ncols = max(1, min(args.ncols, nregions))
        nrows = math.ceil(nregions / ncols)
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(6 * ncols, 3.8 * nrows),
            dpi=160,
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        axes_list = list(axes.flat)

    for label, input_path_str in zip(args.label, args.input, strict=True):
        input_path = Path(input_path_str)
        with xr.open_dataset(input_path, engine="h5netcdf") as ds:
            lead_days = ds["step"] / 86400.0
            for ax, region in zip(axes_list, regions, strict=False):
                precip = _total_precip_mm_day(ds, region=region)
                ax.plot(lead_days.values, precip.values, linewidth=2, label=label)
                print(
                    f"{label} {region}: start={float(precip.isel(step=0)):.3f} mm/day "
                    f"end={float(precip.isel(step=-1)):.3f} mm/day "
                    f"mean={float(precip.mean()):.3f} mm/day"
                )

    for ax, region in zip(axes_list, regions, strict=False):
        title = region if nregions > 1 else f"{region} mean precipitation"
        ax.set_title(title)
        ax.set_xlabel("Lead time (days)")
        ax.set_ylabel("Total precip (mm/day)")
        ax.set_ylim(bottom=0.0)
        ax.grid(True, alpha=0.3)
        ax.legend()

    for ax in axes_list[nregions:]:
        ax.set_visible(False)

    if nregions == 1:
        fig.suptitle("Precipitation comparison", y=0.98)
    else:
        fig.suptitle("Precipitation comparison by region", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    print(output_path)


if __name__ == "__main__":
    main()
