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
        description="Plot regional-average precipitation time series in separate subplots."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a *_regional_averages.nc file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output PNG path.",
    )
    parser.add_argument(
        "--variable",
        default="precip_liq_surf_mass_flux",
        help="Variable to plot from the regional averages file.",
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=None,
        help="Optional list of regions to plot. Defaults to all regions in the file.",
    )
    parser.add_argument(
        "--time-units",
        choices=("hours", "days"),
        default="days",
        help="Units for the lead-time axis.",
    )
    parser.add_argument(
        "--native-units",
        action="store_true",
        help="Plot the variable in native file units instead of converting to mm/day.",
    )
    parser.add_argument(
        "--ncols",
        type=int,
        default=2,
        help="Number of subplot columns.",
    )
    return parser.parse_args()


def _lead_time_axis(
    step_seconds: xr.DataArray, time_units: str
) -> tuple[xr.DataArray, str]:
    if time_units == "hours":
        return step_seconds / 3600.0, "Lead time (hours)"
    return step_seconds / 86400.0, "Lead time (days)"


def _select_regions(ds: xr.Dataset, requested_regions: list[str] | None) -> list[str]:
    available_regions = [str(region) for region in ds["region"].values.tolist()]
    if requested_regions is None:
        return available_regions
    missing = sorted(set(requested_regions) - set(available_regions))
    if missing:
        raise KeyError(
            f"Requested region(s) not found: {', '.join(missing)}. "
            f"Available regions: {', '.join(available_regions)}"
        )
    return requested_regions


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    with xr.open_dataset(input_path, engine="h5netcdf") as ds:
        if args.variable not in ds:
            available = ", ".join(sorted(ds.data_vars))
            raise KeyError(
                f"{args.variable!r} not found. Available variables: {available}"
            )

        field = ds[args.variable]
        if "time" in field.dims:
            field = field.isel(time=0, drop=True)
        if "level" in field.dims:
            raise ValueError(
                f"{args.variable!r} has a level dimension. Select a 2-D variable such as "
                "'precip_liq_surf_mass_flux', or extend this script to pick a level."
            )

        regions = _select_regions(ds, args.regions)
        y = field
        ylabel = args.variable
        if not args.native_units:
            y = y * PRECIP_TO_MM_DAY
            ylabel = f"{args.variable} (mm/day)"

        x, xlabel = _lead_time_axis(ds["step"], args.time_units)
        ncols = max(1, args.ncols)
        nrows = math.ceil(len(regions) / ncols)
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(6 * ncols, 3.5 * nrows),
            dpi=150,
            sharex=True,
            sharey=True,
            squeeze=False,
        )

        for ax, region in zip(axes.flat, regions, strict=False):
            region_series = y.sel(region=region)
            ax.plot(x.values, region_series.values, linewidth=2)
            ax.set_title(region)
            ax.grid(True, alpha=0.3)

        for ax in axes.flat[len(regions) :]:
            ax.set_visible(False)

        for ax in axes[-1, :]:
            if ax.get_visible():
                ax.set_xlabel(xlabel)
        for ax in axes[:, 0]:
            if ax.get_visible():
                ax.set_ylabel(ylabel)

        fig.suptitle(f"Regional mean {args.variable}", y=0.995)
        fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")


if __name__ == "__main__":
    main()
