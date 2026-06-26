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
from pathlib import Path

import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REFERENCE_PRESSURE_HPA = 1000.0
FIXED_COLOR_LIMITS: dict[str, tuple[float, float, str]] = {
    "qv": (-5.0e-4, 5.0e-4, "RdBu_r"),
    "omega": (-0.04, 0.04, "RdBu_r"),
    "U": (-5.0, 5.0, "RdBu_r"),
    "PotentialTemperature": (-2.0, 2.0, "RdBu_r"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a level-dependent regional-average variable with contour "
            "plots, using lead time on x and level on y."
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
        "--variable",
        required=True,
        help="Variable to plot, e.g. qv or omega.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write per-region PNGs into.",
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        required=True,
        help="Regions to render.",
    )
    parser.add_argument(
        "--max-lead-hours",
        type=float,
        default=24.0,
        help="Only plot lead times up to this many hours.",
    )
    return parser.parse_args()


def _validate_regions(ds: xr.Dataset, requested: list[str]) -> list[str]:
    available = [str(region) for region in ds["region"].values.tolist()]
    missing = sorted(set(requested) - set(available))
    if missing:
        raise KeyError(
            f"Requested region(s) not found: {', '.join(missing)}. "
            f"Available regions: {', '.join(available)}"
        )
    return requested


def _load_field(ds: xr.Dataset, variable: str, region: str) -> xr.DataArray:
    if variable not in ds:
        available = ", ".join(sorted(ds.data_vars))
        raise KeyError(f"{variable!r} not found. Available variables: {available}")
    field = ds[variable].sel(region=region)
    if "time" in field.dims:
        field = field.isel(time=0, drop=True)
    if "level" not in field.dims:
        raise ValueError(f"{variable!r} has no level dimension.")
    return field.transpose("level", "step")


def _level_axis(ds: xr.Dataset, field: xr.DataArray) -> np.ndarray:
    if "hyam" in ds and "hybm" in ds:
        hyam = np.asarray(ds["hyam"].values, dtype=float)
        hybm = np.asarray(ds["hybm"].values, dtype=float)
        if hyam.shape == hybm.shape == (field.sizes["level"],):
            return REFERENCE_PRESSURE_HPA * (hyam + hybm)
    level_values = np.asarray(field["level"].values)
    if np.issubdtype(level_values.dtype, np.number):
        return level_values.astype(float)
    return np.arange(field.sizes["level"], dtype=float)


def _lead_time_hours(ds: xr.Dataset) -> np.ndarray:
    step = np.asarray(ds["step"].values)
    if np.issubdtype(step.dtype, np.timedelta64):
        return step / np.timedelta64(1, "h")
    return step.astype(float) / 3600.0


def _color_limits(arrays: list[np.ndarray]) -> tuple[float, float, str]:
    flat = np.concatenate([a[np.isfinite(a)] for a in arrays])
    if flat.size == 0:
        return 0.0, 1.0, "viridis"
    if np.nanmin(flat) < 0.0 and np.nanmax(flat) > 0.0:
        vmax = float(np.nanpercentile(np.abs(flat), 99))
        if vmax == 0.0:
            vmax = float(np.nanmax(np.abs(flat))) or 1.0
        return -vmax, vmax, "RdBu_r"
    vmin = float(np.nanpercentile(flat, 1))
    vmax = float(np.nanpercentile(flat, 99))
    if vmin == vmax:
        vmax = vmin + 1.0
    return vmin, vmax, "viridis"


def _plot_limits(variable: str, arrays: list[np.ndarray]) -> tuple[float, float, str]:
    fixed = FIXED_COLOR_LIMITS.get(variable)
    if fixed is not None:
        return fixed
    return _color_limits(arrays)


def _contour_levels(vmin: float, vmax: float) -> np.ndarray:
    return np.linspace(vmin, vmax, 21, dtype=float)


def _profile_levels(vmin: float, vmax: float) -> np.ndarray:
    return np.linspace(vmin, vmax, 9, dtype=float)


def _variable_label(ds: xr.Dataset, variable: str) -> str:
    units = ds[variable].attrs.get("units")
    if units:
        return f"{variable} ({units})"
    return variable


def _interp_profile(
    truth_profile: np.ndarray,
    truth_level_axis: np.ndarray,
    target_level_axis: np.ndarray,
) -> np.ndarray:
    order = np.argsort(truth_level_axis)
    return np.interp(target_level_axis, truth_level_axis[order], truth_profile[order])


def _slugify(text: str) -> str:
    chars = []
    for ch in text.lower():
        if ch.isalnum():
            chars.append(ch)
        else:
            chars.append("_")
    slug = "".join(chars)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def main() -> None:
    args = parse_args()
    if len(args.input) != len(args.label):
        raise ValueError(
            f"Expected the same number of --input and --label values, got "
            f"{len(args.input)} and {len(args.label)}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets: list[tuple[str, Path]] = [
        (label, Path(path_str))
        for label, path_str in zip(args.label, args.input, strict=True)
    ]

    with xr.open_dataset(datasets[0][1], engine="h5netcdf") as first_ds:
        regions = _validate_regions(first_ds, list(args.regions))
        truth_variable_label = _variable_label(first_ds, args.variable)

    loaded_by_region: dict[
        str, list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]
    ] = {}
    all_values: list[np.ndarray] = []
    truth_profiles: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for region in regions:
        loaded: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
        for label, path in datasets:
            with xr.open_dataset(path, engine="h5netcdf") as ds:
                field = _load_field(ds, args.variable, region)
                lead_hours = _lead_time_hours(ds)
                mask = lead_hours <= args.max_lead_hours + 1e-9
                if not np.any(mask):
                    raise ValueError(
                        f"No lead times <= {args.max_lead_hours} hours found in {path}"
                    )
                values = field.values[:, mask]
                level_axis = _level_axis(ds, field)
                if path == datasets[0][1]:
                    truth_level_axis = level_axis.copy()
                    truth_profile = values[:, 0].copy()
                    truth_profiles[region] = (truth_level_axis, truth_profile)
                else:
                    truth_level_axis, truth_profile = truth_profiles[region]
                anomaly = (
                    values
                    - _interp_profile(truth_profile, truth_level_axis, level_axis)[
                        :, np.newaxis
                    ]
                )
                loaded.append((label, lead_hours[mask], level_axis, anomaly))
                all_values.append(anomaly)
        loaded_by_region[region] = loaded

    vmin, vmax, cmap = _plot_limits(args.variable, all_values)

    contour_levels = _contour_levels(vmin, vmax)
    colorbar_ticks = np.linspace(vmin, vmax, 7, dtype=float)
    all_loaded = [
        item for region_loaded in loaded_by_region.values() for item in region_loaded
    ]
    common_ymin = max(float(level_axis.min()) for _, _, level_axis, _ in all_loaded)
    common_ymax = min(float(level_axis.max()) for _, _, level_axis, _ in all_loaded)
    if not common_ymin < common_ymax:
        common_ymin = min(float(level_axis.min()) for _, _, level_axis, _ in all_loaded)
        common_ymax = max(float(level_axis.max()) for _, _, level_axis, _ in all_loaded)

    for region in regions:
        loaded = loaded_by_region[region]
        truth_level_axis, truth_profile = truth_profiles[region]
        fig, ax = plt.subplots(
            figsize=(4.2, 4.8),
            dpi=160,
            constrained_layout=False,
        )
        ax.plot(truth_profile, truth_level_axis, linewidth=2, color="black")
        ax.set_title("truth at t=0", pad=10)
        ax.set_xlabel(truth_variable_label)
        ax.set_ylabel("Approx. pressure (hPa)")
        ax.set_ylim(common_ymin, common_ymax)
        if common_ymin < common_ymax:
            ax.invert_yaxis()
        fig.suptitle(f"{args.variable} profile: {region}", y=0.985)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
        profile_out = output_dir / f"{args.variable}_{region}_truth_profile.png"
        fig.savefig(profile_out, bbox_inches="tight")
        plt.close(fig)
        print(profile_out)

        for label, lead_hours, level_axis, values in loaded:
            fig, ax = plt.subplots(
                figsize=(6.0, 4.8),
                dpi=160,
                constrained_layout=False,
            )
            mappable = ax.contourf(
                lead_hours,
                level_axis,
                values,
                levels=contour_levels,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                extend="both",
            )
            ax.set_title(label, pad=10)
            ax.set_xlabel("Lead time (hours)")
            ax.set_ylabel("Approx. pressure (hPa)")
            ax.set_xlim(0.0, args.max_lead_hours)
            ax.set_ylim(common_ymin, common_ymax)
            if common_ymin < common_ymax:
                ax.invert_yaxis()
            cbar = fig.colorbar(
                mappable, ax=ax, shrink=0.92, pad=0.02, ticks=colorbar_ticks
            )
            cbar.set_label(f"{args.variable} anomaly")
            fig.suptitle(f"{args.variable} anomaly comparison: {region}", y=0.985)
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
            out = (
                output_dir
                / f"{args.variable}_{region}_{_slugify(label)}_anomaly_contours.png"
            )
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            print(out)


if __name__ == "__main__":
    main()
