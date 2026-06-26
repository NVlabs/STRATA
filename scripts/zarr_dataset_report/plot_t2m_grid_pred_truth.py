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
"""T_2m snapshots, prediction vs truth, over a 24 h rollout.

Renders one figure with ``len(--frames)`` rows and 2 columns:

    col 0 : SCREAMCAST prediction  (T_2m, K)
    col 1 : SCREAM truth           (same)

The default frames are ``17, 35, 53, 71, 89, 107, 125, 143`` (stride 18
on a 10-min rollout), i.e. snapshots every 3 h from 3 h to 24 h lead.
Latitude defaults to 60S-60N (polar caps clipped so the color scale is
not dominated by Antarctic/Arctic extremes); pass ``--lat-min/--lat-max``
to override.

Matches the T_2m row in ``plot_snapshot_pred_truth_diff.py``: linear
plasma color scale, robust vmin/vmax from joint quantiles computed over
the *displayed* latitude band only.
"""
# ruff: noqa: E402  (matplotlib.use("Agg") must run before pyplot import)
from __future__ import annotations

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr
from plot_snapshot_pred_truth_diff import (
    NSIDE,
    load_coarsened_latlon,
    load_pred_faces,
    load_truth_faces,
    render_face_pcolormesh,
)

T2M_VAR = "T_2m"
CMAP = "plasma"


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred-zarr", required=True)
    p.add_argument("--truth-zarr", required=True)
    p.add_argument(
        "--frames",
        default="17,35,53,71,89,107,125,143",
        help=(
            "Comma-separated 0-based pred-zarr step indices (default: "
            "stride-18 from step 17 to 143, ~every 3h on a 10-min rollout)."
        ),
    )
    p.add_argument(
        "--initial-frame",
        type=int,
        default=1729,
        help="Truth frame aligned with pred step 0.",
    )
    p.add_argument(
        "--coarsen",
        type=int,
        default=8,
        help="Per-axis stride applied to each 2048x2048 face before rendering.",
    )
    p.add_argument("--lat-min", type=float, default=-60.0)
    p.add_argument("--lat-max", type=float, default=60.0)
    p.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Floor of the color scale in K (default: joint 1%% quantile).",
    )
    p.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Ceiling of the color scale in K (default: joint 99%% quantile).",
    )
    p.add_argument("--step-minutes", type=float, default=10.0)
    p.add_argument("--out-dir", required=True)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    # load_pred_faces / load_truth_faces require coarsen to divide NSIDE.
    if NSIDE % args.coarsen != 0:
        raise ValueError(
            f"--coarsen={args.coarsen} must divide NSIDE={NSIDE} evenly "
            f"(got NSIDE % coarsen = {NSIDE % args.coarsen})"
        )
    frames = [int(x) for x in args.frames.split(",") if x.strip()]
    if not frames:
        raise ValueError("--frames parsed to an empty list")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Pred zarr  : {args.pred_zarr}")
    print(f"Truth zarr : {args.truth_zarr}")
    print(f"Frames     : {frames}")
    print(f"Lat band   : [{args.lat_min}, {args.lat_max}]")
    print(
        f"Coarsen    : {args.coarsen}  ({NSIDE}/{args.coarsen}"
        f"={NSIDE // args.coarsen} per face)"
    )
    print(f"Out dir    : {out_dir}")

    pred_group = zarr.open_group(args.pred_zarr, mode="r")
    pred_level_arr = np.asarray(pred_group["level"])
    truth_group = zarr.open_group(args.truth_zarr, mode="r")

    lat, lon = load_coarsened_latlon(pred_group, args.coarsen)

    pred_panels: list[np.ndarray] = []
    truth_panels: list[np.ndarray] = []
    for frame in frames:
        pred = load_pred_faces(
            pred_group, T2M_VAR, frame, None, pred_level_arr, args.coarsen
        )
        truth = load_truth_faces(
            truth_group, T2M_VAR, args.initial_frame + frame, None, args.coarsen
        )
        pred_panels.append(pred)
        truth_panels.append(truth)
        lead_h = (frame + 1) * args.step_minutes / 60.0
        print(
            f"  frame {frame:3d}  lead {lead_h:4.1f}h  "
            f"pred[min={float(np.nanmin(pred)):.1f},"
            f"max={float(np.nanmax(pred)):.1f}]  "
            f"truth[min={float(np.nanmin(truth)):.1f},"
            f"max={float(np.nanmax(truth)):.1f}]  K"
        )

    # Shared color range from joint 1..99% quantile over all panels, but
    # only over the latitude band that will actually be shown -- otherwise
    # the polar caps (if included in the data) would dominate the scale.
    band_mask = (lat >= args.lat_min) & (lat <= args.lat_max)
    if not np.any(band_mask):
        raise ValueError(
            f"No pixels fall inside lat band [{args.lat_min}, {args.lat_max}]"
        )
    all_vals = np.concatenate(
        [p[band_mask].ravel() for p in pred_panels]
        + [t[band_mask].ravel() for t in truth_panels]
    )
    all_vals = all_vals[np.isfinite(all_vals)]
    vmin = (
        float(np.nanquantile(all_vals, 0.01)) if args.vmin is None else float(args.vmin)
    )
    vmax = (
        float(np.nanquantile(all_vals, 0.99)) if args.vmax is None else float(args.vmax)
    )
    if vmax <= vmin:
        vmax = vmin + 1.0
    print(
        f"Color range (linear, lat {args.lat_min:g}..{args.lat_max:g}): "
        f"{vmin:.2f} .. {vmax:.2f} K"
    )

    n_rows = len(frames)
    n_cols = 2
    proj = ccrs.PlateCarree()
    # Plate-carree panel aspect is 360:lat_span. Pick panel_h first so
    # total figure height stays reasonable across lat clippings; panel_w
    # is then fixed by the aspect so cells pack tightly (no whitespace
    # between columns).
    lat_span = args.lat_max - args.lat_min
    if lat_span >= 150.0:
        panel_h = 2.3
    elif lat_span >= 100.0:
        panel_h = 1.9
    else:
        panel_h = 1.55
    panel_w = panel_h * 360.0 / lat_span
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(panel_w * n_cols + 1.8, panel_h * n_rows + 1.0),
        subplot_kw={"projection": proj},
        squeeze=False,
        constrained_layout=True,
    )
    col_titles = ["SCREAMCAST", "SCREAM"]
    extent = [-180.0, 180.0, args.lat_min, args.lat_max]

    last_im = None
    for r, frame in enumerate(frames):
        lead_h = (frame + 1) * args.step_minutes / 60.0
        for c, data in enumerate([pred_panels[r], truth_panels[r]]):
            ax = axes[r, c]
            render_face_pcolormesh(ax, lat, lon, data, cmap=CMAP, vmin=vmin, vmax=vmax)
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.coastlines(linewidth=0.3, color="black")
            ax.add_feature(cfeature.BORDERS, linewidth=0.1, alpha=0.3)
            ax.set_xticks(np.arange(-180.0, 181.0, 60.0), crs=ccrs.PlateCarree())
            ax.set_yticks(
                np.arange(
                    np.ceil(args.lat_min / 30.0) * 30.0,
                    np.floor(args.lat_max / 30.0) * 30.0 + 1.0,
                    30.0,
                ),
                crs=ccrs.PlateCarree(),
            )
            ax.xaxis.set_major_formatter(LongitudeFormatter())
            ax.yaxis.set_major_formatter(LatitudeFormatter())
            ax.tick_params(axis="both", labelsize=7, length=2.5, pad=1.5)
            if r != n_rows - 1:
                ax.set_xticklabels([])
            if c != 0:
                ax.set_yticklabels([])
            last_im = ax.collections[0] if ax.collections else last_im
            if r == 0:
                ax.set_title(col_titles[c], fontsize=12, fontweight="bold")
        # Row label: lead time on the leftmost panel. Pushed out enough to
        # clear the y-tick labels we now draw on the left column.
        axes[r, 0].text(
            -0.13,
            0.5,
            f"step {frame}\n{lead_h:g} h",
            transform=axes[r, 0].transAxes,
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    if last_im is not None:
        cb = fig.colorbar(
            last_im,
            ax=axes.ravel().tolist(),
            orientation="vertical",
            shrink=0.85,
            pad=0.015,
            aspect=40,
        )
        cb.set_label(r"T$_{2m}$ (K)", fontsize=11)
        cb.ax.tick_params(labelsize=9)

    lat_desc = (
        f"{args.lat_min:g}°–{args.lat_max:g}°"
        if (args.lat_min > -89.9 or args.lat_max < 89.9)
        else "global"
    )
    fig.suptitle(
        f"T$_{{2m}}$ ({lat_desc})",
        fontsize=14,
        fontweight="bold",
    )

    stem = "grid"
    for ext in ("png", "pdf"):
        out = out_dir / f"{stem}.{ext}"
        save_kw = dict(bbox_inches="tight", facecolor="white")
        if ext == "png":
            save_kw["dpi"] = 160
        fig.savefig(out, **save_kw)
        print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
