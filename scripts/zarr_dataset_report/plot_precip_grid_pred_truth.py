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
"""Log-scaled precip snapshots, prediction vs truth, over a 24 h rollout.

Renders one figure with ``len(--frames)`` rows and 2 columns:

    col 0 : SCREAMCAST prediction  (precip_liq, mm day^-1, log scale)
    col 1 : SCREAM truth           (same)

The default frames are ``17, 35, 53, 71, 89, 107, 125, 143`` (stride 18
on a 10-min rollout), i.e. snapshots every 3 h from 3 h to 24 h lead.
Latitude is clipped to ±40° by default (tropics + subtropics) since
most heavy precip structure sits in that band and the plate-carrée
layout stays compact.

Truth is pulled from the SCREAM cubesphere zarr at frame
``initial_frame + pred_step`` and reshaped face-by-face using the same
``face_1d_to_2d`` layout as ``plot_snapshot_pred_truth_diff.py``.
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
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import zarr
from plot_snapshot_pred_truth_diff import (
    MS_TO_MM_DAY,
    NSIDE,
    load_coarsened_latlon,
    load_pred_faces,
    load_truth_faces,
    render_face_pcolormesh,
)

PRECIP_VAR = "precip_liq_surf_mass_flux"
# Match the precip_liq row in ``plot_snapshot_pred_truth_diff.py``.
CMAP = "YlGnBu"


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
    p.add_argument("--lat-min", type=float, default=-40.0)
    p.add_argument("--lat-max", type=float, default=40.0)
    p.add_argument(
        "--vmin",
        type=float,
        default=0.1,
        help="Floor of the log color scale, in mm/day (below = clipped).",
    )
    p.add_argument(
        "--vmax",
        type=float,
        default=None,
        help=(
            "Ceiling of the log color scale, in mm/day. If omitted, uses a "
            "99.9%% quantile over all panels."
        ),
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
            pred_group, PRECIP_VAR, frame, None, pred_level_arr, args.coarsen
        )
        truth = load_truth_faces(
            truth_group, PRECIP_VAR, args.initial_frame + frame, None, args.coarsen
        )
        pred = pred * MS_TO_MM_DAY
        truth = truth * MS_TO_MM_DAY
        pred_panels.append(pred)
        truth_panels.append(truth)
        lead_h = (frame + 1) * args.step_minutes / 60.0
        print(
            f"  frame {frame:3d}  lead {lead_h:4.1f}h  "
            f"pred[max={float(np.nanmax(pred)):.1f}]  "
            f"truth[max={float(np.nanmax(truth)):.1f}]  mm/day"
        )

    # Shared color scale: floor from CLI, ceiling from joint 99.9% quantile
    # over all panels unless the user pinned --vmax.
    vmin = max(args.vmin, 1e-3)
    if args.vmax is not None:
        vmax = float(args.vmax)
    else:
        all_vals = np.concatenate(
            [p.ravel() for p in pred_panels] + [t.ravel() for t in truth_panels]
        )
        all_vals = all_vals[np.isfinite(all_vals) & (all_vals > 0)]
        vmax = float(np.nanquantile(all_vals, 0.999)) if all_vals.size else 100.0
    vmax = max(vmax, vmin * 10.0)
    print(f"Color range (log): {vmin:.3g} .. {vmax:.3g} mm/day")
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax, clip=True)

    n_rows = len(frames)
    n_cols = 2
    proj = ccrs.PlateCarree()
    # Panel aspect is 360:80 = 4.5:1 (plate-carree over 40S-40N), so a
    # ~7" wide panel fits ~1.55" tall. Widen to match.
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7.0 * n_cols, 1.55 * n_rows + 1.0),
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
            # LogNorm requires strictly positive input; floor any non-positive
            # cells to well below vmin so LogNorm(clip=True) renders them as
            # the bottom color.
            data_pos = np.where(data > 0, data, vmin * 0.5)
            render_face_pcolormesh(
                ax, lat, lon, data_pos, cmap=CMAP, vmin=vmin, vmax=vmax
            )
            # ``render_face_pcolormesh`` applies a linear vmin/vmax; replace
            # the norm on every QuadMesh it added so the color scale is log.
            for artist in ax.collections:
                artist.set_norm(norm)
                artist.set_cmap(CMAP)
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.coastlines(linewidth=0.3, color="black")
            ax.add_feature(cfeature.BORDERS, linewidth=0.1, alpha=0.3)
            ax.set_xticks(np.arange(-180.0, 181.0, 60.0), crs=ccrs.PlateCarree())
            ax.set_yticks([-40.0, -20.0, 0.0, 20.0, 40.0], crs=ccrs.PlateCarree())
            ax.xaxis.set_major_formatter(LongitudeFormatter())
            ax.yaxis.set_major_formatter(LatitudeFormatter())
            ax.tick_params(axis="both", labelsize=7, length=2.5, pad=1.5)
            # Only label the x-axis on the bottom row to save space.
            if r != n_rows - 1:
                ax.set_xticklabels([])
            # Only label the y-axis on the left column.
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
        cb.set_label(r"precip_liq (mm day$^{-1}$)", fontsize=11)
        cb.ax.tick_params(labelsize=9)

    fig.suptitle(
        f"precip_liq (log scale, {args.lat_min:g}°–{args.lat_max:g}°)",
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
