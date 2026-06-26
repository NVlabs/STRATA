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
"""PDF comparison over a window of frames around a target lead time.

Generates a 1 x 2 figure whose panels are:

    Panel 0  —  omega at a mid-troposphere level
    Panel 1  —  precip_liq_surf_mass_flux (rain rate, mm/day)

Each panel overlays three probability density functions accumulated over
``2 * --frame-window + 1`` consecutive forecast steps (default 7 frames
centered on ``--step``, spanning ~1 hour around the 12-hour lead time)
on the ne1024pg2 cubesphere (6 x 2048 x 2048 = 25,165,824 cells per
frame):

    1. SCREAM (reference)                                — native 25M points / frame
    2. SCREAM (reference) 4x coarsened                   — 1.56M points / frame,
       area-weighted 4x4 block-mean on the (face, x, y) grid using the
       SCRIP cell areas.
    3. SCREAMCAST (rollout at 12-hour lead time)         — model forecast

Histogram counts are accumulated incrementally across the window so we
never materialize a concatenated array larger than one frame.

The prediction source can be either

  * the ACE forecast zarr with dims ``(time, step, face, x, y)`` and a
    4-element ``level`` coord (currently ``[13, 18, 25, 30]`` — a subset
    of SCREAM's 32 native levels), or
  * the unstructured ``output_rollout/<tag>/`` directory with per-variable
    NetCDF files such as ``omega_20.nc`` and
    ``precip_liq_surf_mass_flux_surf.nc`` (dims ``(time, pixels)``).

Truth is always read from ``--truth-zarr`` at frame
``initial_frame + step``; the ``truth`` arrays embedded inside the
``output_rollout/`` NCs are ignored (they are sentinel-filled, see
``plots/plot_precip_hovmoller_ace.py``).
"""
# ruff: noqa: E402  (matplotlib.use("Agg") must run before pyplot import)
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import zarr

# Reuse the 1D <-> (face, x, y) reshape used everywhere else in the repo.
from plot_precip_hovmoller_ace import face_1d_to_2d

MS_TO_MM_DAY = 1000.0 * 86400.0
NSIDE = 2048
NFACE = 6
FACE_PIXELS = NSIDE * NSIDE
TOTAL_PIXELS = NFACE * FACE_PIXELS

# Colorblind-friendly palette (Wong 2011, "Points of View: Color blindness",
# Nature Methods 8, 441). Three distinct colors *and* three distinct linestyles
# so the lines are decodable under deuteranopia/protanopia/tritanopia and in
# black-and-white print. The reference line is thickest and drawn first so it
# acts as the visual baseline; the prediction is a thinner dashed line on top
# so it stays visible even where it overlaps the reference.
#   SCREAM (reference)                          — blue,        solid,  lw 3.0
#   SCREAM (reference) 4x coarsened             — gray,        dotted, lw 2.0
#   SCREAMCAST (rollout at 12-hour lead time)   — vermillion,  dashed, lw 2.2
LINE_STYLE_TRUTH = dict(color="#0072B2", linewidth=3.0, linestyle="-", zorder=2)
LINE_STYLE_COARSE = dict(color="#555555", linewidth=2.0, linestyle=":", zorder=3)
LINE_STYLE_PRED = dict(color="#D55E00", linewidth=2.2, linestyle="--", zorder=4)


def flatten_faces_1d(arr_2d: np.ndarray) -> np.ndarray:
    """Flatten a (..., face, x, y) array to one axis.

    Ordering is irrelevant for PDF histograms, so no cubesphere_faces_2d
    inverse-reshape is needed here.
    """
    return np.asarray(arr_2d).reshape(-1)


def reshape_1d_to_faces(arr_1d: np.ndarray) -> np.ndarray:
    """Reshape the SCREAM 1-D cubesphere_faces_2d layout to
    ``(nface, nside, nside)`` via the standard ne*npg reshape-transpose.
    """
    arr_1d = np.asarray(arr_1d)
    if arr_1d.size != TOTAL_PIXELS:
        raise ValueError(f"Expected {TOTAL_PIXELS} elements, got {arr_1d.size}")
    out = np.empty((NFACE, NSIDE, NSIDE), dtype=arr_1d.dtype)
    for f in range(NFACE):
        out[f] = face_1d_to_2d(arr_1d[f * FACE_PIXELS : (f + 1) * FACE_PIXELS], NSIDE)
    return out


def coarsen_4x_area_weighted(
    arr_1d: np.ndarray, area_2d_faces: np.ndarray, factor: int = 4
) -> np.ndarray:
    """Area-weighted block-mean of a 1-D cubesphere field.

    Reshapes ``arr_1d`` to ``(6, nside, nside)``, multiplies by per-cell
    areas, sums each ``factor x factor`` block, divides by the matching
    block area sum, and returns a flat 1-D array of length
    ``6 * (nside//factor)**2``.
    """
    if NSIDE % factor != 0:
        raise ValueError(f"NSIDE={NSIDE} not divisible by factor={factor}")
    arr_2d = reshape_1d_to_faces(arr_1d).astype(np.float64)
    area = area_2d_faces  # already float64, shape (6, nside, nside)

    nside_c = NSIDE // factor
    area_blocks = area.reshape(NFACE, nside_c, factor, nside_c, factor)
    val_blocks = (arr_2d * area).reshape(NFACE, nside_c, factor, nside_c, factor)
    w_sum = area_blocks.sum(axis=(2, 4))
    v_sum = val_blocks.sum(axis=(2, 4))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = v_sum / w_sum
    return out.reshape(-1)


def load_scrip_area_faces(scrip_path: str) -> np.ndarray:
    """Load SCRIP ``grid_area`` and reshape to ``(6, nside, nside)``."""
    ds = xr.open_dataset(scrip_path)
    area_1d = ds["grid_area"].values.astype(np.float64)
    ds.close()
    return reshape_1d_to_faces(area_1d)


def density_from_counts(counts: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Normalize histogram counts into a PDF over the bin range.

    Mimics ``np.histogram(..., density=True)``: integrates to 1 over the
    bins, which is well-defined even when values outside ``[bins[0],
    bins[-1]]`` were silently dropped from ``counts``.
    """
    widths = np.diff(bins)
    total = counts.sum()
    if total <= 0:
        return np.zeros_like(counts, dtype=np.float64)
    return counts / (total * widths)


def pdf_step_plot(
    ax,
    entries: list[tuple[np.ndarray, str, dict]],
    bins: np.ndarray,
    xscale: str,
    xlabel: str,
    title: str,
) -> None:
    """Render step-histogram PDFs from pre-computed histogram counts.

    ``entries`` items are ``(counts, label, style_kwargs)`` tuples, with
    ``counts`` an array of length ``len(bins) - 1``.  This lets callers
    accumulate counts incrementally across many frames without having to
    concatenate giant arrays in memory.
    """
    for counts, label, kw in entries:
        density = density_from_counts(counts, bins)
        ax.stairs(density, bins, label=label, **kw)
    ax.set_xscale(xscale)
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel("PDF", fontsize=18)
    ax.set_title(title, loc="left", fontsize=19, fontweight="bold")
    ax.tick_params(labelsize=16)
    # Very light y-only gridlines so the data lines dominate the panel.
    ax.grid(True, axis="y", which="major", alpha=0.12, linewidth=0.6)
    ax.grid(False, axis="x")
    # Per-panel legends are removed in favor of a single shared legend that
    # main() places below the figure (identical entries across both panels).


def frame_window_steps(step_center: int, frame_window: int) -> list[int]:
    """Return ``[step_center - W, …, step_center + W]`` inclusive."""
    if frame_window < 0:
        raise ValueError("frame_window must be >= 0")
    return list(range(step_center - frame_window, step_center + frame_window + 1))


def compute_omega_panel(
    ax,
    mode: str,
    pred_zarr_path: str | None,
    pred_nc_dir: str | None,
    truth_zarr_path: str,
    scrip_area_faces: np.ndarray,
    step: int,
    initial_frame: int,
    frame_window: int,
    level_zarr: int,
    level_nc: int,
    omega_xlim: tuple[float, float],
    omega_bins: int,
    progress: bool,
) -> None:
    truth_level = level_zarr if mode == "zarr" else level_nc
    lo, hi = omega_xlim
    bins = np.linspace(lo, hi, omega_bins + 1)
    n_bins = bins.size - 1
    steps = frame_window_steps(step, frame_window)

    # Open the prediction + truth handles once; slice per frame.
    if mode == "zarr":
        pred_ds = xr.open_zarr(pred_zarr_path, consolidated=True)
    else:
        nc_path = os.path.join(pred_nc_dir, f"omega_{level_nc}.nc")
        if not os.path.isfile(nc_path):
            raise FileNotFoundError(f"Missing NC file: {nc_path}")
        pred_ds = xr.open_dataset(nc_path)

    # Wrap the rest in try/finally so pred_ds is closed on any exception
    # path (e.g. a histogram/step failure). The zarr v3 directory-store
    # Group returned by zarr.open_group has no .close() and holds no
    # persistent handles, so it needs no explicit cleanup.
    try:
        if mode == "zarr":
            level_vals = pred_ds["level"].values.tolist()
            if level_zarr not in level_vals:
                raise ValueError(
                    f"--level-zarr {level_zarr} not in zarr level coord "
                    f"{level_vals}"
                )
            pred_da = pred_ds["omega"].isel(time=0).sel(level=level_zarr)
        else:
            pred_da = pred_ds["prediction"]

        truth_group = zarr.open_group(truth_zarr_path, mode="r")
        truth_arr = truth_group["omega"]

        truth_counts = np.zeros(n_bins, dtype=np.float64)
        coarse_counts = np.zeros(n_bins, dtype=np.float64)
        pred_counts = np.zeros(n_bins, dtype=np.float64)

        t0 = time.time()
        for s in steps:
            frame_k = initial_frame + s
            truth_1d = np.asarray(truth_arr[frame_k, truth_level, :]).astype(np.float64)
            truth_counts += np.histogram(truth_1d, bins=bins)[0]
            coarse_1d = coarsen_4x_area_weighted(truth_1d, scrip_area_faces)
            coarse_counts += np.histogram(coarse_1d, bins=bins)[0]
            del truth_1d, coarse_1d

            if mode == "zarr":
                pred_1d = pred_da.isel(step=s).values.astype(np.float64).reshape(-1)
            else:
                pred_1d = pred_da.isel(time=s).values.astype(np.float64)
            pred_counts += np.histogram(pred_1d, bins=bins)[0]
            del pred_1d

            if progress:
                print(
                    f"  omega frame step={s} (truth frame {frame_k}) "
                    f"[{time.time() - t0:.1f}s]"
                )
    finally:
        pred_ds.close()

    title = r"$\omega$ at 500 hPa"
    pdf_step_plot(
        ax,
        [
            (truth_counts, "SCREAM (reference)", LINE_STYLE_TRUTH),
            (coarse_counts, "SCREAM (reference) 4x coarsened", LINE_STYLE_COARSE),
            (pred_counts, "SCREAMCAST (rollout at 12-hour lead time)", LINE_STYLE_PRED),
        ],
        bins=bins,
        xscale="linear",
        xlabel=r"$\omega$ (Pa s$^{-1}$)",
        title=title,
    )
    ax.set_xlim(lo, hi)


def compute_precip_panel(
    ax,
    mode: str,
    pred_zarr_path: str | None,
    pred_nc_dir: str | None,
    truth_zarr_path: str,
    scrip_area_faces: np.ndarray,
    step: int,
    initial_frame: int,
    frame_window: int,
    precip_floor: float,
    precip_xlim: tuple[float, float],
    precip_bins: int,
    progress: bool,
) -> None:
    lo, hi = precip_xlim
    bins = np.logspace(np.log10(lo), np.log10(hi), precip_bins + 1)
    n_bins = bins.size - 1
    steps = frame_window_steps(step, frame_window)

    # Open prediction + truth handles once; slice per frame below.
    if mode == "zarr":
        pred_ds = xr.open_zarr(pred_zarr_path, consolidated=True)
    else:
        nc_path = os.path.join(pred_nc_dir, "precip_liq_surf_mass_flux_surf.nc")
        if not os.path.isfile(nc_path):
            raise FileNotFoundError(f"Missing NC file: {nc_path}")
        pred_ds = xr.open_dataset(nc_path)

    # Clip below precip_floor (mm/day) before binning in log-space to keep
    # pure-zero dry cells out of the -inf bin.
    def _positive_mm(x: np.ndarray) -> np.ndarray:
        mm = x * MS_TO_MM_DAY
        return mm[mm >= precip_floor]

    # try/finally so pred_ds is closed on any exception path. zarr v3
    # directory-store Groups hold no persistent handles and have no
    # .close(), so truth_group needs no explicit cleanup.
    try:
        if mode == "zarr":
            pred_da = pred_ds["precip_liq_surf_mass_flux"].isel(time=0)
        else:
            pred_da = pred_ds["prediction"]

        truth_group = zarr.open_group(truth_zarr_path, mode="r")
        truth_arr = truth_group["precip_liq_surf_mass_flux"]

        truth_counts = np.zeros(n_bins, dtype=np.float64)
        coarse_counts = np.zeros(n_bins, dtype=np.float64)
        pred_counts = np.zeros(n_bins, dtype=np.float64)

        t0 = time.time()
        for s in steps:
            frame_k = initial_frame + s
            truth_1d = np.asarray(truth_arr[frame_k, :]).astype(np.float64)
            truth_counts += np.histogram(_positive_mm(truth_1d), bins=bins)[0]
            coarse_1d = coarsen_4x_area_weighted(truth_1d, scrip_area_faces)
            coarse_counts += np.histogram(_positive_mm(coarse_1d), bins=bins)[0]
            del truth_1d, coarse_1d

            if mode == "zarr":
                pred_1d = pred_da.isel(step=s).values.astype(np.float64).reshape(-1)
            else:
                pred_1d = pred_da.isel(time=s).values.astype(np.float64)
            pred_counts += np.histogram(_positive_mm(pred_1d), bins=bins)[0]
            del pred_1d

            if progress:
                print(
                    f"  precip frame step={s} (truth frame {frame_k}) "
                    f"[{time.time() - t0:.1f}s]"
                )
    finally:
        pred_ds.close()

    title = "Surface precipitation"
    pdf_step_plot(
        ax,
        [
            (truth_counts, "SCREAM (reference)", LINE_STYLE_TRUTH),
            (coarse_counts, "SCREAM (reference) 4x coarsened", LINE_STYLE_COARSE),
            (pred_counts, "SCREAMCAST (rollout at 12-hour lead time)", LINE_STYLE_PRED),
        ],
        bins=bins,
        xscale="log",
        xlabel="rain rate (mm day$^{-1}$)",
        title=title,
    )
    ax.set_xlim(lo, hi)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--pred-zarr",
        default=None,
        help="ACE forecast zarr with dims (time, step, face, x, y) and a "
        "level coord (e.g. forecast_24hr.zarr).",
    )
    g.add_argument(
        "--pred-nc-dir",
        default=None,
        help="output_rollout/<tag>/ directory containing per-variable NC "
        "files such as omega_20.nc and precip_liq_surf_mass_flux_surf.nc.",
    )
    p.add_argument("--truth-zarr", required=True)
    p.add_argument(
        "--scrip-path",
        required=True,
        help="ne1024pg2 SCRIP file; provides grid_area for the 4x coarsen.",
    )
    p.add_argument(
        "--step",
        type=int,
        default=71,
        help="Center forecast step index (0-based). With --step-minutes "
        "10, step 71 ≈ 12-hour lead time.",
    )
    p.add_argument(
        "--frame-window",
        type=int,
        default=3,
        help="Include steps in [step - W, step + W] in the histograms "
        "(default W=3 → 7 frames totaling ~1 hour centered on --step). "
        "Set to 0 for single-frame sampling.",
    )
    p.add_argument("--initial-frame", type=int, default=1729)
    p.add_argument(
        "--level-zarr",
        type=int,
        default=18,
        help="omega level to use when the prediction is a zarr; must be "
        "one of the values in ds.level (currently [13, 18, 25, 30]).",
    )
    p.add_argument(
        "--level-nc",
        type=int,
        default=20,
        help="omega level to use when the prediction is an NC directory "
        "(selects omega_<level>.nc).",
    )
    p.add_argument(
        "--omega-xlim",
        type=float,
        nargs=2,
        default=(-15.0, 15.0),
        metavar=("LO", "HI"),
        help="Omega PDF x-axis range (Pa/s).",
    )
    p.add_argument("--omega-bins", type=int, default=120)
    p.add_argument(
        "--precip-xlim",
        type=float,
        nargs=2,
        default=(1e-2, 1e4),
        metavar=("LO", "HI"),
        help="Precip PDF x-axis range (mm/day, log-spaced).",
    )
    p.add_argument("--precip-bins", type=int, default=80)
    p.add_argument(
        "--precip-floor",
        type=float,
        default=1e-2,
        help="Clip precip values below this floor (mm/day) before binning "
        "in log-space; keeps the dry-cell mass out of -inf bins.",
    )
    p.add_argument(
        "--out-dir",
        default=os.path.dirname(os.path.abspath(__file__)),
    )
    p.add_argument("--out-name", default="pdf_snapshot")
    p.add_argument("--progress", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    mode = "zarr" if args.pred_zarr is not None else "nc"

    lead_hours = (args.step + 1) * 10 / 60.0  # informational
    n_frames = 2 * args.frame_window + 1
    window_first = args.step - args.frame_window
    window_last = args.step + args.frame_window

    print(f"Mode        : {mode}")
    print(f"Prediction  : " f"{args.pred_zarr if mode == 'zarr' else args.pred_nc_dir}")
    print(f"Truth zarr  : {args.truth_zarr}")
    print(f"SCRIP       : {args.scrip_path}")
    print(
        f"Step        : {args.step}  (≈ {lead_hours:.2f} h lead; "
        f"truth frame {args.initial_frame + args.step})"
    )
    print(
        f"Frame window: ±{args.frame_window} "
        f"→ steps [{window_first}..{window_last}] "
        f"({n_frames} frames; ~{n_frames * 10 / 60:.2f} h span)"
    )
    print(
        f"Level       : "
        f"{args.level_zarr if mode == 'zarr' else args.level_nc}  "
        f"({'zarr sel' if mode == 'zarr' else 'NC file suffix'})"
    )

    scrip_area_faces = load_scrip_area_faces(args.scrip_path)
    if args.progress:
        print(f"SCRIP area faces shape: {scrip_area_faces.shape}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), constrained_layout=True)

    print("Computing omega panel ...")
    compute_omega_panel(
        axes[0],
        mode=mode,
        pred_zarr_path=args.pred_zarr,
        pred_nc_dir=args.pred_nc_dir,
        truth_zarr_path=args.truth_zarr,
        scrip_area_faces=scrip_area_faces,
        step=args.step,
        initial_frame=args.initial_frame,
        frame_window=args.frame_window,
        level_zarr=args.level_zarr,
        level_nc=args.level_nc,
        omega_xlim=tuple(args.omega_xlim),
        omega_bins=args.omega_bins,
        progress=args.progress,
    )

    print("Computing precip panel ...")
    compute_precip_panel(
        axes[1],
        mode=mode,
        pred_zarr_path=args.pred_zarr,
        pred_nc_dir=args.pred_nc_dir,
        truth_zarr_path=args.truth_zarr,
        scrip_area_faces=scrip_area_faces,
        step=args.step,
        initial_frame=args.initial_frame,
        frame_window=args.frame_window,
        precip_floor=args.precip_floor,
        precip_xlim=tuple(args.precip_xlim),
        precip_bins=args.precip_bins,
        progress=args.progress,
    )

    # Single shared legend below both panels. Both panels produce the same
    # three handles/labels, so we harvest them from the first axis only and
    # place a 3-column legend outside the plotting area to avoid occluding
    # the data.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=3,
        frameon=False,
        fontsize=15,
        handlelength=3.0,
        columnspacing=2.2,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out_path = out_dir / f"{args.out_name}.{ext}"
        kw = {"bbox_inches": "tight", "facecolor": "white"}
        if ext == "png":
            kw["dpi"] = 200
        fig.savefig(out_path, **kw)
        print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
