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
r"""Pendergrass--Hartmann style rain-rate distributions on the cubesphere.

Implements the "frequency", "amount", and "rain-rate percentile" curves
of Pendergrass & Hartmann 2014 (Eqs. (1)-(4)) adapted to our (cell,
frame) sampling. Each sample $(k,t)$ is a cubesphere cell $k$ at
forecast frame $t$; the SCRIP cell area $a_k$ (m^2) supplies the
area-weighting; rain rates $r_{kt}$ are converted to mm/day via
``MS_TO_MM_DAY``. Let

    S      = all (k, t) pairs in the frame window
    N_T    = sum_{(k,t) in S} a_k            (total "area-time")
    B_i    = {(k,t) in S : R_i^l <= r_{kt} < R_i^r}
    dlnR   = ln(R_i^r) - ln(R_i^l)           (constant by construction)

Then

    f_i = (1 / dlnR) * (sum_{B_i} a_k)       / N_T
    p_i = (1 / dlnR) * (sum_{B_i} a_k r_{kt}) / N_T
    F_i = dry_frac + sum_{j<=i} f_j dlnR     (cumulative incl. dry)

Dry samples ($r < R_0^l$, default 0.1 mm/day) are counted in ``N_T``
but not placed in any rainy bin, so

    int f dlnR over rainy bins = rainy area-time fraction
    int p dlnR over rainy bins = area-weighted mean rain rate (mm/day)

Three curves are overlaid in each panel:

    1. SCREAM (reference)                       -- native ne1024pg2
    2. SCREAM (reference) 4x coarsened          -- area-weighted 4x4 blocks
    3. SCREAMCAST (rollout at 12-hour lead time) -- model forecast

Prediction source can be either an ACE forecast zarr
(``--pred-zarr``, dims ``(time, step, face, x, y)``) or an unstructured
``output_rollout/<tag>/precip_liq_surf_mass_flux_surf.nc``
(``--pred-nc``, dims ``(time, pixels)``). Truth is always read from
``--truth-zarr`` (the NC file's embedded ``truth`` is sentinel-filled
and ignored).
"""
# ruff: noqa: E402  (matplotlib.use("Agg") must run before pyplot import)
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import zarr
from plot_pdf_snapshot import (
    LINE_STYLE_COARSE,
    LINE_STYLE_PRED,
    LINE_STYLE_TRUTH,
    MS_TO_MM_DAY,
    NFACE,
    NSIDE,
    coarsen_4x_area_weighted,
    frame_window_steps,
    load_scrip_area_faces,
)


def coarse_area_1d(area_2d_faces: np.ndarray, factor: int = 4) -> np.ndarray:
    """Block-sum of SCRIP cell areas on each face.

    This is the matching per-cell weight for a field produced by
    ``coarsen_4x_area_weighted`` with the same factor: each coarse cell
    gets the sum of the ``factor**2`` native areas inside it, so
    ``sum(coarse_area) == sum(area_2d_faces)`` -- total earth area is
    preserved.
    """
    if NSIDE % factor != 0:
        raise ValueError(f"NSIDE={NSIDE} not divisible by factor={factor}")
    nside_c = NSIDE // factor
    blocks = area_2d_faces.reshape(NFACE, nside_c, factor, nside_c, factor)
    return blocks.sum(axis=(2, 4)).reshape(-1)


def build_log_bins(
    R_min: float, R_max: float, dlnR: float
) -> tuple[np.ndarray, float, int]:
    """Construct log-spaced bin edges with an *exactly* constant ``dlnR``.

    We snap ``n_bins`` to the nearest integer such that the spacing in
    ``ln R`` is ``dlnR``; the actual ``R_max`` is then
    ``R_min * exp(n_bins * dlnR)``, which will land very close to but
    not exactly on the requested ``R_max``.
    """
    if R_min <= 0 or R_max <= R_min or dlnR <= 0:
        raise ValueError(f"Invalid bin params: R_min={R_min} R_max={R_max} dlnR={dlnR}")
    n_bins = int(np.round(np.log(R_max / R_min) / dlnR))
    if n_bins < 1:
        raise ValueError(
            f"Bin params produce n_bins={n_bins} (need >= 1). "
            f"R_min={R_min}, R_max={R_max}, dlnR={dlnR}. "
            "Either widen R_max/R_min or lower dlnR."
        )
    edges = R_min * np.exp(np.arange(n_bins + 1) * dlnR)
    return edges, dlnR, n_bins


@dataclass
class RainDistAccumulator:
    """Incremental area-weighted accumulator for f_i, p_i, and dry mass.

    One instance per curve (native / coarsened / prediction). Call
    ``add(r_mm_day, w)`` once per frame with paired 1-D arrays.
    """

    bins: np.ndarray
    dlnR: float
    N_T: float = 0.0
    dry_w: float = 0.0
    over_w: float = 0.0
    freq_counts: np.ndarray = field(default=None)
    amount_counts: np.ndarray = field(default=None)

    def __post_init__(self) -> None:
        n = self.bins.size - 1
        self.freq_counts = np.zeros(n, dtype=np.float64)
        self.amount_counts = np.zeros(n, dtype=np.float64)

    def add(self, r_mm_day: np.ndarray, w: np.ndarray) -> None:
        if r_mm_day.shape != w.shape:
            raise ValueError(
                f"shape mismatch: values {r_mm_day.shape} vs weights " f"{w.shape}"
            )
        ok = np.isfinite(r_mm_day) & (r_mm_day >= 0.0)
        r = r_mm_day[ok]
        wv = w[ok]
        self.N_T += float(wv.sum())
        dry = r < self.bins[0]
        self.dry_w += float(wv[dry].sum())
        rv = r[~dry]
        wv_r = wv[~dry]
        # Partition rainy mass into in-range (goes into histogram bins)
        # vs overflow (r >= bins[-1]; np.histogram silently drops these).
        # We track overflow explicitly so percentile/cumulative curves
        # can correctly extend to 100% instead of plateauing early.
        over = rv >= self.bins[-1]
        self.over_w += float(wv_r[over].sum())
        fc, _ = np.histogram(rv, bins=self.bins, weights=wv_r)
        ac, _ = np.histogram(rv, bins=self.bins, weights=wv_r * rv)
        self.freq_counts += fc
        self.amount_counts += ac

    def results(self) -> dict:
        if self.N_T <= 0:
            raise RuntimeError("No samples accumulated (N_T == 0)")
        bin_centers = np.sqrt(self.bins[:-1] * self.bins[1:])
        f = self.freq_counts / (self.N_T * self.dlnR)
        p = self.amount_counts / (self.N_T * self.dlnR)
        dry_frac = self.dry_w / self.N_T
        over_frac = self.over_w / self.N_T
        rainy_frac = float(self.freq_counts.sum() / self.N_T) + over_frac
        # Cumulative probability (including dry mass) evaluated at the
        # *right* edge of each rainy bin. Prepending the (dry_frac,
        # bins[0]) point lets a single line span the dry->rainy
        # transition on the percentile plot. The overflow mass is added
        # back at bin[-1] so the last tabulated percentile matches
        # 1 - epsilon (epsilon = floating-point slop).
        cum_rainy = np.cumsum(f * self.dlnR)
        cum_total_right = dry_frac + cum_rainy
        cum_total_right[-1] += over_frac  # overflow lives at r >= bins[-1]
        mean_rate = float((p * self.dlnR).sum())  # in-range contribution
        return dict(
            bins=self.bins,
            bin_centers=bin_centers,
            f=f,
            p=p,
            dry_frac=dry_frac,
            over_frac=over_frac,
            rainy_frac=rainy_frac,
            cum_total_right=cum_total_right,
            mean_rate_mm_day=mean_rate,
        )


def compute_all_curves(
    *,
    mode: str,
    pred_zarr_path: str | None,
    pred_nc_dir: str | None,
    truth_zarr_path: str,
    scrip_path: str,
    step: int,
    initial_frame: int,
    frame_window: int,
    R_min: float,
    R_max: float,
    dlnR: float,
    progress: bool,
) -> dict:
    bins, dlnR_actual, n_bins = build_log_bins(R_min, R_max, dlnR)
    if progress:
        print(
            f"Bins: {n_bins} from {bins[0]:.4g} to {bins[-1]:.4g} mm/day "
            f"(dlnR={dlnR_actual:.3f}, ~{1/dlnR_actual:.1f} bins/log-unit)"
        )

    area_faces = load_scrip_area_faces(scrip_path)
    area_1d_cubesphere = xr.open_dataset(scrip_path)["grid_area"].values.astype(
        np.float64
    )  # (TOTAL_PIXELS,), matches SCREAM truth zarr ordering
    area_1d_faceflat = area_faces.reshape(-1)  # matches zarr pred (face, x, y)
    area_1d_coarse = coarse_area_1d(area_faces)  # matches coarsen_4x output

    if progress:
        print(
            f"Area totals: native={area_1d_cubesphere.sum():.3e}, "
            f"face-flat={area_1d_faceflat.sum():.3e}, "
            f"coarse={area_1d_coarse.sum():.3e}"
        )

    # Open pred + truth once; slice per frame below.
    if mode == "zarr":
        pred_ds = xr.open_zarr(pred_zarr_path, consolidated=True)
    else:
        nc_path = os.path.join(pred_nc_dir, "precip_liq_surf_mass_flux_surf.nc")
        if not os.path.isfile(nc_path):
            raise FileNotFoundError(f"Missing NC file: {nc_path}")
        pred_ds = xr.open_dataset(nc_path)

    # Wrap the histogram loop in try/finally so pred_ds (an xarray
    # Dataset) is released on any exception path. Note: zarr v3's
    # zarr.Group has no .close() method — it is a thin handle around a
    # LocalStore that holds no persistent OS file descriptors across
    # reads, so truth_group needs no explicit cleanup. Any array data
    # we care about is materialized to numpy inside the loop.
    try:
        if mode == "zarr":
            pred_da = pred_ds["precip_liq_surf_mass_flux"].isel(time=0)
        else:
            pred_da = pred_ds["prediction"]

        truth_group = zarr.open_group(truth_zarr_path, mode="r")
        truth_arr = truth_group["precip_liq_surf_mass_flux"]

        native = RainDistAccumulator(bins=bins, dlnR=dlnR_actual)
        coarse = RainDistAccumulator(bins=bins, dlnR=dlnR_actual)
        pred = RainDistAccumulator(bins=bins, dlnR=dlnR_actual)

        steps = frame_window_steps(step, frame_window)
        t0 = time.time()
        for s in steps:
            frame_k = initial_frame + s

            truth_ms = np.asarray(truth_arr[frame_k, :]).astype(np.float64)
            truth_mm = truth_ms * MS_TO_MM_DAY
            native.add(truth_mm, area_1d_cubesphere)

            coarse_ms = coarsen_4x_area_weighted(truth_ms, area_faces)
            coarse.add(coarse_ms * MS_TO_MM_DAY, area_1d_coarse)
            del truth_ms, coarse_ms, truth_mm

            if mode == "zarr":
                pred_vals = pred_da.isel(step=s).values.astype(np.float64).reshape(-1)
                pred.add(pred_vals * MS_TO_MM_DAY, area_1d_faceflat)
            else:
                pred_vals = pred_da.isel(time=s).values.astype(np.float64)
                pred.add(pred_vals * MS_TO_MM_DAY, area_1d_cubesphere)
            del pred_vals

            if progress:
                print(
                    f"  frame step={s} (truth frame {frame_k}) "
                    f"[{time.time() - t0:.1f}s]"
                )
    finally:
        pred_ds.close()

    return {
        "bins": bins,
        "dlnR": dlnR_actual,
        "native": native.results(),
        "coarse": coarse.results(),
        "pred": pred.results(),
    }


def _percentile_forward(P: np.ndarray) -> np.ndarray:
    """Map percentile P in [0, 100) to -log10(1 - P/100)."""
    x = 1.0 - np.asarray(P, dtype=np.float64) / 100.0
    return -np.log10(np.clip(x, 1e-12, 1.0))


def _percentile_inverse(x: np.ndarray) -> np.ndarray:
    return 100.0 * (1.0 - 10.0 ** (-np.asarray(x, dtype=np.float64)))


def _plot_freq(ax, curves: list[tuple[dict, str, dict]]) -> None:
    for res, label, style in curves:
        ax.plot(res["bin_centers"], 100.0 * res["f"], label=label, **style)
    ax.set_xscale("log")
    ax.set_yscale("log")
    # Clamp the lower y limit to keep the sparse-sample noise floor
    # (Poisson jitter in bins with only a handful of hits) out of the
    # visible panel. 1e-4 % -> ~1 sample / 1e6 area-time weight, which
    # is well below where our 7-frame window can resolve meaningful
    # structure.
    ax.set_ylim(bottom=1e-4)
    ax.set_xlabel(r"Precipitation (mm d$^{-1}$)", fontsize=20)
    ax.set_ylabel(r"Frequency (%)", fontsize=22)
    ax.set_title("Frequency distribution", loc="left", fontsize=22, fontweight="bold")
    ax.tick_params(labelsize=18)
    ax.grid(True, which="major", axis="both", alpha=0.15, linewidth=0.6)


def _plot_amount(ax, curves: list[tuple[dict, str, dict]]) -> None:
    for res, label, style in curves:
        ax.plot(res["bin_centers"], res["p"], label=label, **style)
    ax.set_xscale("log")
    ax.set_xlabel(r"Precipitation (mm d$^{-1}$)", fontsize=20)
    ax.set_ylabel(r"Amount (mm d$^{-1}$)", fontsize=22)
    ax.set_title("Amount distribution", loc="left", fontsize=22, fontweight="bold")
    ax.tick_params(labelsize=18)
    ax.grid(True, which="major", axis="y", alpha=0.15, linewidth=0.6)


def _plot_percentile(ax, curves: list[tuple[dict, str, dict]]) -> None:
    """Rain rate vs percentile on a log-reduction x axis (90, 99, 99.9, ...)."""
    for res, label, style in curves:
        P_edges = 100.0 * res["cum_total_right"]
        R_edges = res["bins"][1:]
        # Prepend the dry->rainy transition so the curve starts at (dry_frac*100, R_min).
        P_plot = np.concatenate([[100.0 * res["dry_frac"]], P_edges])
        R_plot = np.concatenate([[res["bins"][0]], R_edges])
        # Only keep strictly increasing percentiles for a clean line.
        keep = np.concatenate([[True], np.diff(P_plot) > 0])
        ax.plot(P_plot[keep], R_plot[keep], label=label, **style)

    ax.set_xscale("function", functions=(_percentile_forward, _percentile_inverse))
    ticks = [50.0, 90.0, 99.0, 99.9, 99.99]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["50", "90", "99", "99.9", "99.99"])
    ax.set_xlim(50.0, 99.99)
    ax.set_yscale("log")
    ax.set_xlabel("Percentile", fontsize=20)
    ax.set_ylabel(r"Precipitation (mm d$^{-1}$)", fontsize=22)
    ax.set_title("Precipitation percentile", loc="left", fontsize=22, fontweight="bold")
    ax.tick_params(labelsize=18)
    ax.grid(True, which="major", axis="both", alpha=0.15, linewidth=0.6)


def render_figure(results: dict, out_dir: Path, out_name: str) -> None:
    curves = [
        (results["native"], "SCREAM (reference)", LINE_STYLE_TRUTH),
        (
            results["coarse"],
            "SCREAM (reference) 4x coarsened",
            LINE_STYLE_COARSE,
        ),
        (results["pred"], "SCREAMCAST (rollout at 12-hour lead time)", LINE_STYLE_PRED),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), constrained_layout=True)
    _plot_freq(axes[0], curves)
    _plot_amount(axes[1], curves)
    _plot_percentile(axes[2], curves)

    # Compact shared legend below the three panels; identical entries in
    # every panel so one row is sufficient.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=3,
        frameon=False,
        fontsize=16,
        handlelength=2.4,
        columnspacing=1.6,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out_path = out_dir / f"{out_name}.{ext}"
        kw = {"bbox_inches": "tight", "facecolor": "white"}
        if ext == "png":
            kw["dpi"] = 200
        fig.savefig(out_path, **kw)
        print(f"Saved: {out_path}")
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--pred-zarr",
        default=None,
        help="ACE forecast zarr with dims (time, step, face, x, y).",
    )
    g.add_argument(
        "--pred-nc-dir",
        default=None,
        help="output_rollout/<tag>/ directory containing "
        "precip_liq_surf_mass_flux_surf.nc (dims: time, pixels).",
    )
    p.add_argument("--truth-zarr", required=True)
    p.add_argument("--scrip-path", required=True)
    p.add_argument(
        "--step",
        type=int,
        default=71,
        help="Center forecast step index (0-based). Step 71 with "
        "--step-minutes 10 is ~12 h lead time.",
    )
    p.add_argument(
        "--frame-window",
        type=int,
        default=3,
        help="Include steps in [step - W, step + W]; default 3 -> 7 frames.",
    )
    p.add_argument("--initial-frame", type=int, default=1729)
    p.add_argument(
        "--r-min",
        type=float,
        default=0.1,
        help="Dry/rainy threshold in mm/day. Matches Pendergrass & Hartmann (0.1).",
    )
    p.add_argument(
        "--r-max",
        type=float,
        default=10000.0,
        help="Upper bin edge in mm/day. 10000 (~5 decades) covers even "
        "the most extreme instantaneous 3-km convective cells so the "
        "cumulative / percentile curves reach 100% without clamping.",
    )
    p.add_argument(
        "--dlnR",
        type=float,
        default=0.1,
        help="Constant bin spacing in ln(R). 0.1 reproduces the paper's "
        "~10%%-wide bins.",
    )
    p.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    p.add_argument("--out-name", default="rain_distribution")
    p.add_argument("--progress", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    mode = "zarr" if args.pred_zarr is not None else "nc"

    lead_hours = (args.step + 1) * 10 / 60.0
    n_frames = 2 * args.frame_window + 1

    print(f"Mode        : {mode}")
    print(f"Prediction  : " f"{args.pred_zarr if mode == 'zarr' else args.pred_nc_dir}")
    print(f"Truth zarr  : {args.truth_zarr}")
    print(f"SCRIP       : {args.scrip_path}")
    print(
        f"Step        : {args.step} (~{lead_hours:.2f} h lead; "
        f"truth frame {args.initial_frame + args.step}); "
        f"window={n_frames} frames"
    )

    results = compute_all_curves(
        mode=mode,
        pred_zarr_path=args.pred_zarr,
        pred_nc_dir=args.pred_nc_dir,
        truth_zarr_path=args.truth_zarr,
        scrip_path=args.scrip_path,
        step=args.step,
        initial_frame=args.initial_frame,
        frame_window=args.frame_window,
        R_min=args.r_min,
        R_max=args.r_max,
        dlnR=args.dlnR,
        progress=args.progress,
    )

    # Stdout diagnostics only; the figure itself is kept clean (no
    # text overlay) per Pendergrass-style paper figures.
    print(f"\nstep = {args.step}  (~{lead_hours:.1f} h; window {n_frames} frames)")
    print(
        f"bins : {results['native']['bins'].size - 1} log-spaced, "
        f"dlnR={results['dlnR']:.3f}"
    )
    print("                       rainy%   mean(mm/d)   overflow(>r_max)")
    for name, key in (
        ("SCREAM native        ", "native"),
        ("SCREAM 4x coarsened  ", "coarse"),
        ("SCREAMCAST prediction", "pred"),
    ):
        r = results[key]
        print(
            f"  {name} {100 * r['rainy_frac']:6.2f}%  "
            f"{r['mean_rate_mm_day']:8.3f}   "
            f"{100 * r['over_frac']:8.1e}%"
        )
    print()

    render_figure(
        results,
        out_dir=Path(args.out_dir),
        out_name=args.out_name,
    )


if __name__ == "__main__":
    main()
