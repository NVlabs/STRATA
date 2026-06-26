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
"""Global snapshot comparison: prediction, truth, pred - truth.

For each requested frame, renders one figure with 7 rows × 3 columns:

    row 0 : omega                at level 18  (Pa s^-1)
    row 1 : qv                   at level 25  (kg/kg)
    row 2 : U                    at level 25  (m/s)
    row 3 : V                    at level 25  (m/s)
    row 4 : PotentialTemperature at level 18  (K)
    row 5 : T_2m                              (K)
    row 6 : precip_liq_surf_mass_flux         (mm/day)
    cols  : prediction | truth | pred - truth

Level indices index the truth zarr's 32-level vertical axis directly;
they are matched to the pred zarr via the ``level`` coord (which
stores the NC-style indices [13, 18, 25, 30]).

Uses cartopy Plate Carrée with per-face ``pcolormesh``.  Faces are
coarsened by ``--coarsen`` (default 8) so the global render stays
fast; each figure at default settings is <1 min.
"""
# ruff: noqa: E402  (matplotlib.use("Agg") must run before pyplot import)
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr

NSIDE = 2048
MS_TO_MM_DAY = 1000.0 * 86400.0


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------
def face_1d_to_2d(arr1d: np.ndarray, nside: int = NSIDE) -> np.ndarray:
    ne = nside // 2
    npg = 2
    return arr1d.reshape(ne, ne, npg, npg).transpose(0, 2, 1, 3).reshape(nside, nside)


def wrap_lon(lon: np.ndarray) -> np.ndarray:
    """Wrap longitudes to [-180, 180) for plate-carrée plotting."""
    out = lon % 360.0
    return np.where(out > 180.0, out - 360.0, out)


@dataclass(frozen=True)
class VarSpec:
    row_id: str
    pred_var: str
    truth_var: str
    level: int | None  # None for 2D variables
    title: str
    unit: str
    cmap_main: str
    sym: bool
    scale: float = 1.0


def build_specs() -> list[VarSpec]:
    """Fixed 7-row layout.

    Row ordering is deliberate — see module docstring.
    """
    return [
        VarSpec(
            row_id="omega_L18",
            pred_var="omega",
            truth_var="omega",
            level=18,
            title=r"$\omega$ (lev 18)",
            unit=r"Pa s$^{-1}$",
            cmap_main="RdBu_r",
            sym=True,
        ),
        VarSpec(
            row_id="qv_L25",
            pred_var="qv",
            truth_var="qv",
            level=25,
            title="qv (lev 25)",
            unit=r"kg kg$^{-1}$",
            cmap_main="BuPu",
            sym=False,
        ),
        VarSpec(
            row_id="U_L25",
            pred_var="U",
            truth_var="U",
            level=25,
            title="U (lev 25)",
            unit=r"m s$^{-1}$",
            cmap_main="RdBu_r",
            sym=True,
        ),
        VarSpec(
            row_id="V_L25",
            pred_var="V",
            truth_var="V",
            level=25,
            title="V (lev 25)",
            unit=r"m s$^{-1}$",
            cmap_main="RdBu_r",
            sym=True,
        ),
        VarSpec(
            row_id="theta_L18",
            pred_var="PotentialTemperature",
            truth_var="PotentialTemperature",
            level=18,
            title=r"$\theta$ (lev 18)",
            unit="K",
            cmap_main="plasma",
            sym=False,
        ),
        VarSpec(
            row_id="T_2m",
            pred_var="T_2m",
            truth_var="T_2m",
            level=None,
            title=r"T$_{2m}$",
            unit="K",
            cmap_main="plasma",
            sym=False,
        ),
        VarSpec(
            row_id="precip_liq",
            pred_var="precip_liq_surf_mass_flux",
            truth_var="precip_liq_surf_mass_flux",
            level=None,
            title="precip (liq)",
            unit=r"mm day$^{-1}$",
            cmap_main="YlGnBu",
            sym=False,
            scale=MS_TO_MM_DAY,
        ),
    ]


def pred_level_index(pred_level_arr: np.ndarray, nc_level: int) -> int:
    hits = np.where(pred_level_arr == nc_level)[0]
    if hits.size == 0:
        raise KeyError(
            f"NC level {nc_level} not present in pred zarr levels "
            f"{pred_level_arr.tolist()}"
        )
    return int(hits[0])


def load_pred_faces(
    pred_group: zarr.Group,
    var: str,
    frame: int,
    level: int | None,
    pred_level_arr: np.ndarray,
    coarsen: int,
) -> np.ndarray:
    """Return a ``(n_faces, nc, nc)`` array, coarsened by ``coarsen`` per axis."""
    arr = pred_group[var]
    n_faces = int(arr.shape[-3])
    if arr.ndim == 6:
        if level is None:
            raise ValueError(f"{var!r} is 6-D but no level was given")
        lvl_idx = pred_level_index(pred_level_arr, level)
        out = np.empty((n_faces, NSIDE // coarsen, NSIDE // coarsen), dtype=np.float32)
        for f in range(n_faces):
            slab = np.asarray(arr[0, frame, lvl_idx, f, :, :]).astype(np.float32)
            out[f] = slab[::coarsen, ::coarsen]
        return out
    if arr.ndim == 5:
        out = np.empty((n_faces, NSIDE // coarsen, NSIDE // coarsen), dtype=np.float32)
        for f in range(n_faces):
            slab = np.asarray(arr[0, frame, f, :, :]).astype(np.float32)
            out[f] = slab[::coarsen, ::coarsen]
        return out
    raise ValueError(f"Unexpected pred zarr ndim for {var!r}: {arr.ndim}")


def load_truth_faces(
    truth_group: zarr.Group,
    var: str,
    frame: int,
    level: int | None,
    coarsen: int,
    n_faces: int = 6,
) -> np.ndarray:
    arr = truth_group[var]
    if arr.ndim == 3:
        if level is None:
            raise ValueError(f"{var!r} is 3-D but no level was given")
        full = np.asarray(arr[frame, level, :]).astype(np.float32)
    elif arr.ndim == 2:
        full = np.asarray(arr[frame, :]).astype(np.float32)
    else:
        raise ValueError(f"Unexpected truth zarr ndim for {var!r}: {arr.ndim}")
    if full.size != n_faces * NSIDE * NSIDE:
        raise ValueError(f"Truth pixels {full.size} != {n_faces}*{NSIDE}^2")
    out = np.empty((n_faces, NSIDE // coarsen, NSIDE // coarsen), dtype=np.float32)
    for f in range(n_faces):
        face2d = face_1d_to_2d(full[f * NSIDE * NSIDE : (f + 1) * NSIDE * NSIDE], NSIDE)
        out[f] = face2d[::coarsen, ::coarsen]
    return out


def load_coarsened_latlon(
    pred_group: zarr.Group, coarsen: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return coarsened ``(n_faces, nc, nc)`` lat/lon arrays."""
    lat = np.asarray(pred_group["lat"])[:, ::coarsen, ::coarsen].astype(np.float32)
    lon = np.asarray(pred_group["lon"])[:, ::coarsen, ::coarsen].astype(np.float32)
    return lat, lon


# ---------------------------------------------------------------------------
# Rendering.
# ---------------------------------------------------------------------------
def shared_vrange(
    pred: np.ndarray,
    truth: np.ndarray,
    sym: bool,
    q: float = 0.99,
) -> tuple[float, float]:
    concat = np.concatenate([pred.ravel(), truth.ravel()])
    vmin = float(np.nanquantile(concat, 1.0 - q))
    vmax = float(np.nanquantile(concat, q))
    if sym:
        a = max(abs(vmin), abs(vmax)) or 1.0
        vmin, vmax = -a, a
    elif vmin == vmax:
        vmin -= 0.5
        vmax += 0.5
    return vmin, vmax


def diff_vrange(diff: np.ndarray, q: float = 0.99) -> tuple[float, float]:
    a = float(np.nanquantile(np.abs(diff), q)) or 1.0
    return -a, a


def render_face_pcolormesh(
    ax,
    lat: np.ndarray,
    lon: np.ndarray,
    data: np.ndarray,
    cmap: str,
    vmin: float,
    vmax: float,
):
    """Plot one face on a plate-carrée axes, handling the antimeridian."""
    lon_w = wrap_lon(lon)
    for f in range(data.shape[0]):
        d = data[f]
        xf = lon_w[f]
        yf = lat[f]
        # If this face straddles the dateline, render two copies to avoid
        # plate-carrée seams (cartopy accepts this).
        if (xf.max() - xf.min()) > 180:
            # Split into western (+180 shifted) and eastern parts.
            xf2 = np.where(xf < 0, xf + 360.0, xf)
            ax.pcolormesh(
                xf2,
                yf,
                d,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                shading="auto",
                transform=ccrs.PlateCarree(),
                rasterized=True,
            )
            ax.pcolormesh(
                xf2 - 360.0,
                yf,
                d,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                shading="auto",
                transform=ccrs.PlateCarree(),
                rasterized=True,
            )
        else:
            ax.pcolormesh(
                xf,
                yf,
                d,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                shading="auto",
                transform=ccrs.PlateCarree(),
                rasterized=True,
            )


def render_one_frame(
    frame: int,
    specs: list[VarSpec],
    pred_group: zarr.Group,
    truth_group: zarr.Group,
    initial_frame: int,
    pred_level_arr: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    coarsen: int,
    out_dir: Path,
    step_minutes: float,
) -> None:
    lead_h = (frame + 1) * step_minutes / 60.0
    n_rows = len(specs)
    n_cols = 3
    col_titles = ["SCREAMCAST", "SCREAM", "SCREAMCAST $-$ SCREAM"]

    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 1.9 * n_rows + 1.0),
        subplot_kw={"projection": proj},
        squeeze=False,
        constrained_layout=True,
    )

    for r, spec in enumerate(specs):
        pred = load_pred_faces(
            pred_group,
            spec.pred_var,
            frame,
            spec.level,
            pred_level_arr,
            coarsen,
        )
        truth = load_truth_faces(
            truth_group,
            spec.truth_var,
            initial_frame + frame,
            spec.level,
            coarsen,
        )
        if spec.scale != 1.0:
            pred = pred * spec.scale
            truth = truth * spec.scale
        diff = pred - truth

        vmin_main, vmax_main = shared_vrange(pred, truth, spec.sym)
        vmin_diff, vmax_diff = diff_vrange(diff)

        for c, data in enumerate([pred, truth, diff]):
            ax = axes[r, c]
            if c < 2:
                cmap = spec.cmap_main
                vmin, vmax = vmin_main, vmax_main
            else:
                cmap = "RdBu_r"
                vmin, vmax = vmin_diff, vmax_diff
            render_face_pcolormesh(
                ax,
                lat,
                lon,
                data,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_global()
            ax.coastlines(linewidth=0.3, color="black")
            ax.add_feature(cfeature.BORDERS, linewidth=0.1, alpha=0.3)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[c], fontsize=12, fontweight="bold")

        # Row label on the leftmost panel.
        axes[r, 0].text(
            -0.05,
            0.5,
            f"{spec.title}\n[{spec.unit}]",
            transform=axes[r, 0].transAxes,
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

        # Per-row colorbars: one for main (col 0 band), one for diff (col 2).
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        sm_main = cm.ScalarMappable(
            norm=mcolors.Normalize(vmin=vmin_main, vmax=vmax_main),
            cmap=spec.cmap_main,
        )
        sm_diff = cm.ScalarMappable(
            norm=mcolors.Normalize(vmin=vmin_diff, vmax=vmax_diff),
            cmap="RdBu_r",
        )
        sm_main.set_array([])
        sm_diff.set_array([])
        cb_main = fig.colorbar(
            sm_main,
            ax=axes[r, :2].tolist(),
            orientation="vertical",
            shrink=0.85,
            pad=0.01,
            aspect=15,
        )
        cb_main.ax.tick_params(labelsize=8)
        cb_diff = fig.colorbar(
            sm_diff,
            ax=axes[r, 2],
            orientation="vertical",
            shrink=0.85,
            pad=0.01,
            aspect=15,
        )
        cb_diff.ax.tick_params(labelsize=8)

        print(
            f"  row {r:2d} {spec.row_id:<20s} "
            f"pred[{float(pred.min()):+.3g},{float(pred.max()):+.3g}] "
            f"truth[{float(truth.min()):+.3g},{float(truth.max()):+.3g}] "
            f"diff[{float(diff.min()):+.3g},{float(diff.max()):+.3g}]"
        )

    fig.suptitle(
        f"frame {frame}  (lead {lead_h:g} h, coarsen={coarsen})",
        fontsize=14,
        fontweight="bold",
    )

    stem = f"frame{frame:03d}"
    out_png = out_dir / f"{stem}.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_png}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred-zarr", required=True)
    p.add_argument("--truth-zarr", required=True)
    p.add_argument(
        "--frames",
        default="35,71,108,143",
        help="Comma-separated 0-based step indices.",
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
    p.add_argument("--step-minutes", type=float, default=10.0)
    p.add_argument("--out-dir", required=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    # load_pred_faces / load_truth_faces allocate (NSIDE // coarsen) per
    # axis and then assign slab[::coarsen, ::coarsen] into it; this only
    # matches when coarsen evenly divides NSIDE. Catch the mismatch with a
    # clear error instead of a downstream shape-assignment failure.
    if NSIDE % args.coarsen != 0:
        raise ValueError(
            f"--coarsen={args.coarsen} must divide NSIDE={NSIDE} evenly "
            f"(got NSIDE % coarsen = {NSIDE % args.coarsen})"
        )
    frames = [int(x) for x in args.frames.split(",") if x.strip()]
    specs = build_specs()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Pred zarr  : {args.pred_zarr}")
    print(f"Truth zarr : {args.truth_zarr}")
    print(f"Frames     : {frames}")
    print(f"Rows       : {[s.row_id for s in specs]}")
    print(
        f"Coarsen    : {args.coarsen}  ({NSIDE}/{args.coarsen}"
        f"={NSIDE // args.coarsen} per face)"
    )
    print(f"Out dir    : {out_dir}")

    pred_group = zarr.open_group(args.pred_zarr, mode="r")
    pred_level_arr = np.asarray(pred_group["level"])
    truth_group = zarr.open_group(args.truth_zarr, mode="r")

    lat, lon = load_coarsened_latlon(pred_group, args.coarsen)
    print(f"Lat/Lon loaded: {lat.shape}")

    for frame in frames:
        print(f"\n=== Frame {frame} ===")
        render_one_frame(
            frame=frame,
            specs=specs,
            pred_group=pred_group,
            truth_group=truth_group,
            initial_frame=args.initial_frame,
            pred_level_arr=pred_level_arr,
            lat=lat,
            lon=lon,
            coarsen=args.coarsen,
            out_dir=out_dir,
            step_minutes=args.step_minutes,
        )


if __name__ == "__main__":
    main()
