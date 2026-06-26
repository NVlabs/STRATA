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
"""T_2m 18-tile zoom for a single zarr rollout.

Shows ``T_2m`` at a fixed step across 6 cubesphere faces and 3 tile
centers per face ((512,512), (1024,1024), (1536,1536)) in a single
6-row × 3-column figure. Only the forecast is plotted (no truth, no
other experiments), so it's a fast sanity check for face-local patchy
artifacts.

Usage::

    python scripts/zarr_dataset_report/plot_t2m_zoom_18tiles.py \\
        --pred-zarr /.../forecast_24hr.zarr \\
        --out-dir /.../t2m_zoom
"""
# ruff: noqa: E402  (matplotlib.use("Agg") must run before pyplot import)
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr

NSIDE = 2048
TILE_SIZE = 128
CENTERS = [512, 1024, 1536]
VAR = "T_2m"
CMAP = "RdBu_r"
UNIT = "K"


def crop_centered(face2d: np.ndarray, cy: int, cx: int, size: int) -> np.ndarray:
    """Return a ``size x size`` crop centered at ``(cy, cx)``, clamped to bounds."""
    h, w = face2d.shape
    half = size // 2
    y0 = max(0, min(h - size, cy - half))
    x0 = max(0, min(w - size, cx - half))
    return face2d[y0 : y0 + size, x0 : x0 + size]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred-zarr", required=True)
    p.add_argument(
        "--step",
        type=int,
        default=71,
        help="0-based index into the pred zarr's step dim.",
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument("--step-minutes", type=float, default=10.0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    g = zarr.open_group(args.pred_zarr, mode="r")
    if VAR not in g.array_keys():
        raise KeyError(f"'{VAR}' not in pred zarr; available: {sorted(g.array_keys())}")
    arr = g[VAR]
    if arr.ndim != 5:
        raise ValueError(
            f"Expected {VAR} to have 5 dims (time, step, face, x, y); "
            f"got shape {arr.shape}"
        )
    n_faces = int(arr.shape[2])
    n_steps = int(arr.shape[1])
    if not 0 <= args.step < n_steps:
        raise ValueError(
            f"--step={args.step} is out of range for {VAR} "
            f"(valid: 0..{n_steps - 1})"
        )

    lead_h = (args.step + 1) * args.step_minutes / 60.0
    print(f"Pred zarr : {args.pred_zarr}")
    print(f"Variable  : {VAR} shape={arr.shape}")
    print(f"Step      : {args.step}  (lead {lead_h:g} h)")
    print(f"Tiles     : {TILE_SIZE}x{TILE_SIZE} at centers {CENTERS}")

    # Load each face once (6 reads) and crop three tiles per face.
    tiles = np.empty((n_faces, len(CENTERS), TILE_SIZE, TILE_SIZE), dtype=np.float32)
    for f in range(n_faces):
        face2d = np.asarray(arr[0, args.step, f, :, :]).astype(np.float32)
        for c_idx, c in enumerate(CENTERS):
            tiles[f, c_idx] = crop_centered(face2d, c, c, TILE_SIZE)
        print(
            f"  face {f}: {face2d.shape} -> 3 tiles "
            f"(min/max: {float(face2d.min()):.2f} / {float(face2d.max()):.2f})"
        )

    # Per-panel color ranges (robust 1/99% quantiles) so tile-local
    # features aren't washed out by large inter-panel T2m contrasts.
    n_rows, n_cols = n_faces, len(CENTERS)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.0 * n_cols + 0.5, 2.6 * n_rows + 0.8),
        squeeze=False,
        constrained_layout=True,
    )
    for f in range(n_faces):
        for c_idx, c in enumerate(CENTERS):
            ax = axes[f, c_idx]
            tile = tiles[f, c_idx]
            vmin = float(np.nanquantile(tile, 0.01))
            vmax = float(np.nanquantile(tile, 0.99))
            if vmin == vmax:
                vmin -= 0.5
                vmax += 0.5
            im = ax.imshow(
                tile,
                cmap=CMAP,
                vmin=vmin,
                vmax=vmax,
                origin="lower",
                aspect="equal",
                interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if f == 0:
                ax.set_title(f"center ({c},{c})", fontsize=12, fontweight="bold")
            if c_idx == 0:
                ax.text(
                    -0.18,
                    0.5,
                    f"Face {f}",
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    rotation=90,
                )
            cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02, aspect=15)
            cb.ax.tick_params(labelsize=8)
            cb.ax.set_title(UNIT, fontsize=8, pad=2)

    fig.suptitle(
        f"{VAR}  step={args.step}  (lead {lead_h:g} h)",
        fontsize=13,
        fontweight="bold",
    )

    stem = f"step{args.step:03d}"
    for ext in ("png", "pdf"):
        out = out_dir / f"{stem}.{ext}"
        save_kw = dict(bbox_inches="tight", facecolor="white")
        if ext == "png":
            save_kw["dpi"] = 180
        fig.savefig(out, **save_kw)
        print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
