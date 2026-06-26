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
"""Longitude-vs-forecast-time hovmöller for cubesphere rollouts.

Supports two prediction input layouts on the ne1024pg2 cubesphere grid:

  1. ACE rollout zarr (``--pred-zarr``): dims ``(time, step, face, x, y)``
     with per-face ``lat``/``lon`` coords, written by
     ``scripts/ace/run_screamcast_nudged.py`` / ``screamcast.zarr_writer``.
  2. Unstructured NetCDF (``--pred-nc``): dims ``(time, pixels)`` with
     ``pixels = 6 * nside * nside`` in the SCREAM 1-D ``cubesphere_faces_2d``
     ordering (same ordering as the SCREAM truth zarr).  Requires
     ``--scrip-path`` so we can area-weight the longitude binning.

In both cases the truth panel is read from the SCREAM cubesphere zarr
(``--truth-zarr``) when provided.  The ``truth`` variable that lives
alongside ``prediction`` inside ``output_rollout/.../*_surf.nc`` is
intentionally ignored because it is typically sentinel-filled.

Examples::

    # ACE forecast_24hr.zarr (face/x/y layout)
    python plots/plot_precip_hovmoller_ace.py \\
        --pred-zarr /.../forecast_24hr.zarr \\
        --truth-zarr /.../sdecadal.ne1024pg2..out10min.cubesphere.zarr \\
        --initial-frame 1729 --start-step 0 --n-steps 144 \\
        --lat-min -5 --lat-max 5 --lon-min 30 --lon-max 180 --lon-res 1.0 \\
        --out-dir plots --out-name precip_hovmoller_ace

    # output_rollout precip_liq_surf_mass_flux_surf.nc (1D unstructured)
    python plots/plot_precip_hovmoller_ace.py \\
        --pred-nc /.../output_rollout/.../precip_liq_surf_mass_flux_surf.nc \\
        --scrip-path /.../ne1024pg2_scrip.nc \\
        --truth-zarr /.../sdecadal.ne1024pg2..out10min.cubesphere.zarr \\
        --initial-frame 1729 --start-step 0 --n-steps 144 \\
        --lat-min -5 --lat-max 5 --lon-min 30 --lon-max 180 --lon-res 1.0 \\
        --out-dir plots --out-name precip_hovmoller_nc
"""
# ruff: noqa: E402  (matplotlib.use("Agg") must run before pyplot import)
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import zarr

# Match the rest of the repo: raw mass-flux-per-area (kg m^-2 s^-1 ≡ m s^-1
# of water depth) is scaled by 1000 * 86400 to get mm day^-1.
MS_TO_MM_DAY = 1000.0 * 86400.0


@dataclass(frozen=True)
class FacePlan:
    """Pre-computed selection + lon-binning for one cubesphere face.

    Operates on full ``(nside, nside)`` face slabs so a single face read
    per step can be binned into multiple regions. ``mask_full`` is the
    lat-band + lon-range mask on the full face; ``bin_idx`` lists the
    1-D longitude bin of every pixel that survives the mask.
    """

    mask_full: np.ndarray
    bin_idx: np.ndarray


def build_face_plans(
    pred_ds: xr.Dataset,
    lat_band: tuple[float, float],
    lon_edges: np.ndarray,
) -> list[FacePlan | None]:
    """Pre-compute the per-face lat-band mask + longitude bin indices.

    This runs once per region; each step of the hovmöller can then just
    read a full face slab and do a single ``np.bincount``.
    """
    n_faces = int(pred_ds.sizes["face"])
    nlon = len(lon_edges) - 1
    plans: list[FacePlan | None] = []
    for f in range(n_faces):
        lat_f = pred_ds["lat"].isel(face=f).values
        lon_f = pred_ds["lon"].isel(face=f).values % 360.0
        in_band = (lat_f >= lat_band[0]) & (lat_f <= lat_band[1])
        if not in_band.any():
            plans.append(None)
            continue
        bin_idx_full = np.digitize(lon_f, lon_edges) - 1
        in_lon = (bin_idx_full >= 0) & (bin_idx_full < nlon)
        mask_full = in_band & in_lon
        if not mask_full.any():
            plans.append(None)
            continue
        bin_idx = bin_idx_full[mask_full].astype(np.intp)
        plans.append(FacePlan(mask_full=mask_full, bin_idx=bin_idx))
    return plans


def bin_faces_to_lon(
    face_full_data: list[np.ndarray | None],
    plans: list[FacePlan | None],
    nlon: int,
) -> np.ndarray:
    """Average a step's full face slabs into the longitude grid."""
    num = np.zeros(nlon, dtype=np.float64)
    den = np.zeros(nlon, dtype=np.float64)
    for f, plan in enumerate(plans):
        if plan is None:
            continue
        d = face_full_data[f]
        if d is None:
            continue
        vals = d[plan.mask_full]
        num += np.bincount(plan.bin_idx, weights=vals, minlength=nlon)
        den += np.bincount(plan.bin_idx, minlength=nlon)
    out = np.full(nlon, np.nan, dtype=np.float64)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


@dataclass(frozen=True)
class ScripPlan:
    """Pre-computed area-weighted longitude binning for a 1D unstructured
    field laid out in the SCREAM ``cubesphere_faces_2d`` ordering.

    ``valid`` is a bool mask over the full 1-D pixel axis selecting every
    pixel that sits inside the latitude band and the longitude range;
    ``flat`` gives the target longitude-bin index for each of those
    pixels; ``w`` is the per-pixel area weight; ``w_sum`` is the total
    weight per bin (so binning one step is a single ``bincount``).
    """

    valid: np.ndarray
    flat: np.ndarray
    w: np.ndarray
    w_sum: np.ndarray


def build_scrip_plan(
    scrip_path: str,
    lat_band: tuple[float, float],
    lon_edges: np.ndarray,
) -> ScripPlan:
    """Build a ScripPlan from an ne1024pg2 SCRIP file."""
    ds = xr.open_dataset(scrip_path)
    lat = ds["grid_center_lat"].values.astype(np.float64)
    lon = ds["grid_center_lon"].values.astype(np.float64) % 360.0
    area = ds["grid_area"].values.astype(np.float64)
    ds.close()

    in_band = (lat >= lat_band[0]) & (lat <= lat_band[1])
    nlon = len(lon_edges) - 1
    lon_idx = np.digitize(lon, lon_edges) - 1
    valid = in_band & (lon_idx >= 0) & (lon_idx < nlon)

    flat = lon_idx[valid].astype(np.intp)
    w = area[valid]
    w_sum = np.bincount(flat, weights=w, minlength=nlon)
    return ScripPlan(valid=valid, flat=flat, w=w, w_sum=w_sum)


def bin_unstructured_to_lon(
    field_1d: np.ndarray, plan: ScripPlan, nlon: int
) -> np.ndarray:
    """Area-weighted longitude bin average for one 1D unstructured frame."""
    d = field_1d[plan.valid].astype(np.float64)
    s = np.bincount(plan.flat, weights=d * plan.w, minlength=nlon)
    out = np.full(nlon, np.nan, dtype=np.float64)
    ok = plan.w_sum > 0
    out[ok] = s[ok] / plan.w_sum[ok]
    return out


def face_1d_to_2d(arr1d: np.ndarray, nside: int) -> np.ndarray:
    """Un-flatten one cubesphere face from the SCREAM ``cubesphere_faces_2d``
    pixel order into a 2D ``(nside, nside)`` image.

    Matches the reshape used by ``local_notebooks/visualize_tile_zoom.py``
    and by ``screamcast.zarr_writer`` when converting the 1-D truth layout
    to the (face, x, y) layout.
    """
    ne = nside // 2
    npg = 2
    return arr1d.reshape(ne, ne, npg, npg).transpose(0, 2, 1, 3).reshape(nside, nside)


@dataclass(frozen=True)
class Region:
    """A named lat-band + longitude-range request for one hovmöller panel."""

    name: str
    lat_band: tuple[float, float]
    lon_range: tuple[float, float]
    lon_res: float


def compute_hovmoller(
    pred_zarr_path: str | None,
    pred_nc_path: str | None,
    scrip_path: str | None,
    truth_zarr_path: str | None,
    pred_var: str,
    truth_var: str,
    start_step: int,
    n_steps: int,
    initial_frame: int,
    lat_band: tuple[float, float],
    lon_range: tuple[float, float],
    lon_res: float,
    step_minutes: float,
    progress: bool,
) -> dict:
    """Single-region convenience wrapper around ``compute_hovmoller_multi``."""
    results = compute_hovmoller_multi(
        pred_zarr_path=pred_zarr_path,
        pred_nc_path=pred_nc_path,
        scrip_path=scrip_path,
        truth_zarr_path=truth_zarr_path,
        pred_var=pred_var,
        truth_var=truth_var,
        start_step=start_step,
        n_steps=n_steps,
        initial_frame=initial_frame,
        regions=[
            Region(
                name="default", lat_band=lat_band, lon_range=lon_range, lon_res=lon_res
            ),
        ],
        step_minutes=step_minutes,
        progress=progress,
    )
    return results["default"]


def compute_hovmoller_multi(
    pred_zarr_path: str | None,
    pred_nc_path: str | None,
    scrip_path: str | None,
    truth_zarr_path: str | None,
    pred_var: str,
    truth_var: str,
    start_step: int,
    n_steps: int,
    initial_frame: int,
    regions: list["Region"],
    step_minutes: float,
    progress: bool,
) -> dict[str, dict]:
    """Compute hovmöller arrays for every region in one data pass.

    Pred / truth I/O is the dominant cost, so we read each step's full
    face slabs (zarr) or 1-D pixel vector (NC) once and then bin into
    every region. Returns ``{region.name: result_dict}`` where each
    result_dict is schema-compatible with the single-region
    ``compute_hovmoller`` return value.
    """
    if (pred_zarr_path is None) == (pred_nc_path is None):
        raise ValueError(
            "Exactly one of pred_zarr_path or pred_nc_path must be provided"
        )
    if len(regions) == 0:
        raise ValueError("At least one region is required")
    seen = set()
    for r in regions:
        if r.name in seen:
            raise ValueError(f"Duplicate region name: {r.name!r}")
        seen.add(r.name)

    if pred_nc_path is not None:
        if scrip_path is None:
            raise ValueError("scrip_path is required when using an NC prediction input")
        return _compute_hovmoller_multi_nc(
            pred_nc_path=pred_nc_path,
            scrip_path=scrip_path,
            truth_zarr_path=truth_zarr_path,
            pred_var=pred_var,
            truth_var=truth_var,
            start_step=start_step,
            n_steps=n_steps,
            initial_frame=initial_frame,
            regions=regions,
            step_minutes=step_minutes,
            progress=progress,
        )
    return _compute_hovmoller_multi_zarr(
        pred_zarr_path=pred_zarr_path,
        truth_zarr_path=truth_zarr_path,
        pred_var=pred_var,
        truth_var=truth_var,
        start_step=start_step,
        n_steps=n_steps,
        initial_frame=initial_frame,
        regions=regions,
        step_minutes=step_minutes,
        progress=progress,
    )


def _compute_hovmoller_multi_zarr(
    pred_zarr_path: str,
    truth_zarr_path: str | None,
    pred_var: str,
    truth_var: str,
    start_step: int,
    n_steps: int,
    initial_frame: int,
    regions: list[Region],
    step_minutes: float,
    progress: bool,
) -> dict[str, dict]:
    """Multi-region Zarr (face, x, y) prediction path.

    Reads each step's six face slabs exactly once and bins into every
    region, so adding an extra region only costs a few bincount calls.
    """
    pred_ds = xr.open_zarr(pred_zarr_path, consolidated=True)
    # Mirror the cleanup pattern in _compute_hovmoller_multi_nc: wrap the
    # whole body in try/finally so pred_ds is released on any exception
    # path. Note on zarr v3: a zarr.Group (what zarr.open_group returns
    # below) has no .close() method — it's a thin handle around a
    # LocalStore that holds no persistent OS file descriptors across
    # reads, so truth_group needs no explicit cleanup.
    try:
        if pred_var not in pred_ds.data_vars:
            raise ValueError(
                f"'{pred_var}' not found in prediction zarr {pred_zarr_path}"
            )

        n_steps_total = int(pred_ds.sizes.get("step", 1))
        if start_step < 0:
            raise ValueError("--start-step must be >= 0")
        if start_step >= n_steps_total:
            raise ValueError(
                f"--start-step {start_step} >= available steps {n_steps_total}"
            )
        n_steps_eff = min(n_steps, n_steps_total - start_step)
        if n_steps_eff <= 0:
            raise ValueError("No forecast steps in the requested window")

        nside = int(pred_ds.sizes["x"])
        n_faces = int(pred_ds.sizes["face"])

        print(f"Prediction: {pred_zarr_path}")
        print(f"  dims: {dict(pred_ds.sizes)}")
        print(f"  variable: {pred_var}")
        print(f"Steps: [{start_step}, {start_step + n_steps_eff}) ({n_steps_eff})")

        # Build a plan + longitude axis per region.
        region_data: list[dict] = []
        for r in regions:
            lon_edges = np.arange(
                r.lon_range[0], r.lon_range[1] + 0.5 * r.lon_res, r.lon_res
            )
            lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
            nlon = int(lon_centers.size)
            if nlon == 0:
                raise ValueError(f"Region {r.name!r}: lon range produced 0 bins")
            plans = build_face_plans(pred_ds, r.lat_band, lon_edges)
            n_active = sum(p is not None for p in plans)
            print(
                f"  region {r.name:<16s} lat [{r.lat_band[0]},{r.lat_band[1]}] "
                f"lon [{r.lon_range[0]},{r.lon_range[1]}] dx={r.lon_res} -> "
                f"{nlon} bins, active faces {n_active}/{n_faces}"
            )
            region_data.append(
                {
                    "region": r,
                    "plans": plans,
                    "lon_edges": lon_edges,
                    "lon_centers": lon_centers,
                    "nlon": nlon,
                    "hov_pred": np.full((n_steps_eff, nlon), np.nan, dtype=np.float64),
                    "hov_truth": None,  # filled in below if truth_avail
                }
            )

        # Union set of active faces across all regions -> the ones we have
        # to load each step. For global regions this is all 6.
        active_faces: list[int] = sorted(
            {
                f
                for rd in region_data
                for f, plan in enumerate(rd["plans"])
                if plan is not None
            }
        )
        print(f"Active faces (union across regions): {active_faces}")

        truth_avail = False
        truth_arr: zarr.Array | None = None
        if truth_zarr_path is not None:
            truth_group = zarr.open_group(truth_zarr_path, mode="r")
            if truth_var in truth_group:
                truth_avail = True
                truth_arr = truth_group[truth_var]
                print(f"Truth: {truth_zarr_path}")
                print(f"  variable: {truth_var}, shape: {truth_arr.shape}")
                for rd in region_data:
                    rd["hov_truth"] = np.full(
                        (n_steps_eff, rd["nlon"]), np.nan, dtype=np.float64
                    )
            else:
                print(
                    f"Truth variable '{truth_var}' missing from "
                    f"{truth_zarr_path} — drawing prediction only"
                )

        pred_da = pred_ds[pred_var]

        t_start = time.time()
        for k in range(n_steps_eff):
            step = start_step + k
            pred_faces: list[np.ndarray | None] = [None] * n_faces
            for f in active_faces:
                pred_faces[f] = pred_da.isel(time=0, step=step, face=f).values.astype(
                    np.float32
                )
            for rd in region_data:
                rd["hov_pred"][k] = bin_faces_to_lon(
                    pred_faces, rd["plans"], rd["nlon"]
                )

            if truth_avail and truth_arr is not None:
                frame = initial_frame + step
                truth_1d = np.asarray(truth_arr[frame, :]).astype(np.float32)
                truth_faces: list[np.ndarray | None] = [None] * n_faces
                for f in active_faces:
                    truth_faces[f] = face_1d_to_2d(
                        truth_1d[f * nside * nside : (f + 1) * nside * nside],
                        nside,
                    )
                for rd in region_data:
                    rd["hov_truth"][k] = bin_faces_to_lon(
                        truth_faces, rd["plans"], rd["nlon"]
                    )

            if progress and (k + 1) % max(1, n_steps_eff // 10) == 0:
                elapsed = time.time() - t_start
                print(f"  step {k + 1:4d}/{n_steps_eff}  elapsed {elapsed:6.1f}s")

        time_hours = (np.arange(n_steps_eff) + 0.5) * step_minutes / 60.0

        out: dict[str, dict] = {}
        for rd in region_data:
            rd["hov_pred"] *= MS_TO_MM_DAY
            if rd["hov_truth"] is not None:
                rd["hov_truth"] *= MS_TO_MM_DAY
            r = rd["region"]
            out[r.name] = {
                "hov_pred": rd["hov_pred"],
                "hov_truth": rd["hov_truth"],
                "lon_centers": rd["lon_centers"],
                "lon_edges": rd["lon_edges"],
                "lon_range": r.lon_range,
                "lat_band": r.lat_band,
                "start_step": start_step,
                "n_steps": n_steps_eff,
                "time_hours": time_hours,
                "step_minutes": step_minutes,
                "truth_available": truth_avail,
            }
        return out
    finally:
        pred_ds.close()


def _compute_hovmoller_multi_nc(
    pred_nc_path: str,
    scrip_path: str,
    truth_zarr_path: str | None,
    pred_var: str,
    truth_var: str,
    start_step: int,
    n_steps: int,
    initial_frame: int,
    regions: list[Region],
    step_minutes: float,
    progress: bool,
) -> dict[str, dict]:
    """Multi-region unstructured NetCDF prediction path with SCRIP regrid.

    Truth is always read from ``truth_zarr_path`` (when provided).  The
    ``truth`` variable that some rollout NCs carry alongside ``prediction``
    is intentionally ignored: in the rollouts under ``output_rollout/`` it
    is often just a sentinel fill (``9.969e+36``), so matching
    ``plot_precip_hovmoller.py`` we go straight to the SCREAM truth zarr.
    """
    pred_ds = xr.open_dataset(pred_nc_path)

    # The NC files in output_rollout/ store ("prediction", "truth") indexed
    # by ("time", "pixels").  Accept either pred_var (default
    # precip_liq_surf_mass_flux) or the literal "prediction" name.
    pred_name = "prediction" if "prediction" in pred_ds.data_vars else pred_var
    if pred_name not in pred_ds.data_vars:
        raise ValueError(
            f"Prediction variable not found in {pred_nc_path}; "
            f"looked for 'prediction' and '{pred_var}'. "
            f"Available: {list(pred_ds.data_vars)}"
        )
    pred_da = pred_ds[pred_name]

    n_times_total = int(pred_da.sizes["time"])
    if start_step < 0:
        raise ValueError("--start-step must be >= 0")
    if start_step >= n_times_total:
        raise ValueError(
            f"--start-step {start_step} >= available times {n_times_total}"
        )
    n_steps_eff = min(n_steps, n_times_total - start_step)
    if n_steps_eff <= 0:
        raise ValueError("No forecast steps in the requested window")

    n_pixels_total = int(pred_da.sizes["pixels"])

    print(f"Prediction: {pred_nc_path}")
    print(f"  dims: {dict(pred_da.sizes)}")
    print(f"  variable: {pred_name}")
    print(f"SCRIP: {scrip_path}")
    print(f"Steps: [{start_step}, {start_step + n_steps_eff}) ({n_steps_eff})")

    # One ScripPlan per region (share pred_ds + truth zarr IO).
    region_data: list[dict] = []
    for r in regions:
        lon_edges = np.arange(
            r.lon_range[0], r.lon_range[1] + 0.5 * r.lon_res, r.lon_res
        )
        lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
        nlon = int(lon_centers.size)
        if nlon == 0:
            raise ValueError(f"Region {r.name!r}: lon range produced 0 bins")
        plan = build_scrip_plan(scrip_path, r.lat_band, lon_edges)
        if plan.valid.size != n_pixels_total:
            raise ValueError(
                f"SCRIP grid has {plan.valid.size} cells but prediction has "
                f"{n_pixels_total} pixels — grids do not match"
            )
        print(
            f"  region {r.name:<16s} lat [{r.lat_band[0]},{r.lat_band[1]}] "
            f"lon [{r.lon_range[0]},{r.lon_range[1]}] dx={r.lon_res} -> "
            f"{nlon} bins, active pixels {int(plan.valid.sum())}"
        )
        region_data.append(
            {
                "region": r,
                "plan": plan,
                "lon_edges": lon_edges,
                "lon_centers": lon_centers,
                "nlon": nlon,
                "hov_pred": np.full((n_steps_eff, nlon), np.nan, dtype=np.float64),
                "hov_truth": None,
            }
        )

    truth_zarr_arr: zarr.Array | None = None
    if truth_zarr_path is not None:
        truth_group = zarr.open_group(truth_zarr_path, mode="r")
        if truth_var in truth_group:
            truth_zarr_arr = truth_group[truth_var]
            print(f"Truth: {truth_zarr_path}")
            print(f"  variable: {truth_var}, shape: {truth_zarr_arr.shape}")
            for rd in region_data:
                rd["hov_truth"] = np.full(
                    (n_steps_eff, rd["nlon"]), np.nan, dtype=np.float64
                )
        else:
            print(
                f"Truth variable '{truth_var}' missing from "
                f"{truth_zarr_path} — drawing prediction only"
            )

    truth_avail = truth_zarr_arr is not None

    t_start = time.time()
    for k in range(n_steps_eff):
        step = start_step + k
        pred_1d = pred_da.isel(time=step).values.astype(np.float64)
        for rd in region_data:
            rd["hov_pred"][k] = bin_unstructured_to_lon(pred_1d, rd["plan"], rd["nlon"])

        if truth_avail and truth_zarr_arr is not None:
            truth_1d = np.asarray(truth_zarr_arr[initial_frame + step, :]).astype(
                np.float64
            )
            for rd in region_data:
                rd["hov_truth"][k] = bin_unstructured_to_lon(
                    truth_1d, rd["plan"], rd["nlon"]
                )

        if progress and (k + 1) % max(1, n_steps_eff // 10) == 0:
            elapsed = time.time() - t_start
            print(f"  step {k + 1:4d}/{n_steps_eff}  elapsed {elapsed:6.1f}s")

    pred_ds.close()

    time_hours = (np.arange(n_steps_eff) + 0.5) * step_minutes / 60.0

    out: dict[str, dict] = {}
    for rd in region_data:
        rd["hov_pred"] *= MS_TO_MM_DAY
        if rd["hov_truth"] is not None:
            rd["hov_truth"] *= MS_TO_MM_DAY
        r = rd["region"]
        out[r.name] = {
            "hov_pred": rd["hov_pred"],
            "hov_truth": rd["hov_truth"],
            "lon_centers": rd["lon_centers"],
            "lon_edges": rd["lon_edges"],
            "lon_range": r.lon_range,
            "lat_band": r.lat_band,
            "start_step": start_step,
            "n_steps": n_steps_eff,
            "time_hours": time_hours,
            "step_minutes": step_minutes,
            "truth_available": truth_avail,
        }
    return out


def plot_hovmoller(
    data: dict,
    cmap: str,
    vmax_quantile: float,
    out_dir: Path,
    out_name: str,
    pred_label: str,
    truth_label: str,
) -> list[Path]:
    """Render the hovmöller panels to ``out_dir / {out_name}.png|.pdf``."""
    hov_pred: np.ndarray = data["hov_pred"]
    hov_truth: np.ndarray | None = data["hov_truth"]
    lon_range: tuple[float, float] = data["lon_range"]
    time_hours: np.ndarray = data["time_hours"]
    truth_avail: bool = data["truth_available"]

    all_vals_parts = [hov_pred[np.isfinite(hov_pred)]]
    if truth_avail and hov_truth is not None:
        all_vals_parts.append(hov_truth[np.isfinite(hov_truth)])
    concat = (
        np.concatenate(all_vals_parts)
        if all(a.size for a in all_vals_parts)
        else np.array([0.0, 1.0])
    )
    vmax = float(np.quantile(concat, vmax_quantile)) if concat.size else 1.0
    if vmax <= 0:
        vmax = 1.0
    vmin = 0.0
    print(f"Color range: {vmin}..{vmax:.3f} mm/day (q={vmax_quantile})")

    if time_hours.size > 1:
        dy = 0.5 * (time_hours[1] - time_hours[0])
        y_lo = float(time_hours[0] - dy)
        y_hi = float(time_hours[-1] + dy)
    else:
        y_lo = float(time_hours[0])
        y_hi = float(time_hours[0]) + 1.0

    extent = [lon_range[0], lon_range[1], y_lo, y_hi]
    common = dict(
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        extent=extent,
    )

    n_rows = 2 if truth_avail else 1
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(14, (3.2 * n_rows + 0.8) * 0.7),
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes[:, 0]

    row = 0
    im_ref = None
    if truth_avail and hov_truth is not None:
        im_ref = axes[row].imshow(hov_truth, **common)
        axes[row].set_title(
            truth_label,
            loc="left",
            fontsize=18,
            fontweight="bold",
        )
        row += 1
    im_pred = axes[row].imshow(hov_pred, **common)
    axes[row].set_title(
        pred_label,
        loc="left",
        fontsize=18,
        fontweight="bold",
    )
    if im_ref is None:
        im_ref = im_pred

    axes[-1].set_xlabel("Longitude (°E)", fontsize=16.5)
    for ax in axes:
        ax.set_ylabel("Lead time (hours)", fontsize=16.5)
        ax.tick_params(labelsize=15)

    cbar = fig.colorbar(im_ref, ax=list(axes), shrink=0.9, pad=0.02)
    cbar.set_label("mm/day", fontsize=16.5)
    cbar.ax.tick_params(labelsize=15)

    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for ext in ("png", "pdf"):
        out_path = out_dir / f"{out_name}.{ext}"
        save_kw = {"bbox_inches": "tight", "facecolor": "white"}
        if ext == "png":
            save_kw["dpi"] = 200
        fig.savefig(out_path, **save_kw)
        saved.append(out_path)
        print(f"Saved: {out_path}")
    plt.close(fig)
    return saved


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    pred_group = p.add_mutually_exclusive_group(required=True)
    pred_group.add_argument(
        "--pred-zarr",
        default=None,
        help="Path to a prediction cubesphere zarr (dims "
        "time/step/face/x/y with per-face lat/lon coords), e.g. "
        "forecast_24hr.zarr.",
    )
    pred_group.add_argument(
        "--pred-nc",
        default=None,
        help="Path to a prediction NetCDF with dims (time, pixels) in the "
        "SCREAM 1-D cubesphere_faces_2d ordering, e.g. "
        "output_rollout/.../precip_liq_surf_mass_flux_surf.nc. Requires "
        "--scrip-path.",
    )
    p.add_argument(
        "--scrip-path",
        default=None,
        help="Path to the ne1024pg2 SCRIP file (required with --pred-nc; "
        "ignored with --pred-zarr). Used for area-weighted longitude "
        "binning on the unstructured grid.",
    )
    p.add_argument(
        "--truth-zarr",
        default=None,
        help="Path to the SCREAM truth cubesphere zarr. If omitted, only "
        "the prediction panel is plotted. Note: the 'truth' variable "
        "inside output_rollout/.../*_surf.nc files is sentinel-filled, so "
        "that is NOT used as a fallback.",
    )
    p.add_argument(
        "--pred-var",
        default="precip_liq_surf_mass_flux",
        help="Variable name in the prediction zarr.",
    )
    p.add_argument(
        "--truth-var",
        default=None,
        help="Variable name in the truth zarr. Defaults to --pred-var.",
    )
    p.add_argument(
        "--initial-frame",
        type=int,
        default=1729,
        help="Frame index in the truth zarr that corresponds to forecast "
        "step=0. Default 1729 matches 2020-10-13T00:00 vs the out10min "
        "SCREAM zarr.",
    )
    p.add_argument("--start-step", type=int, default=0)
    p.add_argument("--n-steps", type=int, default=144)
    p.add_argument("--lat-min", type=float, default=-5.0)
    p.add_argument("--lat-max", type=float, default=5.0)
    p.add_argument("--lon-min", type=float, default=30.0)
    p.add_argument("--lon-max", type=float, default=180.0)
    p.add_argument("--lon-res", type=float, default=1.0)
    p.add_argument(
        "--step-minutes",
        type=float,
        default=10.0,
        help="Forecast step interval in minutes (used only for the y-axis " "label).",
    )
    p.add_argument("--cmap", default="YlGnBu_r")
    p.add_argument("--vmax-quantile", type=float, default=0.99)
    p.add_argument(
        "--out-dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Directory for the output PNG / PDF.",
    )
    p.add_argument(
        "--out-name",
        default="precip_hovmoller_ace",
        help="Filename stem (without extension). With --regions, the "
        "region name is appended as '_<region>'.",
    )
    p.add_argument(
        "--pred-label",
        default="SCREAMCAST (24-hour rollout)",
    )
    p.add_argument(
        "--truth-label",
        default="SCREAM (reference)",
    )
    p.add_argument(
        "--regions",
        default=None,
        help="Comma-separated region specs to render in a single pass. "
        "Format: 'name:latmin:latmax:lonmin:lonmax:lonres' per region, "
        "e.g. "
        "'indopacific:-5:5:30:180:1.0,global:-60:60:0:360:2.0'. When set, "
        "--lat-*/--lon-* are ignored and one figure is written per "
        "region. Lets you amortize pred/truth I/O across regions.",
    )
    p.add_argument("--progress", action="store_true")
    return p.parse_args(argv)


def parse_regions_flag(s: str) -> list[Region]:
    """Parse the ``--regions`` CLI string into ``Region`` objects."""
    regions: list[Region] = []
    for entry in s.split(","):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(":")
        if len(parts) != 6:
            raise ValueError(
                f"Bad region spec {entry!r}; want "
                f"'name:latmin:latmax:lonmin:lonmax:lonres'"
            )
        name, lat_lo_s, lat_hi_s, lon_lo_s, lon_hi_s, lon_res_s = parts
        lat_lo = float(lat_lo_s)
        lat_hi = float(lat_hi_s)
        lon_lo = float(lon_lo_s)
        lon_hi = float(lon_hi_s)
        lon_res = float(lon_res_s)
        if not lat_lo < lat_hi:
            raise ValueError(
                f"Bad region {name!r}: latmin ({lat_lo}) must be strictly "
                f"less than latmax ({lat_hi})"
            )
        if not lon_lo < lon_hi:
            raise ValueError(
                f"Bad region {name!r}: lonmin ({lon_lo}) must be strictly "
                f"less than lonmax ({lon_hi})"
            )
        if not lon_res > 0.0:
            raise ValueError(f"Bad region {name!r}: lonres ({lon_res}) must be > 0")
        regions.append(
            Region(
                name=name,
                lat_band=(lat_lo, lat_hi),
                lon_range=(lon_lo, lon_hi),
                lon_res=lon_res,
            )
        )
    if len(regions) == 0:
        raise ValueError("--regions parsed to zero regions")
    return regions


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.pred_nc is not None and args.scrip_path is None:
        raise ValueError("--scrip-path is required when using --pred-nc")

    if args.regions is not None:
        regions = parse_regions_flag(args.regions)
    else:
        if args.lat_min >= args.lat_max:
            raise ValueError("--lat-min must be < --lat-max")
        if args.lon_min >= args.lon_max:
            raise ValueError("--lon-min must be < --lon-max")
        regions = [
            Region(
                name="default",
                lat_band=(args.lat_min, args.lat_max),
                lon_range=(args.lon_min, args.lon_max),
                lon_res=args.lon_res,
            )
        ]

    results = compute_hovmoller_multi(
        pred_zarr_path=args.pred_zarr,
        pred_nc_path=args.pred_nc,
        scrip_path=args.scrip_path,
        truth_zarr_path=args.truth_zarr,
        pred_var=args.pred_var,
        truth_var=args.truth_var or args.pred_var,
        start_step=args.start_step,
        n_steps=args.n_steps,
        initial_frame=args.initial_frame,
        regions=regions,
        step_minutes=args.step_minutes,
        progress=args.progress,
    )
    for r in regions:
        if r.name == "default" or args.out_name == r.name:
            out_name = args.out_name
        else:
            out_name = f"{args.out_name}_{r.name}"
        plot_hovmoller(
            data=results[r.name],
            cmap=args.cmap,
            vmax_quantile=args.vmax_quantile,
            out_dir=Path(args.out_dir),
            out_name=out_name,
            pred_label=args.pred_label,
            truth_label=args.truth_label,
        )


if __name__ == "__main__":
    main()
