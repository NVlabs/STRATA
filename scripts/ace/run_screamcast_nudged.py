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
"""Run a local SCREAMcast forecast with optional nudging correction.

For the distributed tiled rollout, each rank owns one or more interior
``tile_size x tile_size`` tiles. The rollout loop explicitly converts those
interior tiles to a halo-padded layout before each model step, then crops the
model output back to the interior layout afterward.

Halo fill is implemented by:
1. ``all_gather``-ing every rank's interior tiles in a fixed global tile order.
2. Flattening those gathered interiors into a global unstructured source vector.
3. Applying a KNN interpolator from source tile interiors to this rank's local
   padded target grid, which includes the halo ring.

This avoids explicit face-neighbor logic. Interior points map to themselves,
while halo points are populated by interpolation from the globally gathered tile
interiors.

For ``ne1024pg2``, each cubed-sphere face has size ``2048 x 2048`` and is split
into a ``4 x 4`` grid of ``512 x 512`` interior tiles. Tiles are identified by
``(face, i, j)``, where ``face`` is the cubed-sphere face id and ``i``/``j``
index the tile within that face. The global ``tiles`` list is the canonical
tile-major ordering for the rollout: it is used to shard work across ranks,
build the flattened source coordinates for KNN, and interpret the
``all_gather``-ed interiors during halo reconstruction.


Performance note:

This implementation has been tested on 8 and 32 H100s. Timings include
- 8 (within node): 21 seconds/step
- 32: 5.6 seconds/step

Currently it is unknown if the all gather is a bottleneck, but network
card utlization monitored via glances peaks at ~5 Gb/s, and gpu utization
occasionally drops to 80%, probably related to I/O events

"""

from __future__ import annotations

import argparse
import functools
import logging
import os
import sys
import time
import traceback
from collections import OrderedDict

import dotenv
import numpy as np
import torch
import xarray as xr
import zarr
from physicsnemo.distributed import DistributedManager
from scipy.signal.windows import kaiser_bessel_derived

import data_catalog
from screamcast.cubesphere_transforms import faces_to_unstructured
from screamcast.dali_ext_src import unstructured_to_6faces
from screamcast.distributed_halo import (
    DistributedTileKNNHaloPadding_AllGather,
    NormalizedHaloAdjoint,
    TileTopology,
)
from screamcast.distributed_state_fetch import (
    fetch_local_tiles,
    load_full_face_scream_state,
)
from screamcast.earth2studio_wrappers import ScreamcastModel
from screamcast.history import history_entry
from screamcast.horizontal_regridding import (
    LatLonToPointGridRegridder,
    UnstructuredToLatLonRegridder,
)
from screamcast.omega_filter import low_pass
from screamcast.regional import (
    AnchoredBoundary,
    ReplicateRing,
    faces_to_pieces,
    find_box_region,
    parse_latlon_box,
    parse_region,
)
from screamcast.regional_averages import (
    RegionalAverager,
    load_grid_area_face,
    load_landfrac_face,
)
from screamcast.replay import (
    BLOCK as REPLAY_BLOCK,
)
from screamcast.replay import (
    DEFAULT_VARS as REPLAY_DEFAULT_VARS,
)
from screamcast.replay import (
    Replay,
    ReplayStep,
    analysis_hours,
    level_ramp_weights,
    replay_plan,
    run_with_replay,
)
from screamcast.replay_targets import (
    FAILED,
    IC_READY,
    ProducerProcess,
    TargetReader,
    build_ic_store,
    check_target,
    load_spec,
    make_source,
    mark,
    producer_stems,
    producer_tiles,
    read_blocks,
    read_box_blocks,
    ring_hours,
    source_hours,
    target_channels,
    target_origin,
    target_schedule,
    wait_for_ic,
)
from screamcast.replay_targets import (
    MODES as PRODUCER_MODES,
)
from screamcast.sht_omega_filter import DistributedSHTHighpass
from screamcast.tile_jitter import JitteredModel
from screamcast.vertical_interpolation import log_pressure_interpolate
from screamcast.zarr_writer import (
    FaceZarrWriteStep,
    ZarrWriter,
    _split_variable_levels,
    output_frame_index,
    prepare_latlon_plev_store,
    prepare_output_store,
)

dotenv.load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_AUX = os.environ.get("AUX_DATA_ROOT", "data")
GRID_PATH = os.path.join(_AUX, "ne1024pg2_scrip.nc")
GRID_TGT_PATH = os.path.join(_AUX, "ne1024halo256pg2_scrip.nc")
P0_SCREAM = 100000.0

zarr.config.set({"threading.max_workers": 18})


def channel_chunked(channel_chunk: int):
    def decorate(fn):
        if channel_chunk <= 0:
            return fn

        def wrapped(x: torch.Tensor):
            outputs = []
            for start in range(0, x.shape[1], channel_chunk):
                stop = min(start + channel_chunk, x.shape[1])
                outputs.append(fn(x[:, start:stop]))
            return torch.cat(outputs, dim=1)

        wrapped.__wrapped__ = fn
        return wrapped

    return decorate


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Python logging level for rank 0 output.",
    )
    parser.add_argument(
        "--run-name",
        default="pixeldit_sem1024d24l_pix128d4l_2stepft",
        help="SCREAMcast run name used to derive the checkpoint path.",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Optional explicit SCREAMcast checkpoint path. Overrides --run-name.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=512,
        help="Interior tile size for distributed rollout.",
    )
    parser.add_argument(
        "--halo-width",
        type=int,
        default=32,
        help="Halo width around each interior tile for distributed inference.",
    )
    parser.add_argument("--n-steps", type=int, default=36, help="Inner forecast steps.")
    parser.add_argument(
        "--n-steps-outer",
        type=int,
        default=1,
        help="Outer correction loops for data-based nudging.",
    )
    parser.add_argument(
        "--coarsen",
        type=int,
        default=32,
        help="Low-pass coarsening factor for data-based nudging.",
    )
    parser.add_argument(
        "--omega-filter-strength",
        type=float,
        default=0.0,
        help="Subtract a low-pass omega component before each model step.",
    )
    parser.add_argument(
        "--sht-omega-lmax",
        type=int,
        default=0,
        help=(
            "If > 0, apply a global spherical-harmonic high-pass to omega channels "
            "after each model step, removing modes with ell < this value. "
            "Distributed via all-reduce on the SHT coefficients. 0 disables."
        ),
    )
    parser.add_argument(
        "--nudge-only-wind",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply nudging only to U/V channels.",
    )
    parser.add_argument(
        "--correction",
        choices=["ace", "data", "none"],
        default="ace",
        help="Correction mode to apply.",
    )
    parser.add_argument(
        "--forecast-model",
        choices=["screamcast", "persistence"],
        default="screamcast",
        help="Forecast model used during rollout.",
    )
    parser.add_argument(
        "--ace-checkpoint",
        default=(
            "/path/to"
            "/project-data/screamcast/inferences/"
            "pixeldit_sem1024d24l_pix128d4l_2stepft/"
            "sdecadal.ne1024pg2_ne1024pg2.F20TR-SCREAMv1.c10-sep11/"
            "finetune_rev2/ace2scream_sfno_rev2_100ep.pt"
        ),
        help="ACE residual checkpoint path for --correction ace.",
    )
    parser.add_argument(
        "--grid-tgt",
        default=GRID_TGT_PATH,
        help="Path to padded target SCRIP grid .nc file.",
    )
    parser.add_argument("--ace-fetch-channel-chunk", type=int, default=32)
    parser.add_argument(
        "--halo-channel-chunk",
        type=int,
        default=32,
        help=(
            "Optional channel chunk size for distributed halo padding. "
            "Use 0 to disable chunking and gather all channels at once."
        ),
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=1,
        help="Max faces per pipeline.step call. Default 1 processes one face at a time.",
    )
    parser.add_argument(
        "--initial-time",
        default=None,
        help=(
            "Forecast start time; must be in SCREAM_MAIN_ZARR_PATH. Default "
            "2020-10-13T00:00:00, or the store's initial_time with --ic-store."
        ),
    )
    parser.add_argument(
        "--ic-store",
        default=None,
        help=(
            "Start from a one-frame IC store built by screamcast/initial_conditions/ "
            "instead of SCREAM_MAIN_ZARR_PATH. Needs --correction none."
        ),
    )
    parser.add_argument(
        "--ic-onthefly",
        default=None,
        metavar="SPEC.json",
        help=(
            "Build the IC store at start-up instead of giving --ic-store: rank 0 "
            "builds it from SPEC's source (as --driver-onthefly, plus the statics "
            "'latlon' and 'scream_ref') into --driver-target-dir, or keeps one "
            "built there earlier from the same origin. The initial time is "
            "--initial-time, by default the init of an HRES source. Needs "
            "--correction none."
        ),
    )
    region = parser.add_mutually_exclusive_group()
    region.add_argument(
        "--region",
        default=None,
        help=(
            "Regional window: predict only this rectangle of tiles, "
            "'face,i0,i1,j0,j1' in tile units (half-open) on the plane of `face`; "
            "i1 or j1 beyond the face's tiles joins the north polar face or the "
            "next equatorial face to the east, a negative i0 the south polar face "
            "(the '=' form: --region=3,-4,4,5,13). The ring is the tiles that touch "
            "the window, on any face. "
            "`python -m screamcast.regional "
            "--lat .. --lon .. --tiles N --latlon latlon_ne1024pg2.nc` prints one "
            "for a point. Needs "
            "--correction none (see docs/regional_window.md)."
        ),
    )
    region.add_argument(
        "--region-latlon",
        default=None,
        help=(
            "Regional window from a lat/lon box 'LAT0,LAT1,LON0,LON1' (degrees; the "
            "box runs east from LON0 to LON1, so LON0 > LON1 crosses the dateline): "
            "the smallest --region that holds every grid cell of the box with "
            "--region-margin-tiles tiles to spare, lies on one plane and has a "
            "multiple of the world size of tiles. `python -m "
            "screamcast.regional --latlon-box ..` prints the same window. A box "
            "south of the equator needs the '=' form: --region-latlon=-40,-20,140,160."
        ),
    )
    parser.add_argument(
        "--region-margin-tiles",
        type=int,
        default=1,
        help=(
            "With --region-latlon: whole tiles of window kept between the box and "
            "the window edge (the driven ring). Default 1."
        ),
    )
    parser.add_argument(
        "--region-boundary",
        choices=["anchored", "replicate"],
        default="anchored",
        help=(
            "Lateral boundary of --region. 'anchored': the initial state plus the "
            "large-scale change of a driver since the start, "
            "IC + (smooth(D(t)) - smooth(D(t0))), with D from --driver-stores. "
            "'replicate': the window's own edge pixels extended outward (no "
            "outside information)."
        ),
    )
    parser.add_argument(
        "--driver-stores",
        "--boundary-stores",
        dest="driver_stores",
        nargs="+",
        default=None,
        help=(
            "Driver of --region-boundary anchored and of --replay: one-frame stores "
            "built by screamcast/initial_conditions/ (same layout as --ic-store), one "
            "per driver time, e.g. HRES forecast leads 0, 6, ..., 24 h "
            "(screamcast.initial_conditions.hres_forecast). Their initial_time "
            "attributes are the valid times. A missing 0 h driver is taken from the "
            "initial condition itself. Without this option the driver is read from "
            "SCREAM_MAIN_ZARR_PATH every --driver-every-hours (not with --ic-store)."
        ),
    )
    parser.add_argument(
        "--driver-onthefly",
        default=None,
        metavar="SPEC.json",
        help=(
            "Build the driver while the run goes instead of reading --driver-stores: "
            "a producer process on rank 0's GPU builds the 8 x 8 block means of the "
            "driver (ERA5 from ARCO, or an ECMWF HRES forecast from open data or "
            "WeatherBench2) that --replay and an anchored --region ring use, into "
            "--driver-target-dir. SPEC names the source and the fast_ic inputs; see "
            "screamcast/replay_targets.py and docs/replay.md. The initial condition "
            "comes from --ic-store or --ic-onthefly."
        ),
    )
    parser.add_argument(
        "--driver-target-dir",
        default=None,
        help=(
            "--driver-onthefly, --ic-onthefly: directory of the targets and the IC, "
            "on a file system every rank sees. Targets and ICs already there (an "
            "earlier run) are used if they were built from the same source and "
            "fast_ic inputs and cover this run; any other one there stops the run. "
            "The producer's log, FAILED/DONE and the run's markers go to "
            "DIR/runs/<run-name>-<time>-<pid>."
        ),
    )
    parser.add_argument(
        "--driver-producer",
        choices=PRODUCER_MODES,
        default="gated",
        help=(
            "--driver-onthefly: 'gated' builds what the run needs before its first "
            "analysis at once and each later target once the run begins the analysis "
            "before it; 'ahead' builds every target as early as it can; 'serial' "
            "builds each one when the run needs it (the run waits)."
        ),
    )
    parser.add_argument(
        "--driver-target-timeout",
        type=float,
        default=3600.0,
        help=(
            "--driver-onthefly: seconds a rank waits for a target before it fails "
            "(--ic-onthefly: for the IC)."
        ),
    )
    parser.add_argument(
        "--driver-every-hours",
        "--boundary-every-hours",
        dest="driver_every_hours",
        type=int,
        default=6,
        help=(
            "Driver interval of the anchored ring when the driver is read from "
            "SCREAM_MAIN_ZARR_PATH or built --driver-onthefly."
        ),
    )
    parser.add_argument(
        "--boundary-block",
        type=int,
        default=8,
        help=(
            "Block size (px) of the driver smoothing: block means, then bilinear "
            "upsampling, so only the driver's large-scale change enters the ring. "
            "1 uses the driver at full resolution."
        ),
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help=(
            "Replay the run to the driver (--driver-stores): every "
            "--replay-every-hours, the large-scale difference between the driver and "
            "the run is added over the window centred on that time (incremental "
            "analysis update; the run is rewound and re-run over each window). Works "
            "for global runs and --region windows. See docs/replay.md."
        ),
    )
    parser.add_argument(
        "--replay-every-hours",
        type=int,
        default=6,
        help="Analysis interval (even hours); each analysis needs a driver state.",
    )
    parser.add_argument(
        "--replay-ell-half",
        type=float,
        default=160.0,
        help=(
            "Spherical-harmonic degree at which the increment's Gaussian taper has "
            "gain 1/2 (160 ~ 250 km wavelength, 40 ~ 1000 km): smaller keeps more of "
            "the scales STRATA's own."
        ),
    )
    parser.add_argument(
        "--replay-vars",
        default=",".join(REPLAY_DEFAULT_VARS),
        help=(
            "Comma-separated input variables that receive the increment, all levels "
            "of each (strata_v1: any of the default; e.g. leave out omega or T_2m). "
            "Diagnostics such as ps are not inputs and cannot be replayed. The "
            "anchored ring of a window drives every input variable regardless."
        ),
    )
    parser.add_argument(
        "--replay-level-ramp",
        default=None,
        metavar="PFULL,PZERO",
        help=(
            "Replay the free atmosphere only: the increment of each 3D channel is "
            "weighted 1 at reference pressures <= PFULL hPa, 0 at >= PZERO hPa, "
            "linear in pressure between (e.g. 800,900). The reference pressure is "
            "(hyam + hybm) * 1000 hPa of the model levels. Surface channels are not "
            "weighted: leave T_2m out of --replay-vars to leave the boundary layer "
            "free."
        ),
    )
    parser.add_argument(
        "--replay-region-edge",
        choices=("normalized", "zero"),
        default="normalized",
        help=(
            "--region windows: the increment exists on the window's blocks only, so "
            "the low-pass gives cells near the window edge only part of it (half on "
            "the edge). 'normalized' divides by the low-passed window indicator, so "
            "the whole increment reaches the edge; 'zero' keeps the share."
        ),
    )
    parser.add_argument(
        "--output-steps",
        type=int,
        default=1,
        help=(
            "Write every N-th forecast step, at leads N*dt, 2N*dt, ... "
            "(default: 1 = every step). --n-steps must be a multiple of N."
        ),
    )
    parser.add_argument(
        "--output-chunk-size",
        type=int,
        default=1024,
        help="Spatial zarr chunk size for raw output x/y dimensions.",
    )
    parser.add_argument(
        "--output-variables",
        nargs="+",
        default=None,
        metavar="VAR",
        help="Base variable names to write (e.g. U T PRECT). Default: all output variables.",
    )
    parser.add_argument(
        "--output-levels",
        type=lambda s: [int(x) for x in s.split(",")],
        default=None,
        metavar="L0,L1,...",
        help="Comma-separated level indices to write for all 3D variables (e.g. 0,5,10). Default: all levels.",
    )
    parser.add_argument(
        "--write-mode",
        choices=["legacy", "face-zarr"],
        default="face-zarr",
        help="Output writer implementation. 'face-zarr' writes owned variables as full faces.",
    )
    parser.add_argument(
        "--halo-adjoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use NormalizedHaloAdjoint (D⁻¹ Aᵀ) to map model outputs back to "
            "interior tiles instead of a plain crop. Averages overlapping halo "
            "contributions rather than discarding them."
        ),
    )
    parser.add_argument(
        "--halo-window",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Apply a Kaiser-Bessel derived window to each padded tile before the "
            "halo adjoint, downweighting halo contributions relative to interior "
            "predictions. Only used when --halo-adjoint is set."
        ),
    )
    parser.add_argument(
        "--halo-adjoint-cross-face",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Let the halo adjoint project predictions made on the cube-face "
            "extension back onto real grid points. Off by default: those targets "
            "do not coincide with the neighbour face's grid, so the KNN operator "
            "interpolates them and the adjoint mixes a grid-scale-smoothed "
            "estimate into the outermost few pixels of every face (see "
            "docs/halo-adjoint-cross-face.md). The default keeps the overlap-add "
            "at intra-face tile seams and drops it across cube-face edges; "
            "--halo-adjoint-cross-face restores the old behaviour. Only used when "
            "--halo-adjoint is set."
        ),
    )
    parser.add_argument(
        "--tile-jitter",
        type=int,
        default=4,
        help=(
            "Per-step sub-patch jitter in pixels. Every step runs the model on a "
            "(padded-J) x (padded-J) sub-window of each padded tile so the 4-px "
            "patch kink does not accumulate. Default 4 (the patch size). 0 turns "
            "jitter off."
        ),
    )
    parser.add_argument(
        "--tile-jitter-mode",
        choices=["cycle", "random", "cycle16"],
        default="cycle16",
        help=(
            "Offset schedule. cycle16 visits all J*J offsets and cancels both "
            "axis lines and both 2-D corners. cycle is the diagonal schedule and "
            "leaves 45-degree stripes. random has no blind spot but only a "
            "random-walk cancellation."
        ),
    )
    parser.add_argument(
        "--tile-jitter-seed",
        type=int,
        default=0,
        help="RNG seed for --tile-jitter-mode random. The same seed is used on every rank.",
    )
    parser.add_argument(
        "--tile-jitter-fill",
        choices=["replicate", "zero"],
        default="replicate",
        help=(
            "Fill for the un-sampled J-px border. replicate copies the sub-window "
            "edge so the halo adjoint's window weight is not applied to zeros."
        ),
    )
    parser.add_argument(
        "--pressure-levels",
        nargs="+",
        type=float,
        default=None,
        metavar="HPA",
        help=(
            "Target pressure levels in hPa for the extra ACE-latlon output. "
            "When provided, an additional zarr is written alongside the raw "
            "output containing model fields regridded to the ACE lat/lon grid "
            "and vertically interpolated to these pressure levels."
        ),
    )
    parser.add_argument(
        "--ace-plev-output",
        default="",
        help=(
            "Optional explicit output path for the ACE-latlon + pressure-level "
            "zarr. Default: derived from the raw output path by appending "
            "`_ace_plev.zarr`."
        ),
    )
    parser.add_argument(
        "--zarr-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write the per-step raw forecast output to the zarr store. "
            "Pass --no-zarr-output to skip zarr preparation and the per-step "
            "face-gather write entirely (useful for smoketests that only need "
            "regional averages)."
        ),
    )
    parser.add_argument(
        "--regional-averages",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Compute area-weighted regional averages (global, land, ocean, "
            "tropics, extratropics) every step and write them to a netCDF "
            "file alongside the zarr output. Default: on for global runs, off "
            "with --region (a window can miss a whole region, e.g. the "
            "extratropics, and then the averages fail at start-up)."
        ),
    )
    parser.add_argument(
        "--regional-averages-output",
        default="",
        help=(
            "Optional explicit path for the regional-averages netCDF file. "
            "Default: derived from the raw output path by appending "
            "`_regional_averages.nc`."
        ),
    )
    parser.add_argument(
        "--tropics-lat-deg",
        type=float,
        default=23.5,
        help=(
            "Latitude cutoff (degrees) for the tropics/extratropics split used "
            "by the regional-averages netCDF. Default 23.5 (approximately the "
            "tropical latitudes defined by Earth's axial tilt)."
        ),
    )
    parser.add_argument(
        "output",
        nargs="?",
        default="",
        help="Optional output path. Defaults to a parameterized `.pt` filename.",
    )
    args = parser.parse_args(argv)
    if args.regional_averages is None:
        args.regional_averages = args.region is None and args.region_latlon is None
    return args


def get_checkpoint_path(run: str) -> str:
    return "/path/to" f"/project-data/screamcast/{run}/output/best.pth"


def resolve_checkpoint_path(args: argparse.Namespace) -> str:
    if args.checkpoint:
        return args.checkpoint
    return get_checkpoint_path(args.run_name)


def default_output_dir(args: argparse.Namespace) -> str:
    return (
        "screamcast_comparison_"
        f"{args.forecast_model}_nudged{args.nudge_only_wind}_steps{args.n_steps}_"
        f"outer{args.n_steps_outer}_coarsen{args.coarsen}_"
        f"omegafilter{args.omega_filter_strength}"
    )


def load_forecast_model(
    checkpoint_path: str,
    domain_size: int,
    device: torch.device,
    forecast_model_name: str,
    inference_batch_size: int | None = None,
) -> ScreamcastModel:
    t_load0 = time.perf_counter()
    logger.info(
        "loading %s forecast metadata from checkpoint %s",
        forecast_model_name,
        checkpoint_path,
    )
    forecast_model = ScreamcastModel.from_checkpoint(checkpoint_path, bf16=True)
    forecast_model.inference_batch_size = inference_batch_size
    forecast_model = forecast_model.to(device)
    forecast_model.eval()
    forecast_model.set_tile_size(domain_size)
    if forecast_model_name != "screamcast":
        forecast_model.to_persistence()
    logger.info(
        "%s forecast model ready in %.1fs",
        forecast_model_name,
        time.perf_counter() - t_load0,
    )
    return forecast_model


def _get_omega_channel_indices(variable_names) -> list[int]:
    return [
        i
        for i, name in enumerate(variable_names)
        if name == "omega" or name.startswith("omega_")
    ]


def run_scream(
    x_tensor,
    coords,
    n_steps,
    model,
    pad,
    unpad,
    topology,
    omega_filter_strength=0.0,
    omega_filter_coarsen=None,
    F=None,
    write_step=None,
    sht_highpass=None,
    before_step=None,
    start_step=0,
    return_coords=False,
    tag="",
):
    """Run ``n_steps`` steps numbered ``start_step, start_step + 1, ...`` (the numbers
    ``before_step`` and ``write_step`` see). ``F`` is added after every step. Returns the
    final state, and its coords with ``return_coords``."""
    t_run0 = time.perf_counter()
    logger.info(
        "run_scream start n_steps=%d F=%s", n_steps, "yes" if F is not None else "no"
    )
    in_variable = coords["variable"]
    in_vars = list(in_variable)
    omega_channel_indices = _get_omega_channel_indices(in_vars)
    model_coords = topology.pad_coords(coords)
    model_coords["batch"] = np.empty(1)
    out_vars_template = list(model.output_coords(model_coords)["variable"])
    out_var_idx = {v: i for i, v in enumerate(out_vars_template)}
    from_out_in = [i for i, v in enumerate(in_vars) if v in out_var_idx]
    from_out_out = [out_var_idx[in_vars[i]] for i in from_out_in]

    with torch.inference_mode():
        for step_index in range(start_step, start_step + n_steps):
            if before_step is not None:
                before_step(step_index)
            logger.debug("step %d/%d starting pad", step_index + 1, n_steps)
            x_with_halo = pad(x_tensor)
            logger.debug("step %d/%d pad done", step_index + 1, n_steps)
            if omega_filter_strength > 0.0:
                if omega_filter_coarsen is None:
                    raise ValueError(
                        "omega_filter_coarsen is required when omega_filter_strength > 0."
                    )
                if not omega_channel_indices:
                    raise ValueError(
                        "No omega channels found in the model input state."
                    )
                omega_low_pass = low_pass(
                    x_with_halo[:, omega_channel_indices], omega_filter_coarsen
                )
                x_with_halo[:, omega_channel_indices] -= (
                    omega_low_pass * omega_filter_strength
                )
            coords_with_halo = topology.pad_coords(coords)
            coords_with_halo["batch"] = np.empty(1)
            logger.debug("step %d/%d starting model forward", step_index + 1, n_steps)
            out_with_halo, out_coords_with_halo = model(x_with_halo, coords_with_halo)
            logger.debug("step %d/%d model forward done", step_index + 1, n_steps)
            out_tensor = unpad(out_with_halo)
            out_coords = topology.crop_coords(out_coords_with_halo)
            if sht_highpass is not None:
                out_tensor = sht_highpass(out_tensor)
            if write_step is not None:
                logger.debug("step %d/%d starting write", step_index + 1, n_steps)
                write_step(step_index, out_tensor)
                logger.debug("step %d/%d write done", step_index + 1, n_steps)
            x_next = x_tensor.clone()
            x_next[:, from_out_in] = out_tensor[:, from_out_out]
            if F is not None:
                x_next += F
            x_tensor = x_next
            coords = dict(out_coords)
            coords["batch"] = np.empty(1)
            logger.info(
                "run_scream step %d/%d%s", step_index + 1, start_step + n_steps, tag
            )
    logger.info("run_scream done in %.1fs", time.perf_counter() - t_run0)
    if return_coords:
        # the loop carries the model's output coords; the state holds the inputs
        coords = dict(coords)
        coords["variable"] = in_variable
        return x_tensor, coords
    return x_tensor


def get_truth(n_steps, t0, model, fetch_tiles):
    t_fetch0 = time.perf_counter()
    logger.info("get_truth fetching local truth for n_steps=%d", n_steps)
    x_tensor, _ = fetch_tiles(t0 + n_steps * model.dt)
    logger.info("get_truth done in %.1fs", time.perf_counter() - t_fetch0)
    return x_tensor


def get_local_ace_correction(
    x_scream_global, n_steps, model, t0, scream2ace, ace_model, ace2local, tile_size
):
    t0_corr = time.perf_counter()
    logger.info(
        "get_local_ace_correction start n_steps=%d x_scream_global.shape=%s",
        n_steps,
        tuple(x_scream_global.shape),
    )
    x_unstructured = faces_to_unstructured(
        x_scream_global, ne=x_scream_global.shape[-1] // 2, npg=2
    )
    x_ace = scream2ace(x_unstructured)
    coords_ace = {
        "time": np.array([t0 + n_steps * model.dt]),
        "lead_time": np.array([np.timedelta64(0, "ns")]),
        "variable": ace_model.input_coords()["variable"],
        "lat": ace_model.ace_lat,
        "lon": ace_model.ace_lon,
    }
    correction_ace, _ = ace_model(x_ace, coords_ace)
    correction_local = ace2local(correction_ace).reshape(
        correction_ace.shape[0], correction_ace.shape[1], 1, tile_size, tile_size
    )
    logger.info("get_local_ace_correction done in %.1fs", time.perf_counter() - t0_corr)
    return correction_local


def subset_channels(x_tensor, source_vars, target_vars):
    source_index = {name: i for i, name in enumerate(source_vars)}
    missing = [name for name in target_vars if name not in source_index]
    if missing:
        raise ValueError(f"Missing required channels when subsetting: {missing}")
    keep = [source_index[name] for name in target_vars]
    return x_tensor[:, keep]


def replay_stepper(
    model, steps_per_hour: int, pad, unpad, topology, write_step, **run_kwargs
):
    """``advance_hour`` of ``run_with_replay``: one hour of ``run_scream``, its steps
    numbered from ``(hour - 1) * steps_per_hour``. A tile-jittered model is set to that
    step first, so a window re-run after a rewind sees the jitter offsets of its first
    run, and a replay with a zero increment is the run without replay."""

    def advance_hour(state, hour, forcing, write):
        x, c = state
        start = (hour - 1) * steps_per_hour
        if isinstance(model, JitteredModel):
            model.step = start
        return run_scream(
            x,
            c,
            steps_per_hour,
            model,
            pad,
            unpad,
            topology,
            F=forcing,
            write_step=write_step if write else None,
            start_step=start,
            return_coords=True,
            tag="" if write else " (replay predictor)",
            **run_kwargs,
        )

    return advance_hour


def load_scrip_lonlat(path: str) -> dict:
    """Load grid_center_lon/lat from a SCRIP .nc file into a float32 dict."""
    with xr.open_dataset(path) as ds:
        return {
            "lon": np.asarray(ds["grid_center_lon"].values, dtype=np.float32),
            "lat": np.asarray(ds["grid_center_lat"].values, dtype=np.float32),
        }


Driver = tuple[float, object, np.datetime64, str]  # hours, data source, time, label


def resolve_drivers(
    *,
    args: argparse.Namespace,
    t0: np.datetime64,
    ds,
    n_steps: int,
    dt_hours: float,
) -> list[Driver]:
    """Driver states, sorted by lead: ``--driver-stores`` or, without them,
    SCREAM_MAIN_ZARR_PATH every ``--driver-every-hours`` over the run."""
    if args.driver_stores:
        profile = os.getenv("SCREAM_ZARR_PROFILE", "")
        drivers = []
        for path in args.driver_stores:
            entry = data_catalog.ic_store(path, profile)
            t = np.datetime64(entry.reference_time)
            hours = float((t - t0) / np.timedelta64(1, "h"))
            drivers.append((hours, entry.to_data_source(), t, path))
    else:
        every = args.driver_every_hours
        if every <= 0:
            raise ValueError("--driver-every-hours must be positive")
        last = int(np.ceil(n_steps * dt_hours / every)) * every
        drivers = [
            (float(h), ds, t0 + np.timedelta64(h, "h"), "SCREAM_MAIN_ZARR_PATH")
            for h in range(0, last + 1, every)
        ]
    drivers.sort(key=lambda d: d[0])
    hours = [d[0] for d in drivers]
    if len(set(hours)) != len(hours):
        raise ValueError(f"two drivers share a valid time: {hours} h")
    if hours[0] < 0:
        raise ValueError(f"driver at {hours[0]} h is before the initial time")
    return drivers


def load_driver(
    args: argparse.Namespace, topology: TileTopology, driver: Driver, in_vars, device
):
    """A driver state as ``(1, C, 6, n, n)`` full faces on every rank."""
    _, source, t, _ = driver
    return load_full_face_scream_state(
        t,
        in_vars,
        topology.face_size,
        source,
        device,
        args.ace_fetch_channel_chunk,
        topology.world_size,
        topology.rank,
    )


def build_anchored_boundary(
    *,
    args: argparse.Namespace,
    topology: TileTopology,
    ic_boxes: list[torch.Tensor],
    drivers: list[Driver],
    in_vars: list[str],
    dt_hours: float,
    n_steps: int,
    device: torch.device,
    on_driver=None,
) -> AnchoredBoundary:
    """The anchored ring of a regional run. ``on_driver(hours, full_faces)`` sees every
    driver state as it is loaded (the replay takes its targets from them)."""
    hours = [d[0] for d in drivers]
    needed = (n_steps - 1) * dt_hours  # the last step pads the state at this time
    if hours[-1] < needed:
        raise ValueError(
            f"boundary drivers end at +{hours[-1]} h; the run needs +{needed:.2f} h"
        )

    def box_of(driver: Driver) -> list[torch.Tensor]:
        full = load_driver(args, topology, driver, in_vars, device)
        if on_driver is not None:
            on_driver(driver[0], full)
        return faces_to_pieces(full[0], topology)

    drivers = list(drivers)
    if drivers[0][0] == 0.0:
        driver = drivers.pop(0)
        driver0 = box_of(driver)
        logger.info("boundary driver +0 h: %s", driver[3])
    else:
        driver0 = ic_boxes
        logger.info("boundary driver +0 h: the initial condition (no 0 h store)")
    boundary = AnchoredBoundary(
        topology=topology,
        ic_boxes=ic_boxes,
        driver0=driver0,
        block=args.boundary_block,
        device=device,
    )
    del driver0
    for driver in drivers:
        boundary.add(driver[0], box_of(driver))
        logger.info("boundary driver +%g h: %s", driver[0], driver[3])
    return boundary


def onthefly_ic_start(args: argparse.Namespace) -> tuple[dict, np.datetime64]:
    """``--ic-onthefly``'s spec and initial time (``--initial-time``, by default the
    init of a forecast source); raises before start-up on what the run cannot do."""
    if args.ic_store is not None:
        raise ValueError("give --ic-store or --ic-onthefly, not both")
    if args.correction != "none":
        raise ValueError("--ic-onthefly needs --correction none")
    if args.driver_target_dir is None:
        raise ValueError("--ic-onthefly needs --driver-target-dir")
    spec = load_spec(args.ic_onthefly, ic=True)
    if args.initial_time is None and spec["source"] == "era5":
        raise ValueError("--ic-onthefly from ERA5 needs --initial-time")
    t0 = np.datetime64(args.initial_time or spec["init"], "s")
    if not make_source(spec).has(t0):
        raise ValueError(f"--ic-onthefly: {spec['source']} has no state at {t0}")
    return spec, t0


def onthefly_ic(
    args: argparse.Namespace,
    spec: dict,
    t0: np.datetime64,
    run_dir: str,
    rank: int,
    device: torch.device,
) -> str:
    """The path of the initial condition ``--ic-onthefly`` builds: rank 0 builds it (or
    keeps a matching one) in ``--driver-target-dir`` and writes ``IC_READY`` (its path)
    or ``FAILED`` (the traceback) in ``run_dir``; the other ranks wait for either. A
    file, not a collective, so the build may outlast the NCCL timeout."""
    if rank != 0:
        return wait_for_ic(run_dir, args.driver_target_timeout)
    os.makedirs(run_dir, exist_ok=True)
    try:
        path = build_ic_store(
            spec,
            t0,
            args.driver_target_dir,
            device=device,
            history=history_entry([sys.executable, *sys.argv]),
        )
    except BaseException:
        with open(os.path.join(run_dir, FAILED), "w") as f:
            f.write(traceback.format_exc())
        raise
    tmp = os.path.join(run_dir, f"{IC_READY}.partial")
    with open(tmp, "w") as f:
        f.write(path)
    os.rename(tmp, os.path.join(run_dir, IC_READY))
    if device.type == "cuda":
        torch.cuda.empty_cache()  # before the model loads
    return path


def main() -> None:
    args = parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    if not 0.0 <= args.omega_filter_strength <= 1.0:
        raise ValueError("--omega-filter-strength must be between 0 and 1 inclusive.")
    if args.sht_omega_lmax < 0:
        raise ValueError("--sht-omega-lmax must be non-negative.")
    if args.output_chunk_size <= 0:
        raise ValueError("--output-chunk-size must be positive.")
    if args.halo_channel_chunk < 0:
        raise ValueError("--halo-channel-chunk must be non-negative.")
    if not 0.0 < args.tropics_lat_deg < 90.0:
        raise ValueError("--tropics-lat-deg must be strictly between 0 and 90.")

    checkpoint_path = resolve_checkpoint_path(args)
    tile_size = args.tile_size
    n_steps = args.n_steps
    if args.output_steps < 1 or n_steps % args.output_steps != 0:
        raise ValueError(
            "--output-steps must be >= 1 and divide --n-steps; got "
            f"--output-steps {args.output_steps}, --n-steps {n_steps}"
        )
    n_steps_outer = args.n_steps_outer
    coarsen = args.coarsen
    correction = args.correction
    ic_spec = None
    if args.ic_onthefly is not None:
        ic_spec, t0 = onthefly_ic_start(args)
        scream_data = None  # opened once rank 0 has built it
    elif args.ic_store is None:
        scream_data = data_catalog.scream_sdecadal
        t0 = np.datetime64(args.initial_time or "2020-10-13T00:00:00")
    else:
        if correction != "none":
            raise ValueError(
                "--ic-store holds one frame; nudging and ACE correction need "
                "SCREAM_MAIN_ZARR_PATH. Use --correction none."
            )
        scream_data = data_catalog.ic_store(
            args.ic_store, os.getenv("SCREAM_ZARR_PROFILE", "")
        )
        t0 = scream_data.reference_time
        if args.initial_time is not None and np.datetime64(args.initial_time) != t0:
            raise ValueError(
                f"--initial-time {args.initial_time} is not the IC store's "
                f"initial_time {t0}."
            )

    one_frame_ic = args.ic_store is not None or args.ic_onthefly is not None
    # --region-latlon becomes a --region once the world size is known
    regional = args.region is not None or args.region_latlon is not None
    if regional:
        if correction != "none":
            raise ValueError("--region needs --correction none")
        if args.pressure_levels is not None:
            raise ValueError("--pressure-levels does not support --region yet")
        if args.boundary_block < 1:
            raise ValueError("--boundary-block must be >= 1")
        if (
            args.region_boundary == "replicate"
            and args.driver_stores
            and not args.replay
        ):
            raise ValueError(
                "--driver-stores needs --region-boundary anchored or --replay"
            )
        if (
            args.region_boundary == "anchored"
            and not args.driver_stores
            and args.driver_onthefly is None
            and one_frame_ic
        ):
            raise ValueError(
                "--region-boundary anchored with --ic-store needs --driver-stores or "
                "--driver-onthefly (the IC store holds one frame)"
            )
        if (
            args.region_boundary == "anchored"
            and args.driver_onthefly is not None
            and args.boundary_block != REPLAY_BLOCK
        ):
            raise ValueError(
                f"an anchored ring built --driver-onthefly needs --boundary-block "
                f"{REPLAY_BLOCK} (the targets are {REPLAY_BLOCK} x {REPLAY_BLOCK} means)"
            )
        if args.halo_channel_chunk != 0:
            logger.info("--region: halo exchange with every channel at once")
            args.halo_channel_chunk = 0
        if args.sht_omega_lmax > 0:
            # The l < lmax omega modes are global; a window cannot know them.
            logger.warning("--region: omega SHT high-pass OFF (a global operator)")
            args.sht_omega_lmax = 0
    elif args.driver_stores and not args.replay:
        raise ValueError("--driver-stores needs --region or --replay")
    level_ramp = None
    if args.replay_level_ramp is not None:
        if not args.replay:
            raise ValueError("--replay-level-ramp needs --replay")
        level_ramp = tuple(float(v) for v in args.replay_level_ramp.split(","))
        if len(level_ramp) != 2 or not level_ramp[1] > level_ramp[0]:
            raise ValueError(
                f"--replay-level-ramp PFULL,PZERO needs PZERO > PFULL (hPa), got "
                f"{args.replay_level_ramp!r}"
            )
    if args.driver_onthefly is not None:
        anchored_ring = regional and args.region_boundary == "anchored"
        if not (args.replay or anchored_ring):
            raise ValueError(
                "--driver-onthefly drives --replay or an anchored --region ring"
            )
        if args.driver_stores:
            raise ValueError("give --driver-stores or --driver-onthefly, not both")
        if args.driver_target_dir is None:
            raise ValueError("--driver-onthefly needs --driver-target-dir")
        load_spec(args.driver_onthefly)  # fail before start-up on a bad spec
    if args.replay:
        if correction != "none":
            raise ValueError("--replay needs --correction none")
        if not args.driver_stores and args.driver_onthefly is None and one_frame_ic:
            raise ValueError(
                "--replay with --ic-store needs --driver-stores or --driver-onthefly "
                "(the IC store holds one frame)"
            )

    DistributedManager.initialize()
    dist_mgr = DistributedManager()

    rank = dist_mgr.rank
    world_size = dist_mgr.world_size
    device = dist_mgr.device

    if rank != 0:
        logger.setLevel(logging.CRITICAL + 1)

    run_dir = None
    if args.driver_onthefly is not None or ic_spec is not None:
        # this run's FAILED, DONE, markers and log; targets and ICs are shared
        name = [f"{args.run_name}-{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}"]
        torch.distributed.broadcast_object_list(name, src=0, device=device)
        run_dir = os.path.join(args.driver_target_dir, "runs", name[0])
    if ic_spec is not None:
        ic_path = onthefly_ic(args, ic_spec, t0, run_dir, rank, device)
        scream_data = data_catalog.ic_store(
            ic_path, os.getenv("SCREAM_ZARR_PROFILE", "")
        )
        logger.info("initial condition on the fly: %s", ic_path)

    npg = 2
    grid_tgt = load_scrip_lonlat(args.grid_tgt)
    ne_padded = 1536
    pad_width_data = 512
    lon = unstructured_to_6faces(torch.from_numpy(grid_tgt["lon"]), ne_padded, npg)
    lat = unstructured_to_6faces(torch.from_numpy(grid_tgt["lat"]), ne_padded, npg)
    if args.region_latlon is not None:
        interior = slice(pad_width_data, pad_width_data + 1024 * npg)
        args.region, margin = find_box_region(
            lat[:, interior, interior].numpy(),
            lon[:, interior, interior].numpy(),
            parse_latlon_box(args.region_latlon),
            tile_size,
            world_size,
            args.region_margin_tiles,
        )
        logger.info(
            "--region-latlon %s: --region %s (margin %d tiles, %d ranks)",
            args.region_latlon,
            args.region,
            margin,
            world_size,
        )

    model = load_forecast_model(
        checkpoint_path,
        tile_size,
        device,
        args.forecast_model,
        inference_batch_size=args.inference_batch_size,
    )

    logger.info("building halo exchange plan")

    topology = TileTopology(
        world_size=world_size,
        rank=rank,
        face_size=1024 * npg,
        tile_size=tile_size,
        halo_width=args.halo_width,
        active_tiles=(
            parse_region(args.region, 1024 * npg // tile_size)
            if args.region is not None
            else None
        ),
    )
    if topology.is_regional:
        logger.info(
            "REGIONAL window %s: %d active tiles, %d ring tiles, boundary=%s",
            args.region,
            topology.total_tiles,
            len(topology.ring_tiles),
            args.region_boundary,
        )
    my_tiles = topology.local_tiles
    tile_size = topology.tile_size

    halo_exchange = DistributedTileKNNHaloPadding_AllGather.from_padded_face_grid(
        topology=topology,
        lon=lon,
        lat=lat,
        pad_width_data=pad_width_data,
        device=device,
    )
    logger.info("local lat/lon shape %s", halo_exchange.lon_deg.shape)
    halo_exchange.to(device)
    pad = channel_chunked(args.halo_channel_chunk)(halo_exchange)

    if args.halo_adjoint:
        logger.info(
            "building NormalizedHaloAdjoint (window=%s, cross_face=%s)",
            args.halo_window,
            args.halo_adjoint_cross_face,
        )
        if args.halo_window:
            padded_size = topology.padded_tile_size
            w1d = torch.tensor(
                kaiser_bessel_derived(padded_size, 10 * np.pi), dtype=torch.float32
            )
            window = w1d.unsqueeze(0) * w1d.unsqueeze(1)
        else:
            window = None
        unpad = channel_chunked(args.halo_channel_chunk)(
            NormalizedHaloAdjoint(
                halo_exchange,
                window=window,
                own_face_only=not args.halo_adjoint_cross_face,
            ).to(device)
        )
    else:
        unpad = channel_chunked(args.halo_channel_chunk)(halo_exchange.topology.crop)

    model.set_latlon(lat_deg=halo_exchange.lat_deg, lon_deg=halo_exchange.lon_deg)
    if args.forecast_model == "screamcast":
        model.compile()

    if args.tile_jitter > 0:
        model = JitteredModel(
            model,
            jitter=args.tile_jitter,
            mode=args.tile_jitter_mode,
            seed=args.tile_jitter_seed,
            fill=args.tile_jitter_fill,
        )
        logger.info(
            "sub-patch jitter ON: %dx%d window in the %d padded tile, mode=%s seed=%d fill=%s",
            topology.padded_tile_size - args.tile_jitter,
            topology.padded_tile_size - args.tile_jitter,
            topology.padded_tile_size,
            args.tile_jitter_mode,
            args.tile_jitter_seed,
            args.tile_jitter_fill,
        )

    in_coords = dict(model.input_coords())
    logger.info("Loaded model: %s (%s)", args.run_name, args.forecast_model)
    logger.info(
        "tile_size=%d in_channels=%d local_tiles=%d",
        tile_size,
        len(in_coords["variable"]),
        topology.tiles_per_rank,
    )

    ds = scream_data.to_data_source()

    ace_model = None
    scream2ace = None
    ace2local = None
    f_nudge = None
    if correction == "ace":
        from screamcast.ace._earth2studio import ACE2ForecastResidualModel

        logger.info("loading ACE residual checkpoint %s", args.ace_checkpoint)
        ace_model = ACE2ForecastResidualModel.from_checkpoint(
            args.ace_checkpoint, device=device
        )
        ace_input_vars = list(ace_model.input_coords()["variable"])

        x_scream_global = load_full_face_scream_state(
            t0,
            ace_input_vars,
            topology.face_size,
            ds,
            device,
            args.ace_fetch_channel_chunk,
            topology.world_size,
            topology.rank,
        )
        with xr.open_dataset(GRID_PATH) as grid:
            src_lon = np.asarray(grid["grid_center_lon"].values, dtype=np.float32)
            src_lat = np.asarray(grid["grid_center_lat"].values, dtype=np.float32)
        ace_lat = np.asarray(ace_model.ace_lat, dtype=np.float32)
        ace_lon = np.asarray(ace_model.ace_lon, dtype=np.float32)
        if ace_lat.ndim == 1 and ace_lon.ndim == 1:
            ace_lat_1d = ace_lat
            ace_lon_1d = ace_lon
        else:
            if ace_lat.shape != ace_lon.shape:
                raise ValueError(
                    f"Expected ACE lat/lon with matching shapes, got {ace_lat.shape} vs {ace_lon.shape}"
                )
            ace_lat_1d = ace_lat[:, 0]
            ace_lon_1d = ace_lon[0, :]
            check_lon, check_lat = np.meshgrid(ace_lon_1d, ace_lat_1d, indexing="xy")
            if not (
                np.allclose(ace_lat, check_lat) and np.allclose(ace_lon, check_lon)
            ):
                raise ValueError("Expected a regular tensor-product lat/lon grid")
        scream2ace = UnstructuredToLatLonRegridder(
            src_lon,
            src_lat,
            ace_lat,
            ace_lon,
        ).to(device)
        if topology.tiles_per_rank != 1:
            raise ValueError(
                "ACE correction currently requires tiles_per_rank == 1 for local regridding."
            )
        local_latlon = halo_exchange.topology.crop(model.latlon)[0]
        local_lat = local_latlon[..., 0].cpu().numpy().reshape(-1)
        local_lon = local_latlon[..., 1].cpu().numpy().reshape(-1)
        ace2local = LatLonToPointGridRegridder(
            local_lon, local_lat, ace_lat_1d, ace_lon_1d
        ).to(device)
        local_correction_full = get_local_ace_correction(
            x_scream_global, 0, model, t0, scream2ace, ace_model, ace2local, tile_size
        )
        local_correction = subset_channels(
            local_correction_full,
            ace_input_vars,
            list(in_coords["variable"]),
        )
        f_nudge = local_correction / n_steps

    dt_minutes = model.dt / np.timedelta64(1, "s") / 60

    output_path = args.output or f"{default_output_dir(args)}.zarr"
    params = {
        # one line per run: every run creates its output stores afresh (no resume)
        "history": history_entry([sys.executable, *sys.argv]),
        "run_name": args.run_name,
        "checkpoint": checkpoint_path,
        "forecast_model": args.forecast_model,
        "tile_size": tile_size,
        "halo_width": args.halo_width,
        "n_steps": n_steps,
        "n_steps_outer": n_steps_outer,
        "coarsen": coarsen,
        "omega_filter_strength": args.omega_filter_strength,
        "sht_omega_lmax": args.sht_omega_lmax,
        "output_chunk_size": args.output_chunk_size,
        "nudge_only_wind": args.nudge_only_wind,
        "correction": correction,
        "ace_checkpoint": args.ace_checkpoint,
        "initial_time": str(t0),
        "device": str(device),
        "halo_adjoint": args.halo_adjoint,
        "halo_adjoint_cross_face": (
            args.halo_adjoint_cross_face if args.halo_adjoint else None
        ),
        "tile_jitter": args.tile_jitter,
        "tile_jitter_mode": args.tile_jitter_mode if args.tile_jitter > 0 else None,
        "region": args.region,
        "region_latlon": args.region_latlon,
        "region_margin_tiles": (
            args.region_margin_tiles if args.region_latlon is not None else None
        ),
        "region_boundary": args.region_boundary if args.region else None,
        "driver_stores": list(args.driver_stores or []),
        "replay": args.replay,
        "replay_every_hours": args.replay_every_hours if args.replay else None,
        "replay_ell_half": args.replay_ell_half if args.replay else None,
        "replay_vars": args.replay_vars if args.replay else None,
        "replay_level_ramp": args.replay_level_ramp,
        "driver_onthefly": (
            load_spec(args.driver_onthefly) if args.driver_onthefly else None
        ),
        "driver_target_dir": args.driver_target_dir,
        "ic_onthefly": ic_spec,
        "driver_producer": args.driver_producer if args.driver_onthefly else None,
        "replay_region_edge": (
            args.replay_region_edge if args.replay and args.region else None
        ),
        "boundary_block": args.boundary_block if args.region else None,
    }
    # Build output variable grouping and output store before rollout so we can
    # write each step as it is produced.
    out_vars = list(model.output_coords(dict(in_coords))["variable"])
    grouped_variables = _split_variable_levels(out_vars)
    if args.output_variables is not None:
        unknown = set(args.output_variables) - grouped_variables.keys()
        if unknown:
            raise ValueError(
                f"Unknown output variables: {sorted(unknown)}. "
                f"Available: {sorted(grouped_variables.keys())}"
            )
        grouped_variables = OrderedDict(
            (k, v) for k, v in grouped_variables.items() if k in args.output_variables
        )
    if args.output_levels is not None:
        requested = set(args.output_levels)
        filtered = OrderedDict()
        for k, entries in grouped_variables.items():
            if entries[0][0] is None:
                filtered[k] = entries  # 2D variable — levels don't apply
            else:
                kept = [(level, ci) for level, ci in entries if level in requested]
                if kept:
                    filtered[k] = kept
        grouped_variables = filtered
    nside = 1024 * npg  # 2048
    # Extract interior lon/lat from the padded face grid [6, 3072, 3072]
    lat_face = lat[
        :,
        pad_width_data : pad_width_data + nside,
        pad_width_data : pad_width_data + nside,
    ].numpy()
    lon_face = lon[
        :,
        pad_width_data : pad_width_data + nside,
        pad_width_data : pad_width_data + nside,
    ].numpy()

    hyam, hybm, _ = scream_data.to_hybrid_vertical_coordinates()
    hyam = hyam.astype(np.float32)
    hybm = hybm.astype(np.float32)

    logger.info("output: %s", output_path)
    output_steps = args.output_steps
    n_output_steps = n_steps // output_steps
    if args.zarr_output and rank == 0:
        prepare_output_store(
            output_path=output_path,
            grouped_variables=grouped_variables,
            n_times=1,
            n_steps=n_output_steps,
            nside=nside,
            tile_size=args.output_chunk_size,
            time_values=np.array([t0]),
            lat=lat_face,
            lon=lon_face,
            dt=model.dt * output_steps,
            hyam=hyam,
            hybm=hybm,
            attrs={**params, "pixel_ordering": "cubesphere_faces_2d"},
        )
        zarr.consolidate_metadata(output_path)
    torch.distributed.barrier()

    store = zarr.open_group(output_path, mode="a") if args.zarr_output else None
    writer: ZarrWriter | FaceZarrWriteStep | None = None

    use_plev_output = args.pressure_levels is not None
    if use_plev_output and not args.zarr_output:
        raise ValueError(
            "--pressure-levels requires --zarr-output (pressure-level output "
            "is written to a zarr store)."
        )
    if args.write_mode == "face-zarr" and use_plev_output:
        raise ValueError(
            "--write-mode face-zarr does not support --pressure-levels yet."
        )
    plev_store = None
    plev_scream2ace = None
    plev_ps_channel = None
    plev_hyam_t = None
    plev_hybm_t = None
    plev_pa_t = None
    plev_grouped: OrderedDict | None = None
    plev_output_path = ""
    if use_plev_output:
        if any(p <= 0 for p in args.pressure_levels):
            raise ValueError("--pressure-levels must all be positive (hPa)")
        # Build grouping from the full un-filtered model output so that 3D
        # interpolation always sees every source sigma level. --output-variables
        # is re-applied (user's variable selection is meaningful for plev too);
        # --output-levels is NOT applied (it selects source sigma levels, which
        # is orthogonal to target pressure levels).
        all_grouped = _split_variable_levels(out_vars)
        if "ps" not in all_grouped or all_grouped["ps"][0][0] is not None:
            raise ValueError(
                "pressure-level output requires 'ps' in the model output; "
                "this checkpoint does not produce it."
            )
        plev_ps_channel = all_grouped["ps"][0][1]
        if args.output_variables is not None:
            plev_grouped = OrderedDict(
                (k, v) for k, v in all_grouped.items() if k in args.output_variables
            )
        else:
            plev_grouped = all_grouped

        # ACE grid: reuse from the correction="ace" path if already loaded to
        # avoid re-downloading the ACE2ERA5 package.
        if correction == "ace":
            ace_lat_plev = ace_lat_1d
            ace_lon_plev = ace_lon_1d
            src_lon_plev = src_lon
            src_lat_plev = src_lat
        else:
            # Pull the 1-D ACE2ERA5 grid constants directly. Calling
            # ``ACE2ERA5.load_model`` just to get the grid triggers fme's
            # ``Distributed.context()`` requirement and downloads model
            # weights — unneeded when we only want lat/lon.
            from earth2studio.models.px.ace2 import ACE_GRID_LAT, ACE_GRID_LON

            logger.info("using ACE2ERA5 grid constants for pressure-level output")
            ace_lat_plev = np.asarray(ACE_GRID_LAT, dtype=np.float32)
            ace_lon_plev = np.asarray(ACE_GRID_LON, dtype=np.float32)
            with xr.open_dataset(GRID_PATH) as _grid:
                src_lon_plev = np.asarray(
                    _grid["grid_center_lon"].values, dtype=np.float32
                )
                src_lat_plev = np.asarray(
                    _grid["grid_center_lat"].values, dtype=np.float32
                )
        plev_scream2ace = UnstructuredToLatLonRegridder(
            src_lon_plev,
            src_lat_plev,
            ace_lat_plev,
            ace_lon_plev,
        ).to(device)

        # Hybrid-sigma source coefficients (SCREAM convention:
        #   P_Pa = hyam * P0_SCREAM + hybm * PS_Pa).
        #
        # ``hyam`` / ``hybm`` here are already subsampled (shape 32) by
        # ``data_catalog.scream_sdecadal``. Channel-level suffixes in the
        # model output are indices into that subsampled array, so index by the
        # actual output levels rather
        # than using all 32 blindly — the checkpoint may emit only a subset.
        first_3d_entries = next(
            (entries for entries in plev_grouped.values() if entries[0][0] is not None),
            None,
        )
        if first_3d_entries is None:
            raise ValueError(
                "--pressure-levels requires at least one 3D output variable."
            )
        output_sigma_levels = sorted(level for level, _ in first_3d_entries)
        plev_hyam_t = torch.from_numpy(hyam[output_sigma_levels]).to(device)
        plev_hybm_t = torch.from_numpy(hybm[output_sigma_levels]).to(device)
        plev_pa_t = torch.tensor(
            [p * 100.0 for p in args.pressure_levels],
            device=device,
            dtype=torch.float32,
        )
        plev_output_path = args.ace_plev_output or f"{output_path}_ace_plev.zarr"
        logger.info("ace-plev output: %s", plev_output_path)

        if rank == 0:
            plev_attrs = {
                **params,
                "grid": "ace2_latlon",
                "vertical": "pressure_pa",
                "source_grid": "ne1024pg2",
                "source_vertical": "hybrid_sigma_subset",
                "pressure_levels_hpa": list(args.pressure_levels),
                "P0_Pa": float(P0_SCREAM),
            }
            prepare_latlon_plev_store(
                output_path=plev_output_path,
                grouped_variables=plev_grouped,
                n_times=1,
                n_steps=n_output_steps,
                time_values=np.array([t0]),
                lat=ace_lat_plev,
                lon=ace_lon_plev,
                plev=plev_pa_t.detach().cpu().numpy(),
                dt=model.dt * output_steps,
                attrs=plev_attrs,
            )
            zarr.consolidate_metadata(plev_output_path)
        torch.distributed.barrier()
        if rank == 0:
            plev_store = zarr.open_group(plev_output_path, mode="a")

    dt_hours = float(model.dt / np.timedelta64(1, "h"))
    onthefly = args.driver_onthefly is not None
    anchored = topology.is_regional and args.region_boundary == "anchored"
    needs_drivers = not onthefly and (args.replay or anchored)
    driver_spec = load_spec(args.driver_onthefly) if onthefly else None
    driver_source = make_source(driver_spec) if onthefly else None
    drivers = (
        resolve_drivers(args=args, t0=t0, ds=ds, n_steps=n_steps, dt_hours=dt_hours)
        if needs_drivers
        else []
    )
    replay = None
    replay_target = None
    target_reader = None
    ring_reader = None
    producer = None
    plan = []
    analyses: set[int] = set()
    if args.replay:
        steps_per_hour = round(1.0 / dt_hours)
        if abs(steps_per_hour * dt_hours - 1.0) > 1e-9 or n_steps % steps_per_hour:
            raise ValueError(
                f"--replay needs whole hours: {n_steps} steps of {dt_hours * 60:g} min"
            )
        if onthefly:
            driver_hours = source_hours(
                driver_source, t0, range(n_steps // steps_per_hour + 1)
            )
        else:
            driver_hours = {int(h) for h, *_ in drivers if float(h).is_integer()}
        plan = replay_plan(
            n_steps // steps_per_hour, args.replay_every_hours, driver_hours
        )
        analyses = set(analysis_hours(plan))
        if not analyses:
            raise ValueError(
                f"--replay: no analysis fits; drivers at {sorted(driver_hours)} h, "
                f"every {args.replay_every_hours} h, run of {n_steps // steps_per_hour} h "
                "(an analysis at t needs t + every/2 <= the run length)"
            )
        replay = Replay(
            in_vars=list(in_coords["variable"]),
            replay_vars=args.replay_vars.split(","),
            lat_face=lat_face,
            lon_face=lon_face,
            area_face=np.asarray(load_grid_area_face(GRID_PATH, ne=1024, npg=npg)),
            topology=topology,
            ell_half=args.replay_ell_half,
            window_steps=args.replay_every_hours * steps_per_hour,
            omega_lmax=args.sht_omega_lmax,
            normalize_edge=(
                topology.is_regional and args.replay_region_edge == "normalized"
            ),
            level_weights=(
                None
                if level_ramp is None
                else level_ramp_weights(hyam, hybm, *level_ramp)
            ),
        ).to(device)
        if level_ramp is not None:
            logger.info(
                "replay level ramp %s hPa: weight by level %s",
                args.replay_level_ramp,
                np.round(level_ramp_weights(hyam, hybm, *level_ramp), 3).tolist(),
            )
        logger.info(
            "REPLAY every %d h at +%s h, ell_half=%g (lmax %d), %d channels%s",
            args.replay_every_hours,
            sorted(analyses),
            args.replay_ell_half,
            replay.lmax,
            len(replay.channels),
            f", window edge {args.replay_region_edge}" if topology.is_regional else "",
        )

        def replay_target(hours: float, full: torch.Tensor) -> None:
            if hours in analyses and not onthefly:
                replay.add_target(int(hours), topology.faces_to_local_tiles(full))
                logger.info("replay target +%g h loaded", hours)

    if onthefly:
        ring = []
        if anchored:
            ring = ring_hours(
                driver_source, t0, args.driver_every_hours, (n_steps - 1) * dt_hours
            )
        n_hours = int(np.ceil(n_steps * dt_hours))
        steps = plan or [ReplayStep("window", h) for h in range(1, n_hours + 1)]
        schedule = target_schedule(steps, ring)
        stems = producer_stems(
            args.replay_vars.split(",") if args.replay else [],
            # the ring drives every input channel
            list(in_coords["variable"]) if anchored else [],
        )
        check = functools.partial(
            check_target,
            channels=target_channels(stems),
            tiles=producer_tiles(
                args.region, topology.tile_size, topology.face_size, anchored
            ),
            tile_size=topology.tile_size,
            origin=target_origin(driver_spec),
        )
        if rank == 0:
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            gpu = str(device.index)
            if visible:
                gpu = visible.split(",")[device.index]
            producer = ProducerProcess(
                spec=args.driver_onthefly,
                out_dir=args.driver_target_dir,
                run_dir=run_dir,
                run_init=t0,
                hours=[h for h, _ in schedule],
                gates=[g for _, g in schedule],
                mode=args.driver_producer,
                stems=stems,
                tile_size=topology.tile_size,
                face_size=topology.face_size,
                region=args.region,
                ring=anchored,
                gpu=gpu,
            )
            logger.info(
                "driver on the fly (%s, %s) on GPU %s -> %s: targets at +%s h",
                driver_source.describe(t0),
                args.driver_producer,
                gpu,
                args.driver_target_dir,
                [h for h, _ in schedule],
            )
        torch.distributed.barrier()  # the producer has made the run directory
        if replay is not None:
            target_reader = TargetReader(
                args.driver_target_dir,
                run_dir,
                t0,
                lambda path: read_blocks(
                    path, replay.names, topology.local_tiles, topology.tile_size
                ),
                check=check,
                mode=args.driver_producer,
                timeout=args.driver_target_timeout,
            )
            target_reader.prefetch(sorted(analyses))
        if anchored:
            ring_names = list(in_coords["variable"])
            ring_reader = TargetReader(
                args.driver_target_dir,
                run_dir,
                t0,
                lambda path: read_box_blocks(path, ring_names, topology),
                check=check,
                mode=args.driver_producer,
                timeout=args.driver_target_timeout,
            )
            ring_reader.prefetch(ring)

    logger.info("opening initial condition")
    before_step = None
    if topology.is_regional:
        full_faces = load_full_face_scream_state(
            t0,
            in_coords["variable"],
            topology.face_size,
            ds,
            device,
            args.ace_fetch_channel_chunk,
            topology.world_size,
            topology.rank,
        )
        x0 = topology.faces_to_local_tiles(full_faces)
        coords = topology.interior_coords(in_coords, time=np.array([t0]))
        if args.region_boundary == "anchored":
            ic_boxes = faces_to_pieces(full_faces[0], topology)
            del full_faces
            if ring_reader is None:
                boundary = build_anchored_boundary(
                    args=args,
                    topology=topology,
                    ic_boxes=ic_boxes,
                    drivers=drivers,
                    in_vars=list(in_coords["variable"]),
                    dt_hours=dt_hours,
                    n_steps=n_steps,
                    device=device,
                    on_driver=replay_target,
                )
                ring_pending: list[int] = []
            else:
                blocks0, _ = ring_reader.get(0)
                boundary = AnchoredBoundary(
                    topology=topology,
                    ic_boxes=ic_boxes,
                    driver0_blocks=[torch.from_numpy(b).to(device) for b in blocks0],
                    block=args.boundary_block,
                    device=device,
                )
                del blocks0
                logger.info("boundary driver +0 h: on the fly")
                ring_pending = ring[1:]
            del ic_boxes
            halo_exchange.ring_source = boundary

            def before_step(step_index: int) -> None:
                t = step_index * dt_hours
                # on the fly: add each driver state when the steps first reach it
                while ring_pending and boundary.hours[-1] < t:
                    h = ring_pending.pop(0)
                    blocks, info = ring_reader.get(h)
                    boundary.add_blocks(
                        h, [torch.from_numpy(b).to(device) for b in blocks]
                    )
                    logger.info(
                        "boundary driver +%d h: on the fly (waited %.1f s here)",
                        h,
                        info["blocked_s"],
                    )
                boundary.time_h = t

        else:
            del full_faces
            halo_exchange.ring_source = ReplicateRing(topology)
    else:
        x0, coords = fetch_local_tiles(
            ds,
            in_coords,
            topology,
            device,
            t0,
            args.ace_fetch_channel_chunk,
        )
    logger.info("initial condition loaded")
    in_vars = list(in_coords["variable"])
    if replay is not None and not onthefly:
        for driver in drivers:
            if driver[0] in analyses and int(driver[0]) not in replay.targets:
                replay_target(
                    driver[0], load_driver(args, topology, driver, in_vars, device)
                )

    wind_mask = torch.zeros(1, len(in_vars), 1, 1, 1, device=device)
    for i, var_name in enumerate(in_vars):
        if var_name.startswith("U") or var_name.startswith("V"):
            wind_mask[0, i] = 1.0

    logger.info("running nudged")
    if correction == "data":
        for outer_step in range(n_steps_outer):
            logger.info("outer_step=%d", outer_step)
            y1 = run_scream(x0, coords, n_steps, model, pad, unpad, topology)
            x1 = get_truth(
                n_steps * (outer_step + 1),
                t0,
                model,
                lambda truth_time: fetch_local_tiles(
                    ds,
                    in_coords,
                    topology,
                    device,
                    truth_time,
                    args.ace_fetch_channel_chunk,
                ),
            )
            error = x1 - y1
            f_nudge = low_pass(error, coarsen) / n_steps
    elif correction == "none":
        f_nudge = None

    if args.nudge_only_wind and f_nudge is not None:
        f_nudge = f_nudge * wind_mask

    write_step: "callable | None" = None
    if not args.zarr_output:
        writer = None
    elif args.write_mode == "legacy":
        writer = ZarrWriter(store, grouped_variables, tile_size)

        def write_step(step_index: int, out_tensor: torch.Tensor) -> None:
            out_index = output_frame_index(step_index, output_steps)
            if out_index is None:
                return
            for local_idx, (face, ti, tj) in enumerate(my_tiles):
                tile = out_tensor[0, :, local_idx]
                writer.write(0, out_index, face, ti * tile_size, tj * tile_size, tile)
            if not use_plev_output:
                return
            n_plev = plev_pa_t.shape[0]
            ps_local = out_tensor[:, plev_ps_channel]
            src_pressure = plev_hyam_t.view(
                -1, 1, 1, 1, 1
            ) * P0_SCREAM + plev_hybm_t.view(-1, 1, 1, 1, 1) * ps_local.unsqueeze(0)
            target_pressure = plev_pa_t.view(-1, 1, 1, 1, 1).expand(-1, *ps_local.shape)
            plev_channels = []
            for entries in plev_grouped.values():
                if entries[0][0] is None:
                    _, ch_idx = entries[0]
                    plev_channels.append(out_tensor[:, ch_idx : ch_idx + 1])
                else:
                    channel_indices = [ci for _, ci in entries]
                    src_vals = out_tensor[0, channel_indices].unsqueeze(1)
                    remapped = log_pressure_interpolate(
                        src_values=src_vals,
                        src_pressure=src_pressure,
                        target_pressure=target_pressure,
                        axis=0,
                    )
                    plev_channels.append(remapped.permute(1, 0, 2, 3, 4))
            plev_local = torch.cat(plev_channels, dim=1)

            global_faces = topology.gather_tiles_to_faces(plev_local)
            if rank != 0:
                return
            x_unstr = faces_to_unstructured(global_faces, ne=nside // 2, npg=2)
            x_latlon = plev_scream2ace(x_unstr)

            cursor = 0
            for base_name, entries in plev_grouped.items():
                if entries[0][0] is None:
                    plev_store[base_name][0, out_index] = (
                        x_latlon[0, cursor]
                        .detach()
                        .to("cpu", dtype=torch.float32)
                        .numpy()
                    )
                    cursor += 1
                else:
                    plev_store[base_name][0, out_index] = (
                        x_latlon[0, cursor : cursor + n_plev]
                        .detach()
                        .to("cpu", dtype=torch.float32)
                        .numpy()
                    )
                    cursor += n_plev

    else:
        writer = FaceZarrWriteStep(
            output_steps=output_steps,
            topology=topology,
            grouped_variables=grouped_variables,
            store=store,
        )
        write_step = writer

    sht_highpass = None
    if args.sht_omega_lmax > 0:
        # SHT filter is applied post-model on out_tensor, so the omega
        # indices must be looked up in the OUTPUT variable ordering (out_vars)
        # — input and output orderings differ in general.
        omega_ch = _get_omega_channel_indices(out_vars)
        if not omega_ch:
            raise ValueError(
                "--sht-omega-lmax > 0 but no omega channels found in model output."
            )
        with xr.open_dataset(GRID_PATH) as _grid_area_ds:
            if "grid_area" not in _grid_area_ds:
                raise ValueError(
                    f"{GRID_PATH} must include grid_area for --sht-omega-lmax."
                )
            area_unstr = np.asarray(_grid_area_ds["grid_area"].values, dtype=np.float32)
        area_face = unstructured_to_6faces(
            torch.from_numpy(area_unstr), 1024, npg
        ).numpy()
        lat_local = topology.faces_to_local_tiles(torch.from_numpy(lat_face)).numpy()
        lon_local = topology.faces_to_local_tiles(torch.from_numpy(lon_face)).numpy()
        area_local = topology.faces_to_local_tiles(torch.from_numpy(area_face)).numpy()
        logger.info(
            "building SHT high-pass (lmax=%d) on %d local points",
            args.sht_omega_lmax,
            lat_local.size,
        )
        sht_op = DistributedSHTHighpass(
            lat_local, lon_local, area_local, args.sht_omega_lmax
        ).to(device)

        def sht_highpass(out_tensor: torch.Tensor) -> torch.Tensor:
            out_tensor[:, omega_ch] = sht_op(out_tensor[:, omega_ch])
            return out_tensor

    regional_averager: RegionalAverager | None = None
    regional_output_path = ""
    if args.region is not None and not args.regional_averages:
        logger.info(
            "regional averages off for the --region window "
            "(pass --regional-averages to compute them)"
        )
    if args.regional_averages:
        regional_output_path = (
            args.regional_averages_output or f"{output_path}_regional_averages.nc"
        )
        logger.info("regional-averages output: %s", regional_output_path)
        area_face = load_grid_area_face(GRID_PATH, ne=1024, npg=npg)
        landfrac_face = load_landfrac_face(scream_data, ne=1024, npg=npg)
        lat_face_t = torch.from_numpy(lat_face).to(torch.float32)
        # Use un-filtered grouping so every output channel is averaged,
        # independent of --output-variables / --output-levels zarr filters.
        all_grouped_for_avg = _split_variable_levels(out_vars)
        regional_averager = RegionalAverager.from_topology(
            topology=topology,
            lat_face_deg=lat_face_t,
            landfrac_face=landfrac_face,
            area_face=area_face,
            variable_names=out_vars,
            grouped_variables=all_grouped_for_avg,
            output_path=regional_output_path,
            t0=t0,
            dt=model.dt,
            n_steps=n_steps,
            device=device,
            tropics_lat_cutoff_deg=args.tropics_lat_deg,
            hyam=hyam,
            hybm=hybm,
            attrs={**params, "pixel_ordering": "cubesphere_faces_2d"},
        )

        _inner_write_step = write_step
        # SCREAM precip fluxes are water-depth rates in m/s.
        _m_s_to_mm_day = 1000.0 * 86400.0

        def write_step(step_index: int, out_tensor: torch.Tensor) -> None:
            if _inner_write_step is not None:
                _inner_write_step(step_index, out_tensor)
            means = regional_averager(step_index, out_tensor)
            global_means = means["global"]
            parts: list[str] = []
            if "precip_liq_surf_mass_flux" in global_means:
                parts.append(
                    "liq="
                    f"{global_means['precip_liq_surf_mass_flux'] * _m_s_to_mm_day:.3f}"
                    " mm/day"
                )
            if "precip_ice_surf_mass_flux" in global_means:
                parts.append(
                    "ice="
                    f"{global_means['precip_ice_surf_mass_flux'] * _m_s_to_mm_day:.3f}"
                    " mm/day"
                )
            if parts:
                logger.info(
                    "step %d global mean precip: %s",
                    step_index + 1,
                    " ".join(parts),
                )

    if replay is None:
        run_scream(
            x0,
            coords,
            n_steps,
            model,
            pad,
            unpad,
            topology,
            omega_filter_strength=args.omega_filter_strength,
            omega_filter_coarsen=coarsen,
            F=f_nudge,
            write_step=write_step,
            sht_highpass=sht_highpass,
            before_step=before_step,
        )
    else:

        advance_hour = replay_stepper(
            model,
            steps_per_hour,
            pad,
            unpad,
            topology,
            write_step,
            omega_filter_strength=args.omega_filter_strength,
            omega_filter_coarsen=coarsen,
            sht_highpass=sht_highpass,
            before_step=before_step,
        )

        def increment(state, hour):
            if target_reader is not None:
                if producer is not None and args.driver_producer == "gated":
                    # the producer builds the next target from here on, on this GPU
                    torch.cuda.empty_cache()
                    mark(run_dir, "analysis", target_reader.valid(hour))
                blocks, info = target_reader.get(hour)
                replay.add_block_target(hour, torch.from_numpy(blocks))
                logger.info(
                    "replay target +%d h: waited %.1f s here (producer %.1f s, read "
                    "%.1f s)",
                    hour,
                    info["blocked_s"],
                    info["waited_s"],
                    info["read_s"],
                )
            forcing, stats = replay.increment(state[0], hour)
            for name in (
                n
                for n in replay.names
                if n in ("U_24", "PotentialTemperature_24", "qv_24", "T_2m")
            ):
                raw, low = stats[name]
                logger.info(
                    "replay +%d h %s: increment rms %.3g, low-passed %.3g",
                    hour,
                    name,
                    raw,
                    low,
                )
            return forcing

        run_with_replay(
            (x0, coords),
            plan,
            advance_hour,
            increment,
            copy=lambda state: (state[0].clone(), dict(state[1])),
        )
    for reader in (target_reader, ring_reader):
        if reader is not None:
            reader.close()
    if producer is not None:
        code = producer.close()
        logger.info("driver target producer exited with code %s", code)

    if writer is not None:
        writer.close()
    if regional_averager is not None:
        regional_averager.close()
    torch.distributed.barrier()
    if rank == 0 and args.zarr_output:
        zarr.consolidate_metadata(output_path)
        if use_plev_output:
            zarr.consolidate_metadata(plev_output_path)

    output_paths: list[str] = []
    if args.zarr_output:
        output_paths.append(output_path)
    if use_plev_output:
        output_paths.append(plev_output_path)
    if regional_averager is not None:
        output_paths.append(regional_output_path)
    logger.info(
        "Forecast done: %d steps = %.1f h — wrote %s",
        n_steps,
        dt_minutes * n_steps / 60,
        ", ".join(output_paths) if output_paths else "<no output>",
    )


if __name__ == "__main__":
    main()
