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
"""Unit tests for the pressure-level output pipeline used by
``scripts/ace/run_screamcast_nudged.py``.

Two concerns that were debugged the hard way during development and deserve
regression coverage:

1. ``hyam``/``hybm`` alignment with checkpoint output levels. The zarr training
   data is pre-subsampled with ``slice(3, 128, 4)``,
   so a channel level label like ``8`` is an index into the 32-level subsampled
   array, NOT the full 128-level ``scream_vertical_coordinate.nc``. Indexing
   with the wrong convention silently produces results at the wrong altitude.

2. The patch-shaped 5-D broadcasting inside
   ``run_screamcast_nudged.write_step`` —
   ``(n_src_lev, 1, tiles_per_rank, tile_size, tile_size)`` — that feeds
   ``log_pressure_interpolate``. Easy to get an axis wrong and produce
   geometrically plausible but physically wrong output.
"""

from collections import OrderedDict

import numpy as np
import torch

from screamcast.ace import P0_SCREAM
from screamcast.vertical_interpolation import log_pressure_interpolate
from screamcast.zarr_writer import _split_variable_levels


def _synthetic_hyam_hybm_128() -> tuple[np.ndarray, np.ndarray]:
    """A stand-in for scream_vertical_coordinate.nc (128 levels).

    Mimics SCREAM's convention: ``hyam`` decreases top->surface,
    ``hybm`` increases 0->1. At a ~1013 hPa surface pressure each column spans
    roughly 5 hPa (top) to 1013 hPa (bottom).
    """
    n = 128
    hyam = np.linspace(5e-3, 1e-5, n, dtype=np.float32)  # decreasing top->sfc
    hybm = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return hyam, hybm


# ---------------------------------------------------------------------------
# hyam/hybm alignment against checkpoint output levels
# ---------------------------------------------------------------------------


def test_subsampled_hyam_matches_128_direct_indexing():
    """`hyam_full[3::4][k]` must equal `hyam_full[3 + 4*k]` for every k.

    This is the invariant that makes
    ``plev_hyam_t = torch.from_numpy(hyam[output_sigma_levels])`` correct in
    ``run_screamcast_nudged.py`` — ``hyam`` is already the ``[3::4]``
    subsampled array and ``output_sigma_levels`` are indices into that.
    """
    hyam, hybm = _synthetic_hyam_hybm_128()
    hyam_sub = hyam[3::4]
    hybm_sub = hybm[3::4]
    assert hyam_sub.shape == (32,)
    for k in range(32):
        orig = 3 + 4 * k
        np.testing.assert_allclose(hyam_sub[k], hyam[orig], rtol=0, atol=0)
        np.testing.assert_allclose(hybm_sub[k], hybm[orig], rtol=0, atol=0)


def test_checkpoint_level_labels_map_to_correct_original_levels():
    """For a checkpoint with output levels [8, 9, ..., 31] on the subsampled
    grid, the resulting pressures must match direct indexing of the full
    128-level array at original positions [35, 39, ..., 127]."""
    hyam, hybm = _synthetic_hyam_hybm_128()
    hyam_sub = hyam[3::4]
    hybm_sub = hybm[3::4]

    output_sigma_levels = list(range(8, 32))  # e.g. unfreeze_3src_4stepft
    ps_Pa = np.float32(101325.0)

    # Path taken by run_screamcast_nudged.py (indexing subsampled array):
    p_via_subsampled = hyam_sub[output_sigma_levels] * P0_SCREAM + (
        hybm_sub[output_sigma_levels] * ps_Pa
    )

    # Path of direct indexing the full 128-level file:
    original_indices = [3 + 4 * k for k in output_sigma_levels]
    p_direct = hyam[original_indices] * P0_SCREAM + hybm[original_indices] * ps_Pa

    np.testing.assert_allclose(p_via_subsampled, p_direct, rtol=0, atol=0)
    # Sanity: pressures should be ascending (monotone-increasing hyam+hybm).
    assert np.all(np.diff(p_via_subsampled) > 0)


def test_wrong_indexing_convention_produces_wrong_pressures():
    """Catch regressions where someone indexes the FULL 128-level array with
    the channel level labels directly (i.e. ``hyam_full[8:32]``) — that gives
    stratosphere-only pressures that bear no relation to the zarr's data."""
    hyam, hybm = _synthetic_hyam_hybm_128()
    hyam_sub = hyam[3::4]
    hybm_sub = hybm[3::4]
    output_sigma_levels = list(range(8, 32))
    ps_Pa = np.float32(101325.0)

    correct = hyam_sub[output_sigma_levels] * P0_SCREAM + (
        hybm_sub[output_sigma_levels] * ps_Pa
    )
    wrong_direct = hyam[output_sigma_levels] * P0_SCREAM + (
        hybm[output_sigma_levels] * ps_Pa
    )

    # The two interpretations disagree everywhere in the troposphere/stratosphere.
    max_diff = float(np.max(np.abs(correct - wrong_direct)))
    assert max_diff > 10000.0, (
        "expected the two indexing conventions to differ by tens of kPa; "
        f"got max diff {max_diff:.2f} Pa"
    )


# ---------------------------------------------------------------------------
# _split_variable_levels entry order and ascending-sort invariant used by
# run_screamcast_nudged.py's plev setup
# ---------------------------------------------------------------------------


def test_split_variable_levels_yields_ascending_channel_index_order():
    """The plev setup's ``sorted(level for level, _ in first_3d_entries)``
    must produce the levels the model actually emitted, in ascending order."""
    out_vars = (
        [f"U_{lv}" for lv in range(8, 32)] + [f"V_{lv}" for lv in range(8, 32)] + ["ps"]
    )
    grouped = _split_variable_levels(out_vars)
    assert list(grouped.keys()) == ["U", "V", "ps"]

    # 3D entries appear in channel-index order, which for SCREAMcast outputs
    # is also level-ascending. Sorting must be a no-op for this canonical case.
    u_levels_raw = [lv for lv, _ in grouped["U"]]
    u_levels_sorted = sorted(lv for lv, _ in grouped["U"])
    assert u_levels_raw == list(range(8, 32))
    assert u_levels_sorted == u_levels_raw


def test_split_variable_levels_sort_recovers_ascending_even_if_scrambled():
    """Even if a future model outputs 3D channels in a non-ascending order,
    sorting recovers the right level-axis. Guards against brittle assumptions
    in the setup code."""
    out_vars = ["U_30", "U_8", "U_20"] + ["ps"]
    grouped = _split_variable_levels(out_vars)
    sorted_levels = sorted(lv for lv, _ in grouped["U"])
    assert sorted_levels == [8, 20, 30]


# ---------------------------------------------------------------------------
# Patch-shaped vertical interpolation with the same 5-D broadcasting shape
# used in run_screamcast_nudged.write_step
# ---------------------------------------------------------------------------


def _make_analytic_patches(
    n_src_lev: int,
    tpr: int,
    ts: int,
    hyam_sub: np.ndarray,
    hybm_sub: np.ndarray,
    output_sigma_levels: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a local-tiles tensor ``(1, C, tpr, ts, ts)`` whose 3D ``U`` values
    follow a known analytic log-pressure profile, so that after vertical interp
    we can check the result in closed form.

    Channel layout (same as _split_variable_levels groups):
      0..n_src_lev-1  : U at output_sigma_levels
      n_src_lev       : ps
    """
    # Build spatial ps field: vary across tiles/pixels to exercise broadcasting.
    rng = np.random.default_rng(0)
    ps_np = rng.uniform(90000.0, 103000.0, size=(tpr, ts, ts)).astype(np.float32)
    ps = torch.from_numpy(ps_np).unsqueeze(0)  # (1, tpr, ts, ts)

    # Reconstruct pressure per column from hyam/hybm at the output levels.
    hyam_out = hyam_sub[output_sigma_levels].astype(np.float32)
    hybm_out = hybm_sub[output_sigma_levels].astype(np.float32)
    # Shape (n_src_lev, tpr, ts, ts): ak*P0 + bk*ps
    p = (
        hyam_out[:, None, None, None] * P0_SCREAM
        + hybm_out[:, None, None, None] * ps_np[None, :, :, :]
    )
    # Analytic profile: T = 300 - 10*log(p/P0) — linear in log-pressure,
    # so log-pressure interpolation is exact to float precision.
    u_values = 300.0 - 10.0 * np.log(p / P0_SCREAM)
    u_tensor = torch.from_numpy(u_values.astype(np.float32)).unsqueeze(0)
    # (1, n_src_lev, tpr, ts, ts) -> channels ordered like out_tensor[:, U_chs]
    # In out_tensor layout we need (1, C, tpr, ts, ts) with U occupying
    # channels 0..n_src_lev-1 and ps at index n_src_lev.
    out_tensor = torch.cat([u_tensor, ps.unsqueeze(1)], dim=1)
    # ps also returned explicitly for use in the expected answer:
    return out_tensor, ps, torch.from_numpy(hyam_out), torch.from_numpy(hybm_out)


def test_write_step_vertical_interp_5d_broadcasting_analytic():
    """Exercise the exact broadcasting pattern used in
    ``run_screamcast_nudged.write_step``:
        src_pressure  = (n_src_lev, 1, tpr, ts, ts)
        src_vals      = (n_src_lev, 1, tpr, ts, ts)
        target_pressure = (n_plev, 1, tpr, ts, ts)

    With a log-pressure-linear profile ``T = 300 - 10*log(p/P0)``,
    ``log_pressure_interpolate`` is exact up to float precision.
    """
    hyam, hybm = _synthetic_hyam_hybm_128()
    hyam_sub = hyam[3::4]
    hybm_sub = hybm[3::4]
    # Use the full 32 subsampled levels so the synthetic column spans roughly
    # 5 hPa to 1013 hPa — covers common met target pressures without clamping.
    output_sigma_levels = list(range(0, 32))
    n_src_lev = len(output_sigma_levels)

    tpr, ts = 3, 4
    out_tensor, ps_local, plev_hyam_t, plev_hybm_t = _make_analytic_patches(
        n_src_lev=n_src_lev,
        tpr=tpr,
        ts=ts,
        hyam_sub=hyam_sub,
        hybm_sub=hybm_sub,
        output_sigma_levels=output_sigma_levels,
    )

    # Build a grouped dict matching write_step's expectation.
    # U channels are 0..n_src_lev-1, ps is n_src_lev.
    plev_grouped = OrderedDict(
        [
            ("U", [(lv, i) for i, lv in enumerate(output_sigma_levels)]),
            ("ps", [(None, n_src_lev)]),
        ]
    )

    # Mirror the production code's broadcasting.
    ps_field = out_tensor[:, n_src_lev]  # (1, tpr, ts, ts)
    src_pressure = plev_hyam_t.view(-1, 1, 1, 1, 1) * P0_SCREAM + plev_hybm_t.view(
        -1, 1, 1, 1, 1
    ) * ps_field.unsqueeze(0)
    target_hpa = [850.0, 500.0, 200.0, 50.0]
    plev_pa_t = torch.tensor([p * 100.0 for p in target_hpa], dtype=torch.float32)
    target_pressure = plev_pa_t.view(-1, 1, 1, 1, 1).expand(-1, *ps_field.shape)

    channel_indices = [ci for _, ci in plev_grouped["U"]]
    src_vals = out_tensor[0, channel_indices].unsqueeze(1)
    assert src_vals.shape == (n_src_lev, 1, tpr, ts, ts)
    assert src_pressure.shape == src_vals.shape

    remapped = log_pressure_interpolate(
        src_values=src_vals,
        src_pressure=src_pressure,
        target_pressure=target_pressure,
        axis=0,
    )
    assert remapped.shape == (len(target_hpa), 1, tpr, ts, ts)

    expected = 300.0 - 10.0 * torch.log(target_pressure / P0_SCREAM)
    # Log-pressure-linear source ⇒ interpolation is exact to float precision.
    torch.testing.assert_close(
        remapped.to(torch.float64),
        expected.to(torch.float64),
        rtol=1e-5,
        atol=1e-4,
    )

    # Verify ps pass-through works as the write_step concat step does it:
    ps_chan = out_tensor[:, n_src_lev : n_src_lev + 1]
    assert ps_chan.shape == (1, 1, tpr, ts, ts)
    torch.testing.assert_close(ps_chan[0, 0], ps_local[0])


def test_write_step_permute_produces_channel_first_plev_layout():
    """``remapped.permute(1, 0, 2, 3, 4)`` must put n_plev on the channel axis,
    matching the raw zarr's 3D layout ``(batch, plev, tpr, ts, ts)`` and the
    ``plev_local = torch.cat(..., dim=1)`` concat in write_step."""
    n_plev, B, tpr, ts = 4, 1, 3, 4
    remapped = torch.randn(n_plev, B, tpr, ts, ts)
    permuted = remapped.permute(1, 0, 2, 3, 4)
    assert permuted.shape == (B, n_plev, tpr, ts, ts)
    # Spot-check value preservation.
    for b in range(B):
        for p in range(n_plev):
            torch.testing.assert_close(permuted[b, p], remapped[p, b])
