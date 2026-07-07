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
"""Tests for ScreamcastModel earth2studio wrapper.

Wrapper tests use a tiny untrained model initialized the same way as training.
The from_checkpoint test requires a real checkpoint path via the
SCREAMCAST_CHECKPOINT env var and is skipped when that variable is not set.
"""

import os
from collections import OrderedDict
from types import MethodType

import numpy as np
import pytest
import torch

if "PROJECT_ROOT" not in os.environ:
    pytest.skip("PROJECT_ROOT is not configured", allow_module_level=True)

from screamcast.dali_ext_src import ScreamV2
from screamcast.strata_wrappers import ScreamcastStrataBackbone
from screamcast.earth2studio_wrappers import ScreamcastModel
from screamcast.model_pipelines import MixedPredictionAsymmetric
from screamcast.normalization import RunningNorm2d

# ---------------------------------------------------------------------------
# Minimal model parameters shared across wrapper tests
# ---------------------------------------------------------------------------

_VARS_PROG = ("T_2m",)
_VARS_FORC = ("coszr",)
_VARS_DIAG = ("precip_ice_surf_mass_flux",)
_PLEVEL = 1
_LEVEL_START = 0
_LEVEL_END = 4  # needs ≥3 depth tokens for attn_kernel=3
_TILE_SIZE = 8
_NSIDE = 512


def _make_screamcast_model(tile_size: int = _TILE_SIZE) -> ScreamcastModel:
    """Construct a tiny untrained ScreamcastModel, mirroring training initialization."""
    levels = np.arange(_LEVEL_START, _LEVEL_END, _PLEVEL)
    num_depth_levels = len(levels)

    in_channels = ScreamV2.num_of_input_channels(
        _VARS_PROG, _VARS_FORC, _PLEVEL, _LEVEL_START, _LEVEL_END
    )
    out_channels = ScreamV2.num_of_output_channels(
        _VARS_PROG, _VARS_DIAG, _PLEVEL, _LEVEL_START, _LEVEL_END
    )

    # Build a tiny DiT (same factory as training) with do_concat_latitude=False
    # so no spatial index is needed for this untrained wrapper test.
    network = ScreamcastStrataBackbone(
        depth=num_depth_levels,
        height=tile_size,
        width=tile_size,
        patch_size=1,
        in_chans=in_channels,
        base_out_chans=out_channels,
        nside=_NSIDE,
        embed_dim=32,
        n_layers=1,
        num_heads=2,
        attn_kernel=3,
        do_rope_2d=False,
        do_concat_latitude=False,
        grid_type="healpix",
    )

    pipeline = MixedPredictionAsymmetric(
        network=network,
        loss_fn=torch.nn.MSELoss(),
        input_norm=RunningNorm2d(in_channels, fit_batches=20),
        target_norm=RunningNorm2d(out_channels, fit_batches=20),
        plevel=_PLEVEL,
        level_start=_LEVEL_START,
        level_end=_LEVEL_END,
        variables_prognostic=_VARS_PROG,
        variables_forcing=_VARS_FORC,
        variables_diagnostic=_VARS_DIAG,
        enable_3d_adapter=True,
    )
    pipeline.eval()

    return ScreamcastModel(
        pipeline=pipeline,
        tile_size=tile_size,
        levels=levels,
        variables_prognostic=_VARS_PROG,
        variables_forcing=_VARS_FORC,
        variables_diagnostic=_VARS_DIAG,
    )


# ---------------------------------------------------------------------------
# Wrapper tests (untrained model)
# ---------------------------------------------------------------------------


def test_wrapper_input_coords():
    model = _make_screamcast_model()
    coords = model.input_coords()

    assert set(coords.keys()) >= {"batch", "lead_time", "variable", "face", "x", "y"}
    assert coords["time"].shape == (0,)
    assert len(coords["x"]) == _TILE_SIZE
    assert len(coords["face"]) == 6
    assert coords["lead_time"][0] == np.timedelta64(0, "ns")

    # Input variables: prognostic + forcing (all 2D, so no level suffix)
    expected_vars = list(_VARS_PROG) + list(_VARS_FORC)
    assert list(coords["variable"]) == expected_vars


def test_wrapper_output_coords():
    model = _make_screamcast_model()
    in_coords = model.input_coords()
    out_coords = model.output_coords(in_coords)

    assert out_coords["lead_time"][0] == np.timedelta64(600, "s")

    # Output variables: prognostic + diagnostic (all 2D)
    expected_vars = list(_VARS_PROG) + list(_VARS_DIAG)
    assert list(out_coords["variable"]) == expected_vars


def test_wrapper_forward():
    tile_size = _TILE_SIZE
    model = _make_screamcast_model(tile_size=tile_size)

    in_coords = model.input_coords()
    in_channels = len(in_coords["variable"])
    out_channels = len(model.output_coords(in_coords)["variable"])

    x = torch.randn(2, in_channels, 1, tile_size, tile_size)

    coords = OrderedDict(in_coords)
    coords["face"] = np.array([0])
    coords["batch"] = np.empty(2)
    coords["time"] = np.array([np.datetime64("2020-10-13T00:00:00")])

    out, out_coords = model(x, coords)

    assert out.shape == (
        2,
        out_channels,
        1,
        tile_size,
        tile_size,
    ), f"Expected (2, {out_channels}, 1, {tile_size}, {tile_size}), got {out.shape}"
    assert out_coords["lead_time"][0] == np.timedelta64(600, "s")


def test_wrapper_forward_requires_time_for_autonomous_forcing():
    model = _make_screamcast_model(tile_size=_TILE_SIZE)
    in_coords = model.input_coords()
    in_channels = len(in_coords["variable"])
    x = torch.randn(1, in_channels, 1, _TILE_SIZE, _TILE_SIZE)

    coords = OrderedDict(in_coords)
    coords["face"] = np.array([0])
    coords["batch"] = np.empty(1)

    with pytest.raises(
        ValueError, match="coords\\['time'\\] must contain at least one timestamp"
    ):
        model(x, coords)


def test_create_iterator_initial_yield_fills_diagnostics_with_nan():
    model = _make_screamcast_model()
    in_coords = model.input_coords()
    in_channels = len(in_coords["variable"])
    out_channels = len(model.output_coords(in_coords)["variable"])

    x = torch.randn(1, in_channels, 1, _TILE_SIZE, _TILE_SIZE)

    coords = OrderedDict(in_coords)
    coords["face"] = np.array([0])
    coords["batch"] = np.empty(1)

    iterator = model.create_iterator(x, coords)
    x0, coords0 = next(iterator)

    assert x0.shape == (1, out_channels, 1, _TILE_SIZE, _TILE_SIZE)
    assert list(coords0["variable"]) == list(model.output_coords(in_coords)["variable"])
    assert torch.equal(x0[:, : model._n_prog_channels], x[:, : model._n_prog_channels])
    assert torch.isnan(x0[:, model._n_prog_channels :]).all()


def _synthetic_latlon(n_faces: int, tile_size: int) -> torch.Tensor:
    """Synthetic [n_faces, tile_size, tile_size, 2] lat/lon grid in degrees."""
    lat = (
        torch.linspace(-30.0, 30.0, tile_size)
        .view(1, tile_size, 1)
        .expand(n_faces, -1, tile_size)
    )
    lon = (
        torch.linspace(100.0, 160.0, tile_size)
        .view(1, 1, tile_size)
        .expand(n_faces, tile_size, -1)
    )
    return torch.stack([lat.contiguous(), lon.contiguous()], dim=-1)


def test_set_latlon_updates_geometry():
    model = _make_screamcast_model()
    assert model._n_faces == 6

    new_lat = torch.zeros(1, _TILE_SIZE, _TILE_SIZE)
    new_lon = torch.zeros(1, _TILE_SIZE, _TILE_SIZE)
    model.set_latlon(new_lat, new_lon)

    assert model._n_faces == 1
    assert model.tile_size == _TILE_SIZE
    assert model.latlon.shape == (1, _TILE_SIZE, _TILE_SIZE, 2)
    assert len(model.input_coords()["face"]) == 1
    assert len(model.input_coords()["x"]) == _TILE_SIZE


def test_set_latlon_multiple_calls_preserve_source():
    """Second set_latlon call must re-grid from _src_latlon, not from the previous target."""
    src_latlon = _synthetic_latlon(6, _TILE_SIZE)
    model = _make_screamcast_model()
    model.register_buffer("_src_latlon", src_latlon, persistent=False)

    model.set_latlon(
        torch.zeros(1, _TILE_SIZE, _TILE_SIZE), torch.zeros(1, _TILE_SIZE, _TILE_SIZE)
    )
    model.set_latlon(
        torch.ones(1, _TILE_SIZE, _TILE_SIZE), torch.ones(1, _TILE_SIZE, _TILE_SIZE)
    )

    assert torch.equal(model._src_latlon, src_latlon), "_src_latlon must not be mutated"


def test_reset_latlon_restores_original():
    src_latlon = _synthetic_latlon(6, _TILE_SIZE)
    model = ScreamcastModel(
        pipeline=_make_screamcast_model().pipeline,
        tile_size=_TILE_SIZE,
        levels=np.arange(_LEVEL_START, _LEVEL_END, _PLEVEL),
        variables_prognostic=_VARS_PROG,
        variables_forcing=_VARS_FORC,
        variables_diagnostic=_VARS_DIAG,
        latlon=src_latlon,
    )

    model.set_latlon(
        torch.zeros(1, _TILE_SIZE, _TILE_SIZE), torch.zeros(1, _TILE_SIZE, _TILE_SIZE)
    )
    assert model._n_faces == 1

    model.reset_latlon()

    assert model._n_faces == 6
    assert model.tile_size == _TILE_SIZE
    assert torch.equal(model.latlon, src_latlon)
    assert len(model.input_coords()["face"]) == 6


def test_reset_latlon_raises_without_src():
    model = _make_screamcast_model()
    with pytest.raises(ValueError, match="reset_latlon"):
        model.reset_latlon()


def test_set_latlon_raises_when_static_forcing_missing():
    model = _make_screamcast_model()
    model._static_forcing_names = ["phis"]  # declared but no tensor provided

    with pytest.raises(ValueError, match="set_latlon cannot re-grid static forcing"):
        model.set_latlon(
            torch.zeros(1, _TILE_SIZE, _TILE_SIZE),
            torch.zeros(1, _TILE_SIZE, _TILE_SIZE),
        )


def test_create_iterator_updates_coszr_when_time_present():
    model = _make_screamcast_model()
    in_coords = model.input_coords()
    in_channels = len(in_coords["variable"])
    x = torch.randn(1, in_channels, 1, _TILE_SIZE, _TILE_SIZE)

    coords = OrderedDict(in_coords)
    coords["face"] = np.array([0])
    coords["batch"] = np.empty(1)
    coords["time"] = np.array([np.datetime64("2020-10-13T00:00:00")])

    seen_coords = []

    def fake_update_coszr(self, x_in, coords_in):
        seen_coords.append((coords_in["time"][0], coords_in["lead_time"][0]))
        return x_in

    model.update_coszr = MethodType(fake_update_coszr, model)

    iterator = model.create_iterator(x, coords)
    next(iterator)
    next(iterator)

    assert seen_coords == [
        (np.datetime64("2020-10-13T00:00:00"), np.timedelta64(600, "s"))
    ]


def test_persistence_model_matches_screamcast_metadata():
    model = _make_screamcast_model()
    persistence = model.to_persistence()

    assert list(persistence.input_coords()["variable"]) == list(
        model.input_coords()["variable"]
    )
    assert list(persistence.output_coords(model.input_coords())["variable"]) == list(
        model.output_coords(model.input_coords())["variable"]
    )
    assert persistence.dt == model.dt


def test_persistence_model_forward_copies_prognostic_and_zeros_diagnostics():
    model = _make_screamcast_model(tile_size=_TILE_SIZE)
    persistence = model.to_persistence()
    in_coords = persistence.input_coords()

    x = torch.randn(2, len(in_coords["variable"]), 1, _TILE_SIZE, _TILE_SIZE)
    coords = OrderedDict(in_coords)
    coords["face"] = np.array([0])
    coords["batch"] = np.empty(2)
    coords["time"] = np.array([np.datetime64("2020-10-13T00:00:00")])

    out, out_coords = persistence(x, coords)

    assert torch.equal(out[:, : persistence._n_prog_channels], x)
    assert (out[:, persistence._n_prog_channels :] == 0).all()
    assert out_coords["lead_time"][0] == np.timedelta64(600, "s")


def test_persistence_iterator_advances_lead_time():
    model = _make_screamcast_model(tile_size=_TILE_SIZE)
    persistence = model.to_persistence()
    in_coords = persistence.input_coords()

    x = torch.randn(1, len(in_coords["variable"]), 1, _TILE_SIZE, _TILE_SIZE)
    coords = OrderedDict(in_coords)
    coords["face"] = np.array([0])
    coords["batch"] = np.empty(1)
    coords["time"] = np.array([np.datetime64("2020-10-13T00:00:00")])

    iterator = persistence.create_iterator(x, coords)
    x0, coords0 = next(iterator)
    x1, coords1 = next(iterator)

    assert torch.equal(x0[:, : persistence._n_prog_channels], x)
    assert (x0[:, persistence._n_prog_channels :] == 0).all()
    assert torch.equal(x1[:, : persistence._n_prog_channels], x)
    assert (x1[:, persistence._n_prog_channels :] == 0).all()
    assert coords0["lead_time"][0] == np.timedelta64(0, "ns")
    assert coords1["lead_time"][0] == persistence.dt


# ---------------------------------------------------------------------------
# from_checkpoint test (requires real checkpoint)
# ---------------------------------------------------------------------------


def test_from_checkpoint():
    checkpoint_path = os.environ.get("SCREAMCAST_CHECKPOINT")
    if not checkpoint_path:
        pytest.skip("Set SCREAMCAST_CHECKPOINT to run this test")

    model = ScreamcastModel.from_checkpoint(checkpoint_path)
    model = model.cuda()

    in_coords = model.input_coords()
    in_channels = len(in_coords["variable"])
    out_channels = len(model.output_coords(in_coords)["variable"])
    tile_size = len(in_coords["x"])

    x = torch.randn(2, in_channels, 1, tile_size, tile_size).cuda()

    coords = OrderedDict(in_coords)
    coords["face"] = np.array([0])
    coords["batch"] = np.empty(2)

    out, _ = model(x, coords)

    assert out.shape == (
        2,
        out_channels,
        1,
        tile_size,
        tile_size,
    ), f"Expected (2, {out_channels}, 1, {tile_size}, {tile_size}), got {out.shape}"
