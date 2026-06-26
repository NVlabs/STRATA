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
from types import SimpleNamespace

import netCDF4 as nc
import numpy as np
import torch
from fme.ace.models.modulus.sfnonet import SphericalFourierNeuralOperatorNet

from screamcast.ace import _train as train_mod
from screamcast.ace import train
from screamcast.ace._channels import ACE_VARIABLE_NAMES
from screamcast.ace._finetune_utils import ACEInputLayout
from screamcast.ace._residual_model import ACE2ForecastResidualSFNO

REV2_LEVELS = list(range(8, 32))
REV2_VARIABLE_NAMES = [
    f"{name}_{level}"
    for name in ["PotentialTemperature", "U", "V", "z_mid", "omega", "qv"]
    for level in REV2_LEVELS
] + ["T_2m", "ps"]


def _build_backbone(
    img_shape: tuple[int, int] = (2, 3),
) -> SphericalFourierNeuralOperatorNet:
    return SphericalFourierNeuralOperatorNet(
        params=SimpleNamespace(),
        img_shape=img_shape,
        in_chans=44,
        out_chans=50,
        embed_dim=16,
        num_layers=1,
        encoder_layers=1,
        num_blocks=4,
        spectral_layers=1,
        operator_type="dhconv",
        filter_type="linear",
        spectral_transform="sht",
        normalization_layer="instance_norm",
        hard_thresholding_fraction=1.0,
        use_mlp=True,
        pos_embed=True,
        big_skip=True,
    )


def _build_layout() -> ACEInputLayout:
    fme_in_names = [
        "land_fraction",
        "ocean_fraction",
        "sea_ice_fraction",
        "DSWRFtoa",
        "HGTsfc",
        "global_mean_co2",
        "PRESsfc",
        "surface_temperature",
        "TMP2m",
        "Q2m",
        "UGRD10m",
        "VGRD10m",
        *[f"air_temperature_{k}" for k in range(8)],
        *[f"specific_total_water_{k}" for k in range(8)],
        *[f"eastward_wind_{k}" for k in range(8)],
        *[f"northward_wind_{k}" for k in range(8)],
    ]
    return ACEInputLayout(
        stack_names=list(ACE_VARIABLE_NAMES),
        fme_in_names=fme_in_names,
        forcing_fme_names=[
            "land_fraction",
            "ocean_fraction",
            "sea_ice_fraction",
            "DSWRFtoa",
            "HGTsfc",
            "global_mean_co2",
        ],
    )


def _build_out_names() -> list[str]:
    return [
        "PRESsfc",
        "surface_temperature",
        *[f"air_temperature_{k}" for k in range(8)],
        *[f"specific_total_water_{k}" for k in range(8)],
        *[f"eastward_wind_{k}" for k in range(8)],
        *[f"northward_wind_{k}" for k in range(8)],
        "LHTFLsfc",
        "SHTFLsfc",
        "PRATEsfc",
        "ULWRFsfc",
        "ULWRFtoa",
        "DLWRFsfc",
        "DSWRFsfc",
        "USWRFsfc",
        "USWRFtoa",
        "tendency_of_total_water_path_due_to_advection",
        "TMP850",
        "h500",
        "TMP2m",
        "Q2m",
        "UGRD10m",
        "VGRD10m",
    ]


def _build_stats(layout: ACEInputLayout, out_names: list[str]):
    means = {name: torch.tensor(0.0) for name in layout.fme_in_names}
    stds = {name: torch.tensor(1.0) for name in layout.fme_in_names}
    for name in out_names:
        means.setdefault(name, torch.tensor(0.0))
        stds.setdefault(name, torch.tensor(1.0))
    means["PRESsfc"] = torch.tensor(100000.0)
    means["TMP2m"] = torch.tensor(280.0)
    means["surface_temperature"] = torch.tensor(285.0)
    return means, stds


def _write_fake_rev2_dataset(path):
    n_times = 6
    n_channels = len(REV2_VARIABLE_NAMES)
    root = nc.Dataset(path, "w")
    root.createDimension("time", n_times)
    root.createDimension("scream_channel", n_channels)
    root.createDimension("lat", 2)
    root.createDimension("lon", 3)
    root.createDimension("level", len(REV2_LEVELS))
    root.dataset_type = "ace_rev2_training_pairs"
    root.scream_variable_names = ",".join(REV2_VARIABLE_NAMES)
    root.createVariable("time", "i8", ("time",))[:] = np.arange(
        1, n_times + 1, dtype=np.int64
    )
    root.createVariable("lat", "f4", ("lat",))[:] = np.array(
        [-5.0, 5.0], dtype=np.float32
    )
    root.createVariable("lon", "f4", ("lon",))[:] = np.array(
        [0.0, 10.0, 20.0], dtype=np.float32
    )
    root.createVariable("hyam", "f8", ("level",))[:] = np.linspace(
        0.20, 0.98, len(REV2_LEVELS), dtype=np.float64
    )
    root.createVariable("hybm", "f8", ("level",))[:] = np.linspace(
        0.80, 0.02, len(REV2_LEVELS), dtype=np.float64
    )
    forecast = (
        np.random.default_rng(0)
        .standard_normal((n_times, n_channels, 2, 3))
        .astype(np.float32)
    )
    truth = forecast + 0.1
    root.createVariable(
        "forecast_state", "f4", ("time", "scream_channel", "lat", "lon")
    )[:] = forecast
    root.createVariable("truth_state", "f4", ("time", "scream_channel", "lat", "lon"))[
        :
    ] = truth
    root.close()


def test_rev2_training_runs_on_preprocessed_dataset(tmp_path, monkeypatch):
    data_path = tmp_path / "rev2_pairs.nc"
    out_path = tmp_path / "rev2.pt"
    _write_fake_rev2_dataset(str(data_path))

    layout = _build_layout()
    out_names = _build_out_names()
    means, stds = _build_stats(layout, out_names)

    monkeypatch.setattr(
        train_mod,
        "fetch_static_ace_forcing",
        lambda **_kwargs: torch.stack(
            [torch.full((2, 3), float(i + 1)) for i in range(5)], dim=0
        ),
    )
    monkeypatch.setattr(
        train_mod,
        "load_ace_backbone",
        lambda _checkpoint: (_build_backbone(), layout, means, stds),
    )
    monkeypatch.setattr(train_mod, "load_ace_out_names", lambda _checkpoint: out_names)

    args = SimpleNamespace(
        data=str(data_path),
        ace_checkpoint="unused.pt",
        output=str(out_path),
        epochs=1,
        batch_size=2,
        lr=1e-3,
        weight_decay=0.0,
        save_every=1,
        device="cpu",
        train_backbone=False,
    )
    train(args)

    ckpt = torch.load(out_path, map_location="cpu", weights_only=True)
    assert ckpt["objective"] == "forecast_residual_on_ace_grid"
    assert ckpt["residual_scale"].shape[0] == len(REV2_VARIABLE_NAMES)
    assert list(ckpt["ace_forcing_names"]) == layout.forcing_fme_names


def test_build_ace_input_preserves_model_dtype_with_float64_vertical_coords():
    layout = _build_layout()
    out_names = _build_out_names()
    means, stds = _build_stats(layout, out_names)
    model = ACE2ForecastResidualSFNO(
        backbone=_build_backbone(),
        input_layout=layout,
        normalizer_means=means,
        normalizer_stds=stds,
        ace_out_names=out_names,
        scream_variable_names=list(REV2_VARIABLE_NAMES),
        scream_hyam_sub=torch.linspace(
            0.20, 0.98, len(REV2_LEVELS), dtype=torch.float64
        ),
        scream_hybm_sub=torch.linspace(
            0.80, 0.02, len(REV2_LEVELS), dtype=torch.float64
        ),
        freeze_backbone=False,
    ).eval()

    scream_input_state = torch.randn(2, len(REV2_VARIABLE_NAMES), 2, 3)
    forcing = torch.randn(2, len(layout.forcing_fme_names), 2, 3)

    ace_input = model.build_ace_input(scream_input_state, forcing)

    assert ace_input.dtype == scream_input_state.dtype
    assert torch.equal(ace_input[:, model.forcing_out_indices], forcing)
