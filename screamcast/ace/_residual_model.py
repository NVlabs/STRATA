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
"""Reusable ACE forecast-residual model components."""

from __future__ import annotations

import netCDF4 as nc
import numpy as np
import torch
import torch.nn as nn

from screamcast.ace._channels import ACE_VARIABLE_NAMES
from screamcast.ace._finetune_utils import (
    ACE_STACK_TO_FME,
    ACEInputLayout,
    load_ace_artifacts_from_finetune_checkpoint,
)
from screamcast.ace._scream_to_ace import ScreamToACEState


class ACE2ForecastResidualSFNO(nn.Module):
    """ACE trunk + residual head for a reduced SCREAM variable layout."""

    def __init__(
        self,
        *,
        backbone: nn.Module,
        input_layout: ACEInputLayout,
        normalizer_means: dict[str, torch.Tensor],
        normalizer_stds: dict[str, torch.Tensor],
        ace_out_names: list[str],
        scream_variable_names: list[str],
        scream_hyam_sub: torch.Tensor,
        scream_hybm_sub: torch.Tensor,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.input_layout = input_layout
        self.ace_out_names = list(ace_out_names)
        self.scream_variable_names = list(scream_variable_names)
        for p in self.backbone.parameters():
            p.requires_grad = not freeze_backbone

        means = torch.stack(
            [normalizer_means[name].float() for name in input_layout.fme_in_names]
        )
        stds = torch.stack(
            [normalizer_stds[name].float() for name in input_layout.fme_in_names]
        )
        self.register_buffer("ace_mean", means.view(1, -1, 1, 1), persistent=False)
        self.register_buffer("ace_std", stds.view(1, -1, 1, 1), persistent=False)
        fme_index = {name: i for i, name in enumerate(input_layout.fme_in_names)}
        self.register_buffer(
            "ace_stack_out_indices",
            torch.tensor(
                [fme_index[ACE_STACK_TO_FME[name]] for name in ACE_VARIABLE_NAMES],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.register_buffer(
            "forcing_out_indices",
            torch.tensor(
                [fme_index[name] for name in input_layout.forcing_fme_names],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.scream_to_ace_state = ScreamToACEState(
            scream_variable_names=self.scream_variable_names,
            scream_hyam_sub=scream_hyam_sub,
            scream_hybm_sub=scream_hybm_sub,
        )
        decoder_hidden = self.backbone.decoder[0].out_channels
        decoder_in = self.backbone.decoder[0].in_channels
        self.scream_head = nn.Sequential(
            nn.Conv2d(decoder_in, decoder_hidden, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(decoder_hidden, len(self.scream_variable_names), 1, bias=False),
        )
        with torch.no_grad():
            self.scream_head[0].weight.copy_(self.backbone.decoder[0].weight)
            self.scream_head[0].bias.copy_(self.backbone.decoder[0].bias)
            nn.init.zeros_(self.scream_head[2].weight)

    @classmethod
    def from_checkpoint(
        cls, path: str, device: torch.device
    ) -> "ACE2ForecastResidualSFNO":
        """Load model from a checkpoint dictionary."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        (
            backbone,
            input_layout,
            ace_means,
            ace_stds,
            ace_out_names,
        ) = load_ace_artifacts_from_finetune_checkpoint(ckpt)
        scream_names = list(ckpt["scream_variable_names"])
        coords = ckpt["coords"]
        model = cls(
            backbone=backbone,
            input_layout=input_layout,
            normalizer_means=ace_means,
            normalizer_stds=ace_stds,
            ace_out_names=ace_out_names,
            scream_variable_names=scream_names,
            scream_hyam_sub=coords["hyam_sub"],
            scream_hybm_sub=coords["hybm_sub"],
            freeze_backbone=False,
        )
        model.load_state_dict(ckpt["model_state"], strict=True)
        return model.to(device).eval()

    def _decoder_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone.big_skip:
            residual = self.backbone.residual_filter_up(
                self.backbone.residual_filter_down(x)
            )
        x = self.backbone.encoder(x)
        if hasattr(self.backbone, "pos_embed"):
            if self.backbone.img_shape_loc != self.backbone.img_shape_eff:
                xp = torch.zeros_like(x)
                xp[
                    ...,
                    : self.backbone.img_shape_loc[0],
                    : self.backbone.img_shape_loc[1],
                ] = (
                    x[
                        ...,
                        : self.backbone.img_shape_loc[0],
                        : self.backbone.img_shape_loc[1],
                    ]
                    + self.backbone.pos_embed
                )
                x = xp
            else:
                x = x + self.backbone.pos_embed
        x = self.backbone.pos_drop(x)
        x = self.backbone._forward_features(x)
        if self.backbone.big_skip:
            x = torch.cat((x, residual), dim=1)
        return x

    def build_ace_input(
        self, scream_input_state: torch.Tensor, forcing: torch.Tensor
    ) -> torch.Tensor:
        ace_state = self.scream_to_ace_state(scream_input_state)
        bsz, _, lat, lon = ace_state.shape
        ace_input = ace_state.new_zeros((bsz, self.ace_mean.shape[1], lat, lon))
        ace_input[:, self.ace_stack_out_indices] = ace_state
        ace_input[:, self.forcing_out_indices] = forcing
        return ace_input

    def forward(
        self, scream_input_state: torch.Tensor, forcing: torch.Tensor
    ) -> torch.Tensor:
        ace_input = self.build_ace_input(scream_input_state, forcing)
        ace_norm = (ace_input - self.ace_mean) / self.ace_std
        trunk_features = self._decoder_input(ace_norm)
        return self.scream_head(trunk_features)


def load_training_tensors(
    *,
    data_path: str,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[str],
    list[np.datetime64],
]:
    with nc.Dataset(data_path, "r") as data:
        objective = str(getattr(data, "dataset_type", ""))
        if objective != "ace_rev2_training_pairs":
            raise ValueError(
                f"Unsupported dataset_type={objective!r} in {data_path}; "
                "expected 'ace_rev2_training_pairs'."
            )
        valid_times = np.array(
            np.asarray(data.variables["time"][:], dtype=np.int64), dtype="datetime64[s]"
        )
        hyam = torch.tensor(data.variables["hyam"][:], dtype=torch.float64)
        hybm = torch.tensor(data.variables["hybm"][:], dtype=torch.float64)
        forecast_state = np.asarray(
            data.variables["forecast_state"][:], dtype=np.float32
        )
        truth_state = np.asarray(data.variables["truth_state"][:], dtype=np.float32)
        lat = np.asarray(data.variables["lat"][:], dtype=np.float32)
        lon = np.asarray(data.variables["lon"][:], dtype=np.float32)
        scream_variable_names = str(data.scream_variable_names).split(",")
    return (
        torch.tensor(forecast_state, dtype=torch.float32),
        torch.tensor(truth_state, dtype=torch.float32),
        torch.tensor(lat, dtype=torch.float32),
        torch.tensor(lon, dtype=torch.float32),
        hyam,
        hybm,
        torch.empty((len(valid_times),), dtype=torch.int64),
        scream_variable_names,
        list(valid_times),
    )
