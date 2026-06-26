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
"""ACE-specific Earth2Studio wrappers."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch

from screamcast.ace._finetune_utils import reorder_forcing_tensor
from screamcast.ace._forcing import build_ace_forcing_tensor
from screamcast.ace._residual_model import ACE2ForecastResidualSFNO


class ACE2ForecastResidualModel(torch.nn.Module):
    """Inference wrapper for the ACE forecast-residual model."""

    def __init__(
        self,
        *,
        model: ACE2ForecastResidualSFNO,
        residual_scale: torch.Tensor,
        forcing_variable_names: list[str],
        model_forcing_variable_names: list[str],
        ace_lat: np.ndarray,
        ace_lon: np.ndarray,
        static_forcing: torch.Tensor,
    ) -> None:
        super().__init__()
        self.model = model.eval()
        self.register_buffer(
            "residual_scale",
            residual_scale.to(torch.float32).view(1, -1, 1, 1),
            persistent=False,
        )
        self.forcing_variable_names = list(forcing_variable_names)
        self.model_forcing_variable_names = list(model_forcing_variable_names)
        self.ace_lat = np.asarray(ace_lat, dtype=np.float32)
        self.ace_lon = np.asarray(ace_lon, dtype=np.float32)
        self.register_buffer(
            "static_forcing",
            static_forcing.to(torch.float32),
            persistent=False,
        )
        self._input_variable_names = np.asarray(
            self.model.scream_variable_names, dtype=object
        )

    def input_coords(self) -> OrderedDict:
        return OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0, dtype="datetime64[ns]"),
                "lead_time": np.array([np.timedelta64(0, "ns")]),
                "variable": self._input_variable_names,
                "lat": self.ace_lat,
                "lon": self.ace_lon,
            }
        )

    def output_coords(self, input_coords: OrderedDict) -> OrderedDict:
        return OrderedDict(input_coords)

    def _build_full_forcing(
        self,
        coords: OrderedDict,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if "time" not in coords or len(coords["time"]) == 0:
            raise ValueError("coords['time'] must contain at least one timestamp.")
        lead_time = coords.get("lead_time", np.array([np.timedelta64(0, "ns")]))
        if len(lead_time) == 0:
            raise ValueError("coords['lead_time'] must contain at least one offset.")
        valid_time = coords["time"][0] + lead_time[-1]
        return build_ace_forcing_tensor(
            time=valid_time,
            lat=self.ace_lat,
            lon=self.ace_lon,
            forcing_names=self.forcing_variable_names,
            static_forcing=self.static_forcing,
            batch_size=batch,
            device=device,
            dtype=dtype,
        )

    def _validate_latlon_coords(self, coords: OrderedDict, x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(
                "ACE2ForecastResidualModel expects input shaped [batch, variable, lat, lon], "
                f"got {tuple(x.shape)}."
            )
        if x.shape[-2:] != (len(self.ace_lat), len(self.ace_lon)):
            raise ValueError(
                "ACE2ForecastResidualModel expects ACE lat/lon spatial dimensions "
                f"({len(self.ace_lat)}, {len(self.ace_lon)}), got {tuple(x.shape[-2:])}."
            )
        if "lat" in coords and not np.array_equal(
            np.asarray(coords["lat"]), self.ace_lat
        ):
            raise ValueError(
                "coords['lat'] does not match the checkpoint ACE latitude grid."
            )
        if "lon" in coords and not np.array_equal(
            np.asarray(coords["lon"]), self.ace_lon
        ):
            raise ValueError(
                "coords['lon'] does not match the checkpoint ACE longitude grid."
            )

    def __call__(
        self,
        x: torch.Tensor,
        coords: OrderedDict,
    ) -> tuple[torch.Tensor, OrderedDict]:
        expected_vars = list(self._input_variable_names)
        if "variable" in coords and list(coords["variable"]) != expected_vars:
            raise ValueError(
                "Input coordinate variables do not match the wrapper's expected "
                f"channel order. Expected {expected_vars}, got {list(coords['variable'])}."
            )

        self._validate_latlon_coords(coords, x)
        forcing = self._build_full_forcing(coords, x.shape[0], x.device, x.dtype)
        forcing = reorder_forcing_tensor(
            forcing,
            list(self.forcing_variable_names),
            list(self.model_forcing_variable_names),
        )

        with torch.inference_mode():
            correction_norm = self.model(x, forcing)
            correction_latlon = (
                self.residual_scale.to(device=x.device, dtype=x.dtype) * correction_norm
            )

        return correction_latlon, self.output_coords(coords)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str | torch.device = "cpu",
    ) -> "ACE2ForecastResidualModel":
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "static_ace_forcing" not in ckpt:
            raise ValueError(
                "Checkpoint is missing 'static_ace_forcing', which is required for inference."
            )
        model = ACE2ForecastResidualSFNO.from_checkpoint(
            checkpoint_path,
            device=torch.device("cpu"),
        )
        wrapper = cls(
            model=model,
            residual_scale=torch.as_tensor(ckpt["residual_scale"]),
            forcing_variable_names=list(ckpt["ace_forcing_names"]),
            model_forcing_variable_names=list(model.input_layout.forcing_fme_names),
            ace_lat=np.asarray(ckpt["coords"]["lat"], dtype=np.float32),
            ace_lon=np.asarray(ckpt["coords"]["lon"], dtype=np.float32),
            static_forcing=torch.as_tensor(
                ckpt["static_ace_forcing"], dtype=torch.float32
            ),
        )
        return wrapper.to(device).eval()
