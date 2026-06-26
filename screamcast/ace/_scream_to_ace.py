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
"""SCREAM-to-ACE state conversion modules."""

from __future__ import annotations

import torch
import torch.nn as nn

from screamcast.ace._channels import P0_SCREAM
from screamcast.ace._vertical_coordinate import AK_ACE2_8L, BK_ACE2_8L
from screamcast.thermodynamics import temperature_from_potential_temperature
from screamcast.vertical_interpolation import regrid_hybrid_vertical


class ScreamToACEState(nn.Module):
    """Convert reduced SCREAM state tensors into the ACE stacked state layout."""

    def __init__(
        self,
        *,
        scream_variable_names: list[str],
        scream_hyam_sub: torch.Tensor,
        scream_hybm_sub: torch.Tensor,
    ):
        super().__init__()
        self.scream_variable_names = list(scream_variable_names)
        self.register_buffer(
            "scream_hyam_sub", scream_hyam_sub.to(torch.float64), persistent=False
        )
        self.register_buffer(
            "scream_hybm_sub", scream_hybm_sub.to(torch.float64), persistent=False
        )
        self.register_buffer("ace_ak", AK_ACE2_8L.clone(), persistent=False)
        self.register_buffer("ace_bk", BK_ACE2_8L.clone(), persistent=False)

        self._three_d_index: dict[str, list[tuple[int, int]]] = {}
        self._surface_index: dict[str, int] = {}
        for idx, name in enumerate(self.scream_variable_names):
            stem, sep, maybe_level = name.rpartition("_")
            if sep and maybe_level.isdigit():
                self._three_d_index.setdefault(stem, []).append((int(maybe_level), idx))
            else:
                self._surface_index[name] = idx

        required_3d = ["PotentialTemperature", "U", "V", "qv"]
        missing_3d = [name for name in required_3d if name not in self._three_d_index]
        if missing_3d:
            raise ValueError(
                f"Rev2 training variables are missing required 3D fields: {missing_3d}"
            )

        required_surface = ["T_2m", "ps"]
        missing_surface = [
            name for name in required_surface if name not in self._surface_index
        ]
        if missing_surface:
            raise ValueError(
                "Rev2 training variables are missing required surface fields: "
                f"{missing_surface}"
            )

        self._sorted_3d: dict[str, list[int]] = {}
        for name, entries in self._three_d_index.items():
            ordered = sorted(entries)
            self._sorted_3d[name] = [idx for _, idx in ordered]

    def _vregrid_batch(
        self, field_sub: torch.Tensor, surface_pressure: torch.Tensor
    ) -> torch.Tensor:
        field_lfirst = field_sub.movedim(1, 0)
        regridded, _ = regrid_hybrid_vertical(
            src_values=field_lfirst,
            src_ak_interface=self.scream_hyam_sub,
            src_bk_interface=self.scream_hybm_sub,
            surface_pressure=surface_pressure,
            src_p0=P0_SCREAM,
            target_ak_interface=self.ace_ak,
            target_bk_interface=self.ace_bk,
            target_p0=1.0,
        )
        return regridded.movedim(0, 1)

    def forward(self, scream_input_state: torch.Tensor) -> torch.Tensor:
        theta = scream_input_state[:, self._sorted_3d["PotentialTemperature"]]
        u = scream_input_state[:, self._sorted_3d["U"]]
        v = scream_input_state[:, self._sorted_3d["V"]]
        qv = scream_input_state[:, self._sorted_3d["qv"]]
        t2m = scream_input_state[
            :, self._surface_index["T_2m"] : self._surface_index["T_2m"] + 1
        ]
        ps = scream_input_state[:, self._surface_index["ps"]]
        q2m = qv[:, -1:]
        u10m = u[:, -1:]
        v10m = v[:, -1:]

        theta_ace_lfirst, p_ace_lfirst = regrid_hybrid_vertical(
            src_values=theta.movedim(1, 0),
            src_ak_interface=self.scream_hyam_sub,
            src_bk_interface=self.scream_hybm_sub,
            surface_pressure=ps,
            src_p0=P0_SCREAM,
            target_ak_interface=self.ace_ak,
            target_bk_interface=self.ace_bk,
            target_p0=1.0,
        )
        t_ace = temperature_from_potential_temperature(
            theta_ace_lfirst, p_ace_lfirst, p0_pa=P0_SCREAM
        ).movedim(0, 1)
        qtot_ace = self._vregrid_batch(qv, ps)
        u_ace = self._vregrid_batch(u, ps)
        v_ace = self._vregrid_batch(v, ps)
        return torch.cat(
            [
                t_ace,
                qtot_ace,
                u_ace,
                v_ace,
                ps.unsqueeze(1),
                # should be skin temperature but this is unavailable.
                # For SCREAM coupling assume this t2m=skt
                t2m,
                u10m,
                v10m,
                t2m,
                q2m,
            ],
            dim=1,
        ).to(dtype=scream_input_state.dtype)
