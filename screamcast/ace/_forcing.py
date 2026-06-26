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
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from earth2studio.data import fetch_data
from earth2studio.data.ace2 import ACE2ERA5Data
from earth2studio.lexicon.ace import ACELexicon

from screamcast.astronomy import calculate_dswrftoa

AUTONOMOUS_ACE_FORCING: frozenset[str] = frozenset({"DSWRFtoa"})
STATIC_ACE_FORCING: frozenset[str] = frozenset(
    {"land_fraction", "ocean_fraction", "sea_ice_fraction", "HGTsfc", "global_mean_co2"}
)


def fetch_static_ace_forcing(
    *,
    time: np.datetime64,
    forcing_names: Sequence[str],
    lat: np.ndarray,
    lon: np.ndarray,
) -> torch.Tensor:
    """Fetch time-invariant ACE forcing channels on the ACE lat/lon grid."""
    static_names = [name for name in forcing_names if name in STATIC_ACE_FORCING]
    if not static_names:
        return torch.empty((0, len(lat), len(lon)), dtype=torch.float32)

    ds = ACE2ERA5Data(mode="forcing", verbose=False)
    forcing_e2s_names = [ACELexicon.get_e2s_from_fme(name) for name in static_names]
    x, _coords = fetch_data(
        source=ds,
        time=np.array([time], dtype="datetime64[ns]"),
        variable=np.array(forcing_e2s_names, dtype=object),
        lead_time=np.array([np.timedelta64(0, "h")]),
        device="cpu",
    )
    return x[0, 0].to(torch.float32).cpu()


def build_ace_forcing_tensor(
    *,
    time: np.datetime64,
    lat: np.ndarray,
    lon: np.ndarray,
    forcing_names: Sequence[str],
    static_forcing: torch.Tensor,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Assemble ACE forcing channels from valid time plus static fields."""
    lat_2d, lon_2d = np.meshgrid(lat, lon, indexing="ij")
    static_names = [name for name in forcing_names if name in STATIC_ACE_FORCING]
    channels: list[torch.Tensor] = []
    for name in forcing_names:
        if name in AUTONOMOUS_ACE_FORCING:
            dswrf = calculate_dswrftoa(time, lat_2d, lon_2d, correct_eot=True).astype(
                np.float32
            )
            dswrf_t = torch.from_numpy(dswrf).to(device=device, dtype=dtype)
            channels.append(dswrf_t[None, None].expand(batch_size, 1, -1, -1))
        elif name in STATIC_ACE_FORCING:
            idx = static_names.index(name)
            channels.append(
                static_forcing[idx : idx + 1]
                .unsqueeze(0)
                .expand(batch_size, 1, -1, -1)
                .to(device=device, dtype=dtype)
            )
        else:
            raise ValueError(
                f"Unsupported ACE forcing variable {name!r}; expected one of "
                f"{sorted(AUTONOMOUS_ACE_FORCING | STATIC_ACE_FORCING)}."
            )
    return torch.cat(channels, dim=1)
