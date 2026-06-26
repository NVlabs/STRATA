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
import torch


def temperature_from_potential_temperature(
    theta: torch.Tensor,
    pressure_pa: torch.Tensor,
    p0_pa: float = 100000.0,
    rd: float = 287.05,
    cp: float = 1004.0,
):
    """Convert potential temperature [K] to temperature [K]."""
    kappa = rd / cp
    return theta * torch.pow(pressure_pa / p0_pa, kappa)


def potential_temperature_from_temperature(
    temperature_k: torch.Tensor,
    pressure_pa: torch.Tensor,
    p0_pa: float = 100000.0,
    rd: float = 287.05,
    cp: float = 1004.0,
):
    """Convert temperature [K] to potential temperature [K]."""
    kappa = rd / cp
    pressure_pa = torch.clamp(pressure_pa, min=1.0)
    return temperature_k * torch.pow(p0_pa / pressure_pa, kappa)
