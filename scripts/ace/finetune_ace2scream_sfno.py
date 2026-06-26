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
"""Fine-tune an ACE SFNO trunk to predict ACE-grid forecast residuals.

The current forecast-residual workflow consumes the preprocessed ACE-grid
netCDF produced by ``scripts/ace/build_ace_forecast_pairs.py`` and trains the
ACE trunk to predict the residual ``truth_valid - forecast_valid``.
"""

from __future__ import annotations

import argparse

import torch

from screamcast.ace import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        required=True,
        help="ACE-grid training netCDF from scripts/ace/build_ace_forecast_pairs.py",
    )
    parser.add_argument(
        "--ace-checkpoint",
        default="",
        help="Optional ACE checkpoint tar. Defaults to ACE2ERA5.load_default_package().",
    )
    parser.add_argument("--output", default="ace2scream_sfno.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--train-backbone", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
