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
"""Defaults that ``run_screamcast_nudged.parse_args`` resolves from other options."""

import importlib.util
import os
from pathlib import Path
from unittest import mock

import pytest

WINDOW = "3,8,12,7,11"
BOX = "0,20,-90,-70"


@pytest.fixture(scope="module")
def runner():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "ace"
        / "run_screamcast_nudged.py"
    )
    spec = importlib.util.spec_from_file_location("run_screamcast_nudged", path)
    module = importlib.util.module_from_spec(spec)
    # the script loads a .env at import; keep it out of the other tests' environment
    with mock.patch.dict(os.environ):
        spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "argv, expected",
    [
        ([], True),
        (["--region", WINDOW], False),
        (["--region", WINDOW, "--regional-averages"], True),
        (["--no-regional-averages"], False),
        (["--region-latlon", BOX], False),
    ],
    ids=["global", "window", "window-explicit-on", "global-explicit-off", "box"],
)
def test_regional_averages_default(runner, argv, expected):
    """On for global runs, off for a --region window unless asked for."""
    assert runner.parse_args(argv).regional_averages is expected


def test_region_and_region_latlon_exclude_each_other(runner):
    with pytest.raises(SystemExit):
        runner.parse_args(["--region", WINDOW, "--region-latlon", BOX])
