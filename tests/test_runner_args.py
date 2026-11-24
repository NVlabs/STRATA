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
"""Defaults that ``run_screamcast_nudged.parse_args`` resolves from other options, and
the start-up of ``--ic-onthefly``."""

import importlib.util
import json
import os
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

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


# ------------------------------------------------------------------ --ic-onthefly


def _spec(tmp_path, **spec):
    path = tmp_path / "spec.json"
    statics = {"map": "m", "phis": "p", "vert": "v", "latlon": "l", "scream_ref": "r"}
    path.write_text(json.dumps({**statics, **spec}))
    return str(path)


def test_ic_onthefly_starts_at_the_forecast_init(runner, tmp_path):
    spec = _spec(tmp_path, source="wb2_hres", init="2020-08-26T00")
    args = runner.parse_args(
        ["--ic-onthefly", spec, "--correction", "none", "--driver-target-dir", "t"]
    )
    _, t0 = runner.onthefly_ic_start(args)
    assert t0 == np.datetime64("2020-08-26T00:00:00")
    args.initial_time = "2020-08-26T06"  # lead 6 of the forecast
    assert runner.onthefly_ic_start(args)[1] == np.datetime64("2020-08-26T06")


@pytest.mark.parametrize(
    "extra, source, match",
    [
        (["--ic-store", "ic.zarr"], {}, "not both"),
        (["--correction", "ace"], {}, "--correction none"),
        (["--no-target-dir"], {}, "--driver-target-dir"),
        ([], {"source": "era5"}, "needs --initial-time"),
        (["--initial-time", "2020-08-26T03"], {}, "no state at"),
        ([], {"init": "2023-06-01T00"}, "WeatherBench2 holds"),
    ],
)
def test_ic_onthefly_refuses_what_it_cannot_build(
    runner, tmp_path, extra, source, match
):
    spec = _spec(tmp_path, **{"source": "wb2_hres", "init": "2020-08-26T00", **source})
    argv = ["--ic-onthefly", spec, "--correction", "none"]
    if "--no-target-dir" in extra:
        extra = []
    else:
        argv += ["--driver-target-dir", "t"]
    args = runner.parse_args(argv + extra)
    with pytest.raises(ValueError, match=match):
        runner.onthefly_ic_start(args)


def test_ic_onthefly_failure_on_rank_0_stops_the_other_ranks(
    runner, tmp_path, monkeypatch
):
    args = runner.parse_args(
        [
            "--driver-target-dir",
            str(tmp_path / "targets"),
            "--driver-target-timeout",
            "5",
        ]
    )
    run_dir = str(tmp_path / "targets" / "runs" / "this")
    t0 = np.datetime64("2020-08-26T00:00:00")

    def unreachable(*a, **k):
        raise OSError("WeatherBench2 unreachable")

    monkeypatch.setattr(runner, "build_ic_store", unreachable)
    with pytest.raises(OSError):
        runner.onthefly_ic(args, {}, t0, run_dir, 0, torch.device("cpu"))
    with pytest.raises(RuntimeError, match="WeatherBench2 unreachable"):
        runner.onthefly_ic(args, {}, t0, run_dir, 1, torch.device("cpu"))


def test_ic_onthefly_path_reaches_every_rank(runner, tmp_path, monkeypatch):
    args = runner.parse_args(
        [
            "--driver-target-dir",
            str(tmp_path / "targets"),
            "--driver-target-timeout",
            "5",
        ]
    )
    run_dir = str(tmp_path / "targets" / "runs" / "this")
    t0 = np.datetime64("2020-08-26T00:00:00")
    built = []

    def build(spec, valid, out_dir, **kw):
        built.append((valid, out_dir))
        return os.path.join(out_dir, "ic_2020082600.zarr")

    monkeypatch.setattr(runner, "build_ic_store", build)
    path = runner.onthefly_ic(args, {}, t0, run_dir, 0, torch.device("cpu"))
    assert path == str(tmp_path / "targets" / "ic_2020082600.zarr")
    assert built == [(t0, str(tmp_path / "targets"))]
    assert runner.onthefly_ic(args, {}, t0, run_dir, 1, torch.device("cpu")) == path
    assert len(built) == 1  # only rank 0 builds
