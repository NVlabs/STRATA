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
"""Crash-safe checkpoint writes (screamcast/checkpoint_io.py).

The cluster failure mode is emulated directly: lightning's fabric.save ends in
``shutil.move(temp, path)``, and when the temp dir is on another filesystem
than the run dir, ``shutil.move`` copies INTO the existing ``path``. Each
scenario also runs against the plain ``fabric.save`` that train.py used
before, as a strict xfail, to show the test detects the bug.
"""

from __future__ import annotations

import multiprocessing
import os
import shutil
import socket
import zipfile
import zlib

import pytest
import torch
from lightning.fabric import Fabric

from screamcast.checkpoint_io import (
    checkpoint_problem,
    link_best,
    link_best_collective,
    recorded_best_valid_loss,
    save_checkpoint_atomic,
    select_resume_checkpoint,
)

_real_copyfile = shutil.copyfile


def _plain_fabric_save(fabric, path, state):
    fabric.save(path, state)


SAVERS = [
    pytest.param(save_checkpoint_atomic, id="atomic"),
    pytest.param(
        _plain_fabric_save,
        id="plain-fabric-save",
        marks=pytest.mark.xfail(strict=True, reason="the bug this module fixes"),
    ),
]


@pytest.fixture
def fabric():
    f = Fabric(accelerator="cpu", devices=1)
    return f


@pytest.fixture
def cross_filesystem_move(monkeypatch):
    """shutil.move as it behaves across filesystems: copy into the existing destination."""

    def move(src, dst):
        _real_copyfile(src, dst)
        os.unlink(src)
        return dst

    monkeypatch.setattr(shutil, "move", move)


def _state(steps: int) -> dict:
    return {"steps": steps, "weights": torch.full((4096,), float(steps))}


@pytest.mark.parametrize("save", SAVERS)
def test_best_keeps_the_best_state_after_later_saves(
    tmp_path, monkeypatch, fabric, cross_filesystem_move, save
):
    monkeypatch.chdir(tmp_path)
    save(fabric, "latest.pth", _state(100))  # validation with the lowest loss
    link_best("latest.pth", "best.pth")
    save(fabric, "latest.pth", _state(200))  # a later, worse validation or backup

    assert torch.load("latest.pth", weights_only=False)["steps"] == 200
    assert torch.load("best.pth", weights_only=False)["steps"] == 100


class _Killed(BaseException):
    """Stands in for the walltime kill (like SystemExit, not an Exception)."""


@pytest.mark.parametrize("save", SAVERS)
def test_interrupted_save_leaves_the_previous_checkpoint_loadable(
    tmp_path, monkeypatch, fabric, save
):
    monkeypatch.chdir(tmp_path)
    save(fabric, "latest.pth", _state(100))

    def killed_move(src, dst):  # copies half of the new checkpoint, then dies
        with open(src, "rb") as f:
            data = f.read()
        with open(dst, "wb") as f:
            f.write(data[: len(data) // 2])
        raise _Killed

    monkeypatch.setattr(shutil, "move", killed_move)
    with pytest.raises(_Killed):
        save(fabric, "latest.pth", _state(200))

    assert checkpoint_problem("latest.pth") is None
    assert torch.load("latest.pth", weights_only=False)["steps"] == 100


def test_link_best_replaces_an_existing_best(tmp_path):
    latest, best = tmp_path / "latest.pth", tmp_path / "best.pth"
    torch.save(_state(1), best)
    torch.save(_state(2), latest)

    link_best(str(latest), str(best))

    assert os.path.samefile(latest, best)
    assert not (tmp_path / "best.pth.tmp").exists()


def _truncated(path):
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])


def test_checkpoint_problem_flags_truncated_and_missing_files(tmp_path):
    good = tmp_path / "good.pth"
    torch.save(_state(1), good)
    bad = tmp_path / "bad.pth"
    torch.save(_state(1), bad)
    _truncated(bad)

    assert checkpoint_problem(str(good)) is None
    assert checkpoint_problem(str(bad)) is not None
    assert checkpoint_problem(str(tmp_path / "missing.pth")) is not None


def test_resume_uses_latest_when_it_is_valid(tmp_path):
    torch.save(_state(300), tmp_path / "latest.pth")
    torch.save(_state(200), tmp_path / "steps_200.pth")

    path, notes, error = select_resume_checkpoint(str(tmp_path))

    assert (path, notes, error) == (str(tmp_path / "latest.pth"), [], None)


def test_resume_falls_back_to_the_newest_valid_steps_checkpoint(tmp_path):
    for steps in (100, 200, 1000):
        torch.save(_state(steps), tmp_path / f"steps_{steps}.pth")
    torch.save(_state(1100), tmp_path / "latest.pth")
    _truncated(tmp_path / "latest.pth")
    _truncated(tmp_path / "steps_1000.pth")  # newest by step number, also broken

    path, notes, error = select_resume_checkpoint(str(tmp_path))

    assert path == str(tmp_path / "steps_200.pth")  # numeric order, not "1000" < "200"
    assert error is None
    assert len(notes) == 2


def test_resume_without_a_usable_fallback_is_an_error(tmp_path):
    torch.save(_state(1100), tmp_path / "latest.pth")
    _truncated(tmp_path / "latest.pth")

    path, notes, error = select_resume_checkpoint(str(tmp_path))

    assert path is None
    assert "save_ckpt_every_epoch" in error
    assert len(notes) == 1


def test_fresh_run_has_nothing_to_resume(tmp_path):
    assert select_resume_checkpoint(str(tmp_path / "not_created_yet")) == (
        None,
        [],
        None,
    )


def test_checkpoint_problem_reports_unexpected_read_errors(tmp_path, monkeypatch):
    path = tmp_path / "latest.pth"
    torch.save(_state(1), path)

    def corrupt_stream(self):
        raise zlib.error("invalid stored block lengths")

    monkeypatch.setattr(zipfile.ZipFile, "testzip", corrupt_stream)

    assert "invalid stored block lengths" in checkpoint_problem(str(path))


def test_failed_fabric_save_raises_and_keeps_the_previous_checkpoint(
    tmp_path, monkeypatch, fabric
):
    monkeypatch.chdir(tmp_path)
    save_checkpoint_atomic(fabric, "latest.pth", _state(100))

    def disk_full(path, state):
        raise OSError("No space left on device")

    monkeypatch.setattr(fabric, "save", disk_full)
    with pytest.raises(RuntimeError, match="No space left on device"):
        save_checkpoint_atomic(fabric, "latest.pth", _state(200))

    assert torch.load("latest.pth", weights_only=False)["steps"] == 100


def test_failed_link_raises_and_keeps_the_previous_best(tmp_path, monkeypatch, fabric):
    latest, best = tmp_path / "latest.pth", tmp_path / "best.pth"
    torch.save(_state(1), best)
    torch.save(_state(2), latest)

    def no_hard_links(src, dst):
        raise PermissionError("hard links not permitted")

    monkeypatch.setattr(os, "link", no_hard_links)
    with pytest.raises(RuntimeError, match="hard links not permitted"):
        link_best_collective(fabric, True, str(latest), str(best))

    assert torch.load(best, weights_only=False)["steps"] == 1


def test_link_best_collective_without_a_new_best_leaves_best_alone(tmp_path, fabric):
    latest, best = tmp_path / "latest.pth", tmp_path / "best.pth"
    torch.save(_state(2), latest)

    link_best_collective(fabric, False, str(latest), str(best))

    assert not best.exists()


def test_recorded_best_valid_loss(tmp_path):
    with_loss = tmp_path / "with_loss.pth"
    torch.save({**_state(1), "best_valid_loss": 0.25}, with_loss)
    without_loss = tmp_path / "without_loss.pth"
    torch.save(_state(1), without_loss)
    truncated = tmp_path / "truncated.pth"
    torch.save({**_state(1), "best_valid_loss": 0.25}, truncated)
    _truncated(truncated)

    assert recorded_best_valid_loss(str(with_loss)) == 0.25
    assert recorded_best_valid_loss(str(without_loss)) is None
    assert recorded_best_valid_loss(str(truncated)) is None
    assert recorded_best_valid_loss(str(tmp_path / "missing.pth")) is None


# Two CPU ranks (gloo): a failure on rank 0 must make BOTH ranks raise rather
# than leave rank 1 waiting forever in a barrier or collective. Both scenarios
# share one spawn: each rank pays ~40 s importing lightning.

_SCENARIOS = ("save", "link")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _two_rank_worker(rank: int, port: int, rundir: str) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        LOCAL_RANK=str(rank),
        NODE_RANK="0",
    )
    fabric = Fabric(accelerator="cpu", devices=2, strategy="ddp")
    fabric.launch()
    for scenario in _SCENARIOS:
        try:
            if scenario == "save":
                # rank 0 writes; its target directory is a regular file
                path = os.path.join(rundir, "not_a_dir", "latest.pth")
                save_checkpoint_atomic(fabric, path, _state(1))
            else:
                # only rank 0 thinks there is a new best; latest.pth does not exist
                link_best_collective(
                    fabric,
                    rank == 0,
                    os.path.join(rundir, "latest.pth"),
                    os.path.join(rundir, "best.pth"),
                )
            outcome = "no error"
        except RuntimeError as exc:
            outcome = f"RuntimeError: {exc}"
        with open(os.path.join(rundir, f"rank{rank}_{scenario}.txt"), "w") as f:
            f.write(outcome)


def test_rank0_failure_raises_on_every_rank(tmp_path):
    (tmp_path / "not_a_dir").write_text("")
    ctx = multiprocessing.get_context("spawn")
    port = _free_port()
    procs = [
        ctx.Process(target=_two_rank_worker, args=(rank, port, str(tmp_path)))
        for rank in range(2)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=180)
    hung = [p for p in procs if p.is_alive()]
    for p in hung:
        p.kill()
    assert not hung, "a rank is stuck waiting for the other"

    for scenario in _SCENARIOS:
        for rank in range(2):
            outcome = (tmp_path / f"rank{rank}_{scenario}.txt").read_text()
            assert outcome.startswith("RuntimeError: "), (scenario, rank, outcome)
