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
    apply_best_valid_loss_floor,
    checkpoint_problem,
    choose_resume_checkpoint,
    link_best,
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
        marks=pytest.mark.xfail(
            strict=True, raises=AssertionError, reason="the bug this module fixes"
        ),
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


def _state(steps: int, best_valid_loss: float = 100.0) -> dict:
    return {
        "steps": steps,
        "best_valid_loss": best_valid_loss,
        "weights": torch.full((4096,), float(steps)),
    }


def _steps(path) -> int:
    return torch.load(path, weights_only=False)["steps"]


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
    (tmp_path / "best.pth.tmp").write_bytes(b"left by a killed link")

    link_best(str(latest), str(best))

    assert os.path.samefile(latest, best)
    assert not (tmp_path / "best.pth.tmp").exists()


def test_save_with_best_links_both_names_to_the_new_file(tmp_path, monkeypatch, fabric):
    monkeypatch.chdir(tmp_path)
    save_checkpoint_atomic(fabric, "latest.pth", _state(100), best="best.pth")
    save_checkpoint_atomic(fabric, "latest.pth", _state(200))

    assert _steps("best.pth") == 100
    assert _steps("latest.pth") == 200
    save_checkpoint_atomic(fabric, "latest.pth", _state(300), best="best.pth")
    assert os.path.samefile("latest.pth", "best.pth")
    assert sorted(os.listdir(".")) == ["best.pth", "latest.pth"]


def test_kill_between_link_and_rename_keeps_the_new_best_and_its_loss(
    tmp_path, monkeypatch, fabric
):
    monkeypatch.chdir(tmp_path)
    save_checkpoint_atomic(fabric, "latest.pth", _state(100, 0.5), best="best.pth")

    real_replace = os.replace

    def killed_before_latest(src, dst):
        if dst == "latest.pth":
            raise _Killed
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", killed_before_latest)
    with pytest.raises(_Killed):
        save_checkpoint_atomic(fabric, "latest.pth", _state(200, 0.3), best="best.pth")
    monkeypatch.setattr(os, "replace", real_replace)

    assert _steps("best.pth") == 200
    assert _steps("latest.pth") == 100
    # Resuming from the older latest.pth must not forget best.pth's lower loss,
    # or a worse state would replace best.pth at the next validation.
    path, floor = choose_resume_checkpoint(fabric, str(tmp_path))
    assert path == str(tmp_path / "latest.pth")
    state = {"best_valid_loss": 0.5}
    apply_best_valid_loss_floor(fabric, state, floor)
    assert state["best_valid_loss"] == 0.3


def test_stale_complete_tmp_is_not_promoted_when_the_new_save_fails(
    tmp_path, monkeypatch, fabric
):
    """fsspec ignores a PermissionError while copying onto an existing file.

    A complete latest.pth.tmp left by a kill would then pass the CRC check
    and replace a newer latest.pth.
    """
    rundir = tmp_path / "run"
    rundir.mkdir()
    monkeypatch.chdir(rundir)
    save_checkpoint_atomic(fabric, "latest.pth", _state(10))
    torch.save(_state(5), "latest.pth.tmp")  # left by a kill before os.replace

    def denied(src, dst, *args, **kwargs):
        raise PermissionError(13, "Permission denied", dst)

    real_rename = os.rename

    def cross_device(src, dst):
        if not os.path.abspath(src).startswith(str(rundir)):
            raise OSError(18, "Invalid cross-device link")
        return real_rename(src, dst)

    monkeypatch.setattr(shutil, "copyfile", denied)
    monkeypatch.setattr(os, "rename", cross_device)
    with pytest.raises(RuntimeError, match="checkpoint save to latest.pth failed"):
        save_checkpoint_atomic(fabric, "latest.pth", _state(20))

    assert _steps(rundir / "latest.pth") == 10
    assert not (rundir / "latest.pth.tmp").exists()


def test_corrupt_save_is_not_renamed_onto_the_checkpoint(tmp_path, monkeypatch, fabric):
    monkeypatch.chdir(tmp_path)
    save_checkpoint_atomic(fabric, "latest.pth", _state(100), best="best.pth")

    def corrupt_save(path, state):
        torch.save(state, path)
        _truncated(tmp_path / path)

    monkeypatch.setattr(fabric, "save", corrupt_save)
    with pytest.raises(RuntimeError, match="leaving latest.pth untouched"):
        save_checkpoint_atomic(fabric, "latest.pth", _state(200), best="best.pth")

    assert _steps("latest.pth") == 100
    assert _steps("best.pth") == 100
    assert not (tmp_path / "latest.pth.tmp").exists()


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

    assert path == str(tmp_path / "steps_200.pth")
    assert error is None
    assert len(notes) == 2


def test_resume_fallback_takes_the_highest_valid_step_numerically(tmp_path):
    for steps in (90, 200, 1000):
        torch.save(_state(steps), tmp_path / f"steps_{steps}.pth")
    _truncated(tmp_path / "steps_90.pth")  # a lexical sort would rank "90" first
    torch.save(_state(1100), tmp_path / "latest.pth")
    _truncated(tmp_path / "latest.pth")

    path, notes, error = select_resume_checkpoint(str(tmp_path))

    assert path == str(tmp_path / "steps_1000.pth")
    assert error is None
    assert len(notes) == 1


def test_missing_latest_with_steps_checkpoints_warns(tmp_path):
    torch.save(_state(200), tmp_path / "steps_200.pth")

    path, notes, error = select_resume_checkpoint(str(tmp_path))

    assert path is None and error is None
    assert len(notes) == 1 and "latest.pth does not exist" in notes[0]


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


def test_failed_link_raises_and_keeps_both_previous_checkpoints(
    tmp_path, monkeypatch, fabric
):
    monkeypatch.chdir(tmp_path)
    save_checkpoint_atomic(fabric, "latest.pth", _state(100), best="best.pth")

    def no_hard_links(src, dst):
        raise PermissionError("hard links not permitted")

    monkeypatch.setattr(os, "link", no_hard_links)
    with pytest.raises(RuntimeError, match="hard links not permitted"):
        save_checkpoint_atomic(fabric, "latest.pth", _state(200), best="best.pth")

    assert _steps("best.pth") == 100
    assert _steps("latest.pth") == 100
    assert not (tmp_path / "latest.pth.tmp").exists()


def test_choose_resume_floors_with_best_for_latest_and_fallback(tmp_path, fabric):
    torch.save(_state(300, 0.5), tmp_path / "latest.pth")
    torch.save(_state(200, 0.2), tmp_path / "best.pth")
    torch.save(_state(100, 0.9), tmp_path / "steps_100.pth")

    assert choose_resume_checkpoint(fabric, str(tmp_path)) == (
        str(tmp_path / "latest.pth"),
        0.2,
    )

    _truncated(tmp_path / "latest.pth")
    assert choose_resume_checkpoint(fabric, str(tmp_path)) == (
        str(tmp_path / "steps_100.pth"),
        0.2,
    )


def test_choose_resume_has_no_floor_for_a_fresh_run(tmp_path, fabric):
    torch.save(_state(1, 0.2), tmp_path / "best.pth")  # without latest.pth: not resumed

    assert choose_resume_checkpoint(fabric, str(tmp_path)) == (None, None)


def test_choose_resume_raises_without_a_usable_checkpoint(tmp_path, fabric):
    torch.save(_state(1100), tmp_path / "latest.pth")
    _truncated(tmp_path / "latest.pth")

    with pytest.raises(RuntimeError, match="save_ckpt_every_epoch"):
        choose_resume_checkpoint(fabric, str(tmp_path))


def test_choose_resume_removes_leftover_tmp_files(tmp_path, fabric):
    torch.save(_state(300), tmp_path / "latest.pth")
    for name in ("latest.pth.tmp", "best.pth.tmp", "steps_400.pth.tmp"):
        (tmp_path / name).write_bytes(b"partial")
    (tmp_path / "notes.tmp").write_text("not a checkpoint")

    choose_resume_checkpoint(fabric, str(tmp_path))

    assert sorted(os.listdir(tmp_path)) == ["latest.pth", "notes.tmp"]


@pytest.mark.parametrize(
    "loaded, floor, expected",
    [(0.5, 0.2, 0.2), (0.2, 0.5, 0.2), (0.5, None, 0.5)],
)
def test_best_valid_loss_floor_only_lowers(fabric, loaded, floor, expected):
    state = {"best_valid_loss": loaded}

    apply_best_valid_loss_floor(fabric, state, floor)

    assert state["best_valid_loss"] == expected


def test_recorded_best_valid_loss(tmp_path):
    with_loss = tmp_path / "with_loss.pth"
    torch.save({**_state(1), "best_valid_loss": 0.25}, with_loss)
    without_loss = tmp_path / "without_loss.pth"
    torch.save({"steps": 1}, without_loss)
    truncated = tmp_path / "truncated.pth"
    torch.save({**_state(1), "best_valid_loss": 0.25}, truncated)
    _truncated(truncated)

    assert recorded_best_valid_loss(str(with_loss)) == 0.25
    assert recorded_best_valid_loss(str(without_loss)) is None
    assert recorded_best_valid_loss(str(truncated)) is None
    assert recorded_best_valid_loss(str(tmp_path / "missing.pth")) is None


# Two CPU ranks (gloo): a failure on rank 0 must make BOTH ranks raise rather
# than leave rank 1 waiting forever in a barrier or collective, and a save
# that works must leave the same files as on one rank. All scenarios share one
# spawn: each rank pays ~40 s importing lightning.

_FAILING_SCENARIOS = ("save", "link", "resume")


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
    for scenario in (*_FAILING_SCENARIOS, "success"):
        try:
            if scenario == "save":
                # rank 0 writes; its target directory is a regular file
                path = os.path.join(rundir, "not_a_dir", "latest.pth")
                save_checkpoint_atomic(fabric, path, _state(1))
            elif scenario == "link":
                # only rank 0 asks for best; its target directory is a regular file
                save_checkpoint_atomic(
                    fabric,
                    os.path.join(rundir, "ok", "latest.pth"),
                    _state(1),
                    best=(
                        os.path.join(rundir, "not_a_dir", "best.pth")
                        if rank == 0
                        else None
                    ),
                )
            elif scenario == "resume":
                # latest.pth is broken and nothing can replace it
                choose_resume_checkpoint(fabric, os.path.join(rundir, "broken"))
            else:
                success = os.path.join(rundir, "success")
                save_checkpoint_atomic(
                    fabric,
                    os.path.join(success, "latest.pth"),
                    _state(2),
                    best=os.path.join(success, "best.pth"),
                )
            outcome = "no error"
        except RuntimeError as exc:
            outcome = f"RuntimeError: {exc}"
        with open(os.path.join(rundir, f"rank{rank}_{scenario}.txt"), "w") as f:
            f.write(outcome)


def test_two_ranks_raise_together_and_save_together(tmp_path):
    (tmp_path / "not_a_dir").write_text("")
    for name in ("ok", "broken", "success"):
        (tmp_path / name).mkdir()
    (tmp_path / "broken" / "latest.pth").write_bytes(b"truncated")
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

    for scenario in _FAILING_SCENARIOS:
        for rank in range(2):
            outcome = (tmp_path / f"rank{rank}_{scenario}.txt").read_text()
            assert outcome.startswith("RuntimeError: "), (scenario, rank, outcome)
    for rank in range(2):
        assert (tmp_path / f"rank{rank}_success.txt").read_text() == "no error"
    success = tmp_path / "success"
    assert sorted(os.listdir(success)) == ["best.pth", "latest.pth"]
    assert os.path.samefile(success / "latest.pth", success / "best.pth")
    assert _steps(success / "latest.pth") == 2
