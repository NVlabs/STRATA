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
"""Crash-safe checkpoint files for train.py.

``fabric.save(path)`` goes through lightning's ``_atomic_save``, which writes a
temp file in the system temp dir and then ``shutil.move``s it onto ``path``.
When that temp dir is on another filesystem than the run dir (node-local /tmp
vs. a Lustre run dir on every cluster), ``shutil.move`` cannot rename and
COPIES INTO the existing ``path`` instead, truncating it in place. Two bugs
follow:

* ``best.pth`` is a hard link to ``latest.pth``, so every later in-place save
  of ``latest.pth`` also overwrites ``best.pth``: it tracks the latest state
  instead of the one with the lowest validation loss.
* A walltime kill during that copy leaves a truncated ``latest.pth`` that the
  next segment cannot load.

Here every checkpoint is first saved to ``<path>.tmp`` in the run dir,
verified, and then ``os.replace``d onto ``path``: a rename within one
directory, so ``path`` is always either the complete old file or the complete
new one, and never rewritten in place.

The collective helpers make every rank raise when one rank fails, instead of
leaving the others waiting in a barrier.
"""

from __future__ import annotations

import os
import re
import zipfile

import torch
import torch.distributed as dist
from lightning.fabric import Fabric

_STEPS_CKPT = re.compile(r"^steps_(\d+)\.pth$")


def checkpoint_problem(path: str) -> str | None:
    """Why the checkpoint zip at ``path`` is unusable, or None when every member passes its CRC check."""
    try:
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()
    except Exception as exc:  # noqa: BLE001 - any failure to read it makes it unusable
        return f"{type(exc).__name__}: {exc}"
    if bad is not None:
        return f"member {bad!r} fails its CRC check"
    return None


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _first_error(fabric: Fabric, error: str | None) -> str | None:
    """The first non-None ``error`` over all ranks; every rank gets the same answer."""
    if fabric.world_size == 1:
        return error
    errors: list[str | None] = [None] * fabric.world_size
    dist.all_gather_object(errors, error)
    return next((e for e in errors if e is not None), None)


def _fsync_dir(directory: str) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save_checkpoint_atomic(fabric: Fabric, path: str, state: dict) -> None:
    """``fabric.save(path, state)``, written to ``<path>.tmp``, verified and renamed onto ``path`` by rank 0.

    Collective: every rank must call it, and every rank raises if any rank's
    part failed (``path`` is then left as it was).
    """
    tmp_path = path + ".tmp"
    error = None
    try:
        fabric.save(tmp_path, state)
    except Exception as exc:  # noqa: BLE001 - shared with every rank below
        error = f"fabric.save: {_describe(exc)}"
        # fabric.save ends in a barrier; a rank that raised before it never
        # joined, and the other ranks are waiting there. Join it now.
        fabric.barrier()
    if error is None and fabric.global_rank == 0:
        try:
            with open(tmp_path, "rb") as f:
                os.fsync(f.fileno())
            problem = checkpoint_problem(tmp_path)
            if problem is not None:
                raise RuntimeError(f"{tmp_path}: {problem}; leaving {path} untouched")
            os.replace(tmp_path, path)
            _fsync_dir(os.path.dirname(os.path.abspath(path)))
        except Exception as exc:  # noqa: BLE001 - shared with every rank below
            error = _describe(exc)
    error = _first_error(fabric, error)
    if error is not None:
        raise RuntimeError(f"checkpoint save to {path} failed: {error}")


def link_best(latest: str, best: str) -> None:
    """Point ``best`` at the file ``latest`` names now.

    A hard link costs no copy and stays the best state because ``latest`` is
    only ever replaced by rename, never rewritten in place. Linked to a temp
    name and renamed, so ``best`` never disappears in between.
    """
    tmp_path = best + ".tmp"
    if os.path.lexists(tmp_path):
        os.unlink(tmp_path)
    os.link(latest, tmp_path)
    os.replace(tmp_path, best)


def link_best_collective(fabric: Fabric, link: bool, latest: str, best: str) -> None:
    """Rank 0 runs ``link_best`` when ``link`` is true there; every rank raises if it failed.

    Collective: call it on every rank after every save, whatever ``link`` is
    on that rank, so the ranks never disagree about entering it.
    """
    error = None
    if link and fabric.global_rank == 0:
        try:
            link_best(latest, best)
        except Exception as exc:  # noqa: BLE001 - shared with every rank below
            error = _describe(exc)
    error = fabric.broadcast(error, src=0)
    if error is not None:
        raise RuntimeError(f"linking {best} to {latest} failed on rank 0: {error}")


def recorded_best_valid_loss(path: str) -> float | None:
    """``best_valid_loss`` stored in the checkpoint at ``path``, or None when it is missing, unusable or has none.

    Memory-maps the file, so the weights are not read.
    """
    if not os.path.exists(path) or checkpoint_problem(path) is not None:
        return None
    ckpt = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    value = ckpt.get("best_valid_loss")
    return None if value is None else float(value)


def select_resume_checkpoint(rundir: str) -> tuple[str | None, list[str], str | None]:
    """Pick the checkpoint an interrupted run resumes from.

    Returns ``(path, notes, error)``. ``path`` is ``latest.pth`` when it passes
    the CRC check, else the newest ``steps_<N>.pth`` that does; None with no
    error when ``latest.pth`` does not exist (a fresh run, or one starting
    from ``experiment.resume_from``). ``notes`` describe every rejected file;
    ``error`` is set when ``latest.pth`` is unusable and no ``steps_*.pth``
    can replace it.
    """
    latest = os.path.join(rundir, "latest.pth")
    if not os.path.exists(latest):
        return None, [], None
    problem = checkpoint_problem(latest)
    if problem is None:
        return latest, [], None
    notes = [f"{latest} is unusable ({problem})"]
    steps = []
    for name in os.listdir(rundir):
        match = _STEPS_CKPT.match(name)
        if match is not None:
            steps.append((int(match.group(1)), os.path.join(rundir, name)))
    for _, path in sorted(steps, reverse=True):
        problem = checkpoint_problem(path)
        if problem is None:
            return path, notes, None
        notes.append(f"{path} is unusable ({problem})")
    error = (
        f"{latest} cannot be loaded and no usable steps_*.pth exists in {rundir} "
        "(steps_*.pth are written only with training.save_ckpt_every_epoch=True). "
        "Tried: " + "; ".join(notes)
    )
    return None, notes, error
