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

import contextlib
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


def save_checkpoint_atomic(
    fabric: Fabric, path: str, state: dict, best: str | None = None
) -> None:
    """``fabric.save(path, state)``, written to ``<path>.tmp``, verified and renamed onto ``path`` by rank 0.

    With ``best`` (rank 0's value counts), rank 0 also points ``best`` at the
    new file, before renaming it onto ``path``. A kill between the two then
    leaves ``best`` on the new state and ``path`` on the previous one, and
    ``choose_resume_checkpoint`` keeps ``best``'s lower loss when resuming.
    The other order would leave a ``path`` recording a loss that no
    ``best`` holds, and ``best`` would stay on the older state.

    Collective: every rank must call it, and every rank raises if any rank's
    part failed. When rank 0's part fails, ``path`` and ``best`` are left as
    they were; when only another rank's part fails, rank 0 has already
    replaced them.
    """
    tmp_path = path + ".tmp"
    error = None
    try:
        if fabric.global_rank == 0 and os.path.lexists(tmp_path):
            # Left by a run killed after its fabric.save. fabric.save would copy
            # over it, and fsspec ignores a failed copy onto an existing file,
            # so this stale file could pass the check below and be promoted.
            os.unlink(tmp_path)
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
            if best is not None:
                link_best(tmp_path, best)
            os.replace(tmp_path, path)
            _fsync_dir(os.path.dirname(os.path.abspath(path)))
        except Exception as exc:  # noqa: BLE001 - shared with every rank below
            error = _describe(exc)
    if error is not None and fabric.global_rank == 0:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
    error = _first_error(fabric, error)
    if error is not None:
        raise RuntimeError(f"checkpoint save to {path} failed: {error}")


def link_best(src: str, best: str) -> None:
    """Point ``best`` at the file ``src`` names now.

    A hard link costs no copy and stays the best state because checkpoints
    are only ever replaced by rename, never rewritten in place. Linked to a
    temp name and renamed, so ``best`` never disappears in between.
    """
    tmp_path = best + ".tmp"
    if os.path.lexists(tmp_path):
        os.unlink(tmp_path)
    os.link(src, tmp_path)
    os.replace(tmp_path, best)


def recorded_best_valid_loss(path: str) -> float | None:
    """``best_valid_loss`` stored in the checkpoint at ``path``, or None when it is missing, unusable or has none.

    Reads the whole file once for the CRC check (a few seconds per GB), then
    memory-maps it to read the value without loading the weights.
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
    from ``experiment.resume_from``). ``notes`` describe every rejected file,
    and warn when ``latest.pth`` is missing but ``steps_*.pth`` exist;
    ``error`` is set when ``latest.pth`` is unusable and no ``steps_*.pth``
    can replace it.
    """
    latest = os.path.join(rundir, "latest.pth")
    steps = []
    if os.path.isdir(rundir):
        for name in os.listdir(rundir):
            match = _STEPS_CKPT.match(name)
            if match is not None:
                steps.append((int(match.group(1)), os.path.join(rundir, name)))
    if not os.path.exists(latest):
        if steps:
            # Usually a corrupt latest.pth deleted by hand; resuming from a
            # steps_*.pth then needs experiment.resume_from.
            return (
                None,
                [
                    f"{latest} does not exist but {len(steps)} steps_*.pth do; "
                    "starting from experiment.resume_from or from scratch, "
                    "not resuming this run"
                ],
                None,
            )
        return None, [], None
    problem = checkpoint_problem(latest)
    if problem is None:
        return latest, [], None
    notes = [f"{latest} is unusable ({problem})"]
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


_STALE_TMP = re.compile(r"^(latest|best|steps_\d+)\.pth\.tmp$")


def remove_stale_tmp_files(rundir: str) -> list[str]:
    """Delete the ``*.pth.tmp`` files a killed run left in ``rundir``; return a note for each.

    Call only before this run's first save, when every such file is a
    leftover. A ``steps_<N>.pth.tmp`` would otherwise stay forever, because
    no later save uses the same N.
    """
    if not os.path.isdir(rundir):
        return []
    notes = []
    for name in sorted(os.listdir(rundir)):
        if _STALE_TMP.match(name) is not None:
            path = os.path.join(rundir, name)
            size_gb = os.path.getsize(path) / 1e9
            os.unlink(path)
            notes.append(
                f"removed {path} ({size_gb:.2f} GB), left by an interrupted save"
            )
    return notes


def choose_resume_checkpoint(
    fabric: Fabric, rundir: str
) -> tuple[str | None, float | None]:
    """The checkpoint this run resumes from, and the ``best_valid_loss`` floor to apply after loading it.

    Rank 0 removes leftover ``*.pth.tmp`` files, runs
    ``select_resume_checkpoint`` and, when this run is resuming, reads the
    ``best_valid_loss`` that ``best.pth`` records; every rank gets its answer.
    ``best.pth`` can record a lower loss than the resumed file: a
    ``steps_*.pth`` fallback predates later validations, and a kill inside
    ``save_checkpoint_atomic`` can leave ``latest.pth`` one save behind
    ``best.pth``. Applying the floor (``apply_best_valid_loss_floor``) keeps
    ``best.pth`` from being replaced by a worse state.

    Collective: every rank must call it. Every rank raises when rank 0 found
    no usable checkpoint to resume from, or failed while looking.
    """
    choice = None
    if fabric.global_rank == 0:
        try:
            notes = remove_stale_tmp_files(rundir)
            path, select_notes, error = select_resume_checkpoint(rundir)
            notes += select_notes
            floor = None
            if path is not None:
                floor = recorded_best_valid_loss(os.path.join(rundir, "best.pth"))
            choice = (path, notes, error, floor)
        except Exception as exc:  # noqa: BLE001 - raised on every rank below
            choice = (
                None,
                [],
                f"choosing the checkpoint to resume from failed: {_describe(exc)}",
                None,
            )
    path, notes, error, floor = fabric.broadcast(choice, src=0)
    for note in notes:
        fabric.print(f"WARNING: {note}")
    if error is not None:
        raise RuntimeError(error)
    if path is not None and os.path.basename(path) != "latest.pth":
        fabric.print(f"WARNING: resuming from {path} instead of latest.pth")
    return path, floor


def apply_best_valid_loss_floor(
    fabric: Fabric, state: dict, floor: float | None
) -> None:
    """Lower ``state["best_valid_loss"]`` to ``floor`` when ``floor`` is lower; never raise it."""
    if floor is None or floor >= state["best_valid_loss"]:
        return
    fabric.print(
        f"best_valid_loss {state['best_valid_loss']:.6g} -> {floor:.6g}, "
        "recorded in best.pth"
    )
    state["best_valid_loss"] = floor
