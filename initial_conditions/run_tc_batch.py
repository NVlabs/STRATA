#!/usr/bin/env python
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
# ruff: noqa: S603 S607
"""Build, convert, and upload the tropical-cyclone HICCUP ICs listed in layout.CASES.

For each case: run_hiccup_eamxx -> hiccup_l128_to_zarr (validates and
consolidates) -> rclone copy to ``<remote>/<tag>/``, rclone check, then the
manifest. The manifest goes up last, so cases whose manifest is already on the
remote are skipped. The local NetCDF and HICCUP temp files are deleted after a
checked upload.

Run from the repository root::

    python -m initial_conditions.run_tc_batch --work-root ... --remote REMOTE:PREFIX
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import shlex
import shutil
import subprocess
import sys

from initial_conditions.layout import CASES, nc_name, zarr_name
from screamcast.history import git_commit

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def log(msg: str) -> None:
    print(msg, flush=True)


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    log("$ " + shlex.join(cmd))
    subprocess.check_call(cmd, env=env, cwd=REPO)


def remote_done(remote: str, tag: str) -> bool:
    """True when ``<remote>/<tag>/manifest.txt`` exists; it is uploaded only after the store."""
    try:
        listing = subprocess.check_output(
            ["rclone", "lsf", f"{remote}/{tag}/"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError:
        return False
    return "manifest.txt" in listing.split()


def sha256(path: str) -> str:
    out = subprocess.check_output(["sha256sum", path], text=True)
    return out.split()[0]


def nco_version() -> str:
    try:
        out = subprocess.run(
            ["ncremap", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return str(shutil.which("ncremap"))
    return (out.stdout or out.stderr).strip().splitlines()[0]


def write_manifest(
    args, case: dict, out_dir: str, nc_path: str, zarr_path: str, cmds: dict
) -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(args.map + ".command.txt") as handle:
        map_cmd = handle.read().strip()
    lines = [
        f"timestamp: {case['timestamp']}",
        f"tag: {case['tag']}",
        f"storm: {case['storm']}",
        f"map_command: {map_cmd}",
        f"hiccup_command: {cmds['hiccup']}",
        f"zarr_command: {cmds['zarr']}",
        f"git_sha: {git_commit()}",
        f"nco: {nco_version()}",
        f"hiccup_root: {args.hiccup_root}",
        f"map_path: {args.map}",
        f"dst_scrip: {args.dst_scrip}",
        f"latlon: {args.latlon}",
        f"scream_ref: {args.scream_ref}",
        f"nc_sha256: {sha256(nc_path) if os.path.isfile(nc_path) else 'deleted'}",
        f"zarr_json_sha256: {sha256(os.path.join(zarr_path, 'zarr.json'))}",
        f"written: {stamp}",
        "",
    ]
    path = os.path.join(out_dir, "manifest.txt")
    with open(path, "w") as handle:
        handle.write("\n".join(lines))
    return path


def process_case(args, case: dict) -> None:
    tag = case["tag"]
    timestamp = case["timestamp"]
    work = os.path.join(args.work_root, "cases", tag)
    tmp = os.path.join(work, "tmp")
    out_dir = os.path.join(work, "out")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(tmp, exist_ok=True)
    nc_path = os.path.join(out_dir, nc_name(timestamp))
    z_path = os.path.join(out_dir, zarr_name(tag))
    env = os.environ.copy()
    env["PATH"] = os.path.dirname(args.python) + os.pathsep + env.get("PATH", "")
    env["TMPDIR"] = tmp
    hiccup_cmd = [
        args.python,
        "-m",
        "initial_conditions.run_hiccup_eamxx",
        f"--timestamp={timestamp}",
        f"--map={args.map}",
        f"--dst-scrip={args.dst_scrip}",
        f"--out={nc_path}",
        f"--data-root={args.work_root}",
        f"--hiccup-root={args.hiccup_root}",
        f"--tmp-dir={tmp}",
    ]
    zarr_cmd = [
        args.python,
        "-m",
        "initial_conditions.hiccup_l128_to_zarr",
        f"--ic={nc_path}",
        f"--out={z_path}",
        f"--timestamp={timestamp}",
        f"--latlon={args.latlon}",
        f"--scream-ref={args.scream_ref}",
        f"--workers={args.workers}",
    ]
    if os.path.isfile(os.path.join(z_path, "zarr.json")):
        log(f"{tag}: local zarr exists, skipping convert")
    else:
        if not (os.path.isfile(nc_path) and os.path.getsize(nc_path) > 10_000_000_000):
            shutil.rmtree(tmp)
            os.makedirs(tmp)
            run(hiccup_cmd, env=env)
        run(zarr_cmd, env=env)
    cmds = {"hiccup": shlex.join(hiccup_cmd), "zarr": shlex.join(zarr_cmd)}
    manifest = write_manifest(args, case, out_dir, nc_path, z_path, cmds)
    dest = f"{args.remote}/{tag}/"
    run(["rclone", "copy", z_path, dest + os.path.basename(z_path)])
    # every local file must be on the remote with the same size before the
    # manifest (the completion marker) goes up and the local NetCDF is deleted
    run(["rclone", "check", "--one-way", z_path, dest + os.path.basename(z_path)])
    run(["rclone", "copyto", manifest, dest + "manifest.txt"])
    if not remote_done(args.remote, tag):
        raise SystemExit(f"upload missing manifest for {tag}")
    if os.path.isfile(nc_path):
        os.remove(nc_path)
        log(f"{tag}: deleted local NC after Zarr validated and uploaded")
    shutil.rmtree(tmp)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work-root", required=True, help="per-case work dirs and inputs")
    p.add_argument("--hiccup-root", required=True, help="HICCUP clone")
    p.add_argument("--map", required=True, help="ERA5 721x1440 -> ne1024pg2 map")
    p.add_argument("--dst-scrip", required=True, help="ne1024pg2 SCRIP grid file")
    p.add_argument("--latlon", required=True, help="NetCDF with lat/lon on ncol")
    p.add_argument(
        "--scream-ref", required=True, help="dir with sgh30.zarr and landfrac.zarr"
    )
    p.add_argument("--remote", required=True, help="rclone destination prefix")
    p.add_argument(
        "--python",
        default=sys.executable,
        help="interpreter of the HICCUP/NCO environment",
    )
    p.add_argument("--tags", nargs="*", help="subset of CASES tags to run")
    p.add_argument("--workers", type=int, default=16)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cases = CASES
    if args.tags:
        unknown = set(args.tags) - {case["tag"] for case in CASES}
        if unknown:
            raise SystemExit(f"unknown tags {sorted(unknown)}")
        cases = [case for case in CASES if case["tag"] in args.tags]
    done = []
    for case in cases:
        tag = case["tag"]
        if remote_done(args.remote, tag):
            log(f"{tag}: already on {args.remote}, skipping")
        else:
            log(f"==== {tag} {case['timestamp']} ====")
            process_case(args, case)
        done.append(tag)
        log(f"{tag}: done ({len(done)}/{len(cases)})")
    log("all cases: " + " ".join(done))


if __name__ == "__main__":
    main()
