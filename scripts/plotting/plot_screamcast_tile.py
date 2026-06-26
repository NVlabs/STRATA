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
from __future__ import annotations

import argparse

import matplotlib
import zarr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a single SCREAMcast face/x/y tile from a forecast zarr."
    )
    parser.add_argument("--input", required=True, help="Input forecast zarr path")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--variable", default="qv")
    parser.add_argument("--time-index", type=int, default=0)
    parser.add_argument("--level-index", type=int, default=20)
    parser.add_argument("--face", type=int, default=0)
    parser.add_argument("--x-start", type=int, default=0)
    parser.add_argument("--x-stop", type=int, default=512)
    parser.add_argument("--y-start", type=int, default=0)
    parser.add_argument("--y-stop", type=int, default=512)
    parser.add_argument("--cmap", default="viridis")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    group = zarr.open_group(args.input, mode="r")
    var = group[args.variable]
    tile = var[
        args.time_index,
        args.level_index,
        args.face,
        args.x_start : args.x_stop,
        args.y_start : args.y_stop,
    ]

    level_coord = None
    if "level" in group:
        level_coord = group["level"][args.level_index].item()

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    im = ax.imshow(tile, origin="lower", cmap=args.cmap)
    title = (
        f"{args.variable}, time={args.time_index}, level_index={args.level_index}, "
        f"face={args.face}, x={args.x_start}:{args.x_stop}, y={args.y_start}:{args.y_stop}"
    )
    if level_coord is not None:
        title += f", level_coord={level_coord}"
    ax.set_title(title)
    ax.set_xlabel("y")
    ax.set_ylabel("x")
    fig.colorbar(im, ax=ax, label=args.variable)
    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight")


if __name__ == "__main__":
    main()
