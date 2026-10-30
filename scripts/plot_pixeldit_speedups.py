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
"""Render the historical PixelDiT accepted-change summary from a CSV table."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.data.open() as stream:
        rows = list(csv.DictReader(stream))
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
    x = np.arange(len(rows))
    for ax, field, color, title in zip(
        axes,
        ("full_model_ms", "decoder_ms"),
        ("#0072B2", "#009E73"),
        ("Full network", "Pixel decoder"),
    ):
        values = np.array([float(row[field]) for row in rows])
        measured = [(i, v) for i, v in enumerate(values) if math.isfinite(v)]
        reduction = 100 * (1 - measured[-1][1] / measured[0][1])
        speedup = measured[0][1] / measured[-1][1]
        ax.step(x, values, where="post", color=color, linewidth=2)
        ax.scatter(*zip(*measured), color=color, s=20, zorder=3)
        for i, value in measured:
            ax.annotate(
                f"{value:.1f}",
                (i, value),
                xytext=(0, 10 + 12 * (i % 2)),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        ax.set_title(
            f"{title}: {measured[0][1]:.2f} → {measured[-1][1]:.2f} ms"
            f"  ·  {reduction:.1f}% lower latency  ·  {speedup:.2f}× speedup",
            loc="left",
            fontweight="bold",
        )
        ax.set_ylabel("Median latency (ms)")
        ax.grid(axis="y", alpha=0.2)
        ax.axvspan(4.65, 5.35, color="#000000", alpha=0.05)
        ax.set_ylim(min(v for _, v in measured) * 0.9, measured[0][1] * 1.17)
    axes[1].text(
        11, 163, "Decoder not separately timed in these backbone trials", ha="center"
    )
    axes[1].set_xticks(
        x,
        [row["change"].replace("†", "") for row in rows],
        rotation=48,
        ha="right",
        fontsize=8,
    )
    fig.suptitle("PixelDiT optimization progress", x=0.07, ha="left", fontsize=20)
    fig.text(
        0.07, 0.93, "GB300 · Batch 8 · Tile 128 · BF16 inference · October 19–20, 2026"
    )
    fig.text(
        0.07,
        0.025,
        "Accepted composition snapshots; independent confirmations used where available."
        " Separate runs have different neighboring controls.\n"
        "* Compiler recalibration is shown separately, not attributed to a kernel."
        " Gaps mark unmeasured decoder stages.\n"
        "Warmed forward only: compilation, loading and I/O excluded."
        " Historical measurements; no new GPU benchmarks. Cirrus-126 (AI agent).",
        fontsize=9,
        linespacing=1.5,
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.33, hspace=0.46)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)


if __name__ == "__main__":
    main()
