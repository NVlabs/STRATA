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
"""Screen all BF16 inputs and production GELU geometry; Cirrus-42 (AI agent)."""

import argparse
import json
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import torch
from torch.nn import functional as F

from screamcast.cuda_timing import measure
from screamcast.gelu_table import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    torch.manual_seed(20261020)
    native = torch.compile(partial(F.gelu, approximate="tanh"))
    with torch.inference_mode():
        patterns = (
            torch.arange(65536, device="cuda", dtype=torch.int32)
            .to(torch.int16)
            .view(torch.bfloat16)
        )
        table = native(patterns)
        trials = []
        for mode in (0, 1):
            actual = evaluate(patterns, table, mode=mode)
            finite = torch.isfinite(patterns)
            trials.append(
                dict(
                    mode=mode,
                    exhaustive_different_bits=int(
                        (actual.view(torch.int16) != table.view(torch.int16)).sum()
                    ),
                    exhaustive_finite_different_bits=int(
                        (
                            (actual.view(torch.int16) != table.view(torch.int16))
                            & finite
                        ).sum()
                    ),
                )
            )
        probe = patterns.repeat(12288).reshape(8, 24576, 4096)
        probe_expected = native(probe)
        actual = evaluate(probe, table)
        production_shape_exhaustive_bits = int(
            (actual.view(torch.int16) != probe_expected.view(torch.int16)).sum()
        )
        del actual, patterns, probe, probe_expected
        input = torch.randn((8, 24576, 4096), device="cuda", dtype=torch.bfloat16)
        expected = native(input)
        row = dict(
            agent="Cirrus-42 (AI agent)",
            time=datetime.now(timezone.utc).isoformat(),
            commit=args.commit,
            experiment="bf16-gelu-table-screen",
            gpu=torch.cuda.get_device_name(),
            torch=torch.__version__,
            input_shape=list(input.shape),
            approximate="tanh",
            exhaustive=trials,
            production_shape_exhaustive_different_bits=production_shape_exhaustive_bits,
            trials=[],
            repeats=args.repeats,
            scope="Synthetic BF16 activation; exhaustive bit-pattern comparison and production backbone MLP shape; full network pending",
        )
        row["baseline_before"] = measure(partial(native, input), 5, args.repeats)
        for mode in (0, 1):
            for block, warps in ((512, 4), (1024, 4), (2048, 4), (4096, 4), (2048, 8)):
                actual = evaluate(input, table, mode=mode, block=block, warps=warps)
                item = dict(
                    mode=mode,
                    block=block,
                    warps=warps,
                    different_bits=int(
                        (actual.view(torch.int16) != expected.view(torch.int16)).sum()
                    ),
                )
                del actual
                item["timing"] = measure(
                    partial(
                        evaluate, input, table, mode=mode, block=block, warps=warps
                    ),
                    5,
                    args.repeats,
                )
                row["trials"].append(item)
                print(
                    "GELU",
                    mode,
                    block,
                    warps,
                    item["different_bits"],
                    item["timing"]["median_ms"],
                    flush=True,
                )
        row["baseline_after"] = measure(partial(native, input), 5, args.repeats)
        Path(args.results).write_text(json.dumps(row, indent=2) + "\n")
        print(
            "COMPLETE",
            row["baseline_before"]["median_ms"],
            row["baseline_after"]["median_ms"],
            trials,
            flush=True,
        )


if __name__ == "__main__":
    main()
