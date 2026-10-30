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
"""Compare LUT with in-place native GELU, excluding resets; Cirrus-42 (AI agent)."""

import argparse
import json
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import torch
from torch.nn import functional as F

from screamcast.cuda_timing import measure
from screamcast.gelu_table import evaluate


def native_inplace(input):
    input.copy_(F.gelu(input, approximate="tanh"))
    return input


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--geometry", choices=("backbone", "pixel"), default="backbone")
    args = parser.parse_args()
    torch.manual_seed(20261020)
    oracle = torch.compile(partial(F.gelu, approximate="tanh"))
    native = torch.compile(native_inplace)
    with torch.inference_mode():
        patterns = (
            torch.arange(65536, device="cuda", dtype=torch.int32)
            .to(torch.int16)
            .view(torch.bfloat16)
        )
        table = oracle(patterns)
        shape = (8, 24576, 4096) if args.geometry == "backbone" else (8, 393216, 512)
        source = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        work = source.clone()
        reset = partial(work.copy_, source)
        expected = oracle(source)
        native(work)
        native_bits = int((work.view(torch.int16) != expected.view(torch.int16)).sum())
        actual = evaluate(source, table, block=4096)
        candidate_bits = int(
            (actual.view(torch.int16) != expected.view(torch.int16)).sum()
        )
        del actual, expected
        row = dict(
            agent="Cirrus-42 (AI agent)",
            time=datetime.now(timezone.utc).isoformat(),
            commit=args.commit,
            experiment="bf16-gelu-table-inplace-control",
            gpu=torch.cuda.get_device_name(),
            torch=torch.__version__,
            input_shape=list(source.shape),
            geometry=args.geometry,
            native_different_bits=native_bits,
            candidate_different_bits=candidate_bits,
            warmup=5,
            repeats=50,
            scope="Native compiled in-place GELU versus out-of-place lookup; identical random BF16 source restored before every call; source copy and synchronization excluded from CUDA and wall samples",
        )
        row["baseline_before"] = measure(
            partial(native, work), 5, 50, before_each=reset
        )
        row["candidate"] = measure(
            partial(evaluate, work, table, block=4096), 5, 50, before_each=reset
        )
        row["baseline_after"] = measure(partial(native, work), 5, 50, before_each=reset)
        Path(args.results).write_text(json.dumps(row, indent=2) + "\n")
        print(
            "COMPLETE",
            native_bits,
            candidate_bits,
            {
                k: row[k]["median_ms"]
                for k in ("baseline_before", "candidate", "baseline_after")
            },
            flush=True,
        )


if __name__ == "__main__":
    main()
