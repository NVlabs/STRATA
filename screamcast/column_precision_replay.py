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
"""Replay verified real column operands without model loading; Cirrus-42 (AI agent)."""

import hashlib
import json
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import torch

from screamcast.column_precision import error_metrics, native_attention, variant
from screamcast.cuda_timing import measure

MODE_LABELS = {
    0: "exp_shift",
    1: "exp2_div_full",
    2: "exp2_fma",
    3: "libdevice_exp",
    4: "exp2_reciprocal_rn",
    5: "exp2_rcp_approx",
    6: "exp2_div_rn",
    7: "exp2_sum_f64",
    8: "exp2_fragment_sequential_xor",
    9: "exp2_fragment_sequential_adjacent",
    10: "exp2_fragment_pairs_xor",
    11: "exp2_fragment_column_order_xor",
    12: "exp2_inverse_log_sum",
    13: "exp2_lse_normalization",
    14: "exp2_zero_max",
}


def load_operands(entry, index):
    path = Path(entry["path"])
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != entry["sha256"]:
            raise ValueError(f"Operand checksum differs: {path}")
    data = torch.load(path, map_location="cuda", weights_only=True)
    if (
        data["column_block"] != index
        or data["repeats"] != 8
        or data["full_shape"] != [8192, 16, 24, 64]
    ):
        raise ValueError("Operand corpus metadata differs from fixed B8 geometry")
    operands = []
    for name in ("q", "k", "v", "expected"):
        tensor = data[name]
        if (
            tensor.shape != (1024, 16, 24, 64)
            or tensor.dtype != torch.bfloat16
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                "Operand corpus requires contiguous BF16 first-batch arrays"
            )
        operands.append(tensor.repeat(8, 1, 1, 1))
    return operands


def run(args):
    manifest = json.loads(Path(args.manifest).read_text())
    if manifest["canonical_backbone_error"]["max_absolute_difference"] != 0:
        raise ValueError("Operand corpus did not preserve canonical backbone output")
    entries = manifest["operand_files"]
    if len(entries) != 12:
        raise ValueError("Operand corpus requires twelve complete column blocks")
    row = dict(
        agent="Cirrus-42 (AI agent)",
        time=datetime.now(timezone.utc).isoformat(),
        commit=args.commit,
        experiment="column-real-operand-replay",
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        corpus_manifest=args.manifest,
        corpus_source_commit=manifest["commit"],
        input_sha256=manifest["input_sha256"],
        modes=args.modes,
        warmup=args.warmup,
        repeats=args.repeats,
        expected_column_blocks=12,
        completed_column_blocks=0,
        scope="Twelve real column blocks; verify file hashes; reconstruct B8 from eight verified identical batches; component CUDA timing excludes data loading and compilation",
        column_blocks=[],
    )
    with torch.inference_mode():
        for index, entry in enumerate(entries):
            q, k, v, expected = load_operands(entry, index)
            native = native_attention(q, k, v)
            different = int(
                (native.view(torch.int16) != expected.view(torch.int16)).sum()
            )
            block = dict(
                column_block=index, native_replay_different_bits=different, trials=[]
            )
            row["column_blocks"].append(block)
            if different:
                Path(args.results).write_text(json.dumps(row, indent=2) + "\n")
                raise RuntimeError(
                    f"Native replay differs in {different} output elements"
                )
            del native
            block["baseline_before"] = measure(
                partial(native_attention, q, k, v), args.warmup, args.repeats
            )
            for mode in args.modes:
                actual = variant(q, k, v, mode)
                item = dict(
                    mode=mode,
                    label=MODE_LABELS[mode],
                    **error_metrics(actual, expected),
                )
                del actual
                item["timing"] = measure(
                    partial(variant, q, k, v, mode), args.warmup, args.repeats
                )
                block["trials"].append(item)
                print(
                    "REPLAY",
                    index,
                    mode,
                    item["relative_rms"],
                    item["different_output_bits"],
                    item["timing"]["median_ms"],
                    flush=True,
                )
            block["baseline_after"] = measure(
                partial(native_attention, q, k, v), args.warmup, args.repeats
            )
            row["completed_column_blocks"] = index + 1
            Path(args.results).write_text(json.dumps(row, indent=2) + "\n")
            del q, k, v, expected
    print("COMPLETE", row["completed_column_blocks"], flush=True)
