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
"""Exhaustively check the BF16 gate boundary against native compilation."""

import json
from pathlib import Path

import torch

from screamcast.semantic_gate_precision import rounded_sigmoid


def run(args):
    with torch.inference_mode():
        words = torch.arange(-32768, 32768, dtype=torch.int32, device="cuda").to(
            torch.int16
        )
        x = words.view(torch.bfloat16)
        reference = torch.compile(torch.sigmoid)(x)
        actual = rounded_sigmoid(x)
        mask = ~x.isnan()
        different = (
            (actual.view(torch.int16).ne(reference.view(torch.int16)) & mask)
            .sum()
            .item()
        )
        error = (actual[mask].float() - reference[mask].float()).abs().max().item()
        row = dict(
            agent="Cirrus-42 (AI agent)",
            commit=args.commit,
            experiment="semantic-gate-exhaustive-bf16",
            gpu=torch.cuda.get_device_name(),
            torch=torch.__version__,
            non_nan_bf16_inputs=mask.sum().item(),
            different_output_bit_patterns=different,
            max_absolute_difference=error,
            decision="pass" if different == 0 else "fail",
        )
        Path(args.results).write_text(json.dumps(row, indent=2) + "\n")
        print("RESULT", json.dumps(row), flush=True)
