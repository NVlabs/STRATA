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
"""Screen fixed semantic backbone attention tiles; Cirrus-42 (AI agent)."""

import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from natten._libnatten import blackwell_na3d_forward
from natten.backends.configs.cutlass_blackwell import BLACKWELL_FORWARD_TILE_SHAPES
from natten.token_permute.torch_impl import token_permute_torch, token_unpermute_torch

from screamcast.cuda_timing import measure


@torch.library.custom_op("cirrus42::semantic_tile_core", mutates_args=())
def core(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_tile: list[int],
    kv_tile: list[int],
    persistent: bool,
) -> torch.Tensor:
    output, _ = blackwell_na3d_forward(
        q,
        k,
        v,
        (9, 9, 9),
        (1, 1, 1),
        (1, 1, 1),
        (False, False, False),
        0.125,
        (24, 32, 32),
        (24, 32, 32),
        (24, 32, 32),
        tuple(q_tile),
        tuple(kv_tile),
        persistent,
    )
    return output


@core.register_fake
def core_fake(q, k, v, q_tile, kv_tile, persistent):
    return torch.empty_like(q)


def attention(q, k, v, q_tile, kv_tile, persistent):
    q, k, v = [
        t.reshape(8, 16, 24, 32, 32, 64).permute(0, 2, 3, 4, 1, 5) for t in (q, k, v)
    ]
    q = token_permute_torch(q, q_tile, (1, 1, 1), True)
    k = token_permute_torch(k, kv_tile, (1, 1, 1), True)
    v = token_permute_torch(v, kv_tile, (1, 1, 1), True)
    output = core(q, k, v, list(q_tile), list(kv_tile), persistent)
    return token_unpermute_torch(output, (24, 32, 32), q_tile, (1, 1, 1), True).reshape(
        8, 24576, 1024
    )


def run(args):
    torch.manual_seed(20261020)
    operands = [
        torch.randn(8, 16, 24576, 64, device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    ]
    record = dict(
        agent="Cirrus-42 (AI agent)",
        time=datetime.now(timezone.utc).isoformat(),
        commit=args.commit,
        experiment="semantic-attention-tile-screen",
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        seed=20261020,
        layout_path="Traceable Torch token permutations around native Blackwell kernel; no backward-only LSE permutation",
        scope="B8,D24,H=W32,16 heads64,BF16 inference; synthetic heads-first QKV; includes required input/output layout; default compile, graph breaks allowed; excludes projections, norm, RoPE",
        trials=[],
    )
    default_q, default_kv = (8, 4, 8), (4, 4, 8)
    with torch.inference_mode():
        reference = attention(*operands, default_q, default_kv, False)
        configs = [(default_q, default_kv, False)]
        configs += [
            (q, kv, persistent)
            for q, kv in BLACKWELL_FORWARD_TILE_SHAPES[3]
            for persistent in (False, True)
        ]
        configs += [(default_q, default_kv, False)]
        for q_tile, kv_tile, persistent in configs:
            torch._dynamo.reset()
            call = torch.compile(
                lambda: attention(*operands, q_tile, kv_tile, persistent)
            )
            print("START", q_tile, kv_tile, persistent, flush=True)
            row = dict(q_tile=q_tile, kv_tile=kv_tile, persistent=persistent)
            try:
                actual = call()
                delta = actual.float() - reference.float()
                row.update(
                    finite=bool(actual.isfinite().all()),
                    relative_rms=(
                        delta.square().mean().sqrt()
                        / reference.float().square().mean().sqrt()
                    ).item(),
                    max_abs=delta.abs().max().item(),
                )
                del actual, delta
                row["timing"] = measure(call, args.warmup, args.repeats)
                print(
                    "RESULT",
                    q_tile,
                    kv_tile,
                    persistent,
                    row["timing"]["median_ms"],
                    row["relative_rms"],
                    flush=True,
                )
            except (RuntimeError, NotImplementedError) as error:
                row["failure"] = str(error)
                print("FAILED", str(error), flush=True)
            record["trials"].append(row)
            Path(args.results).write_text(json.dumps(record, indent=2) + "\n")
