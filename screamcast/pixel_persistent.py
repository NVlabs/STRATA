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
"""Screen pixel NA persistent scheduling; Cirrus-42 (AI agent)."""

import json
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import torch
from natten._libnatten import blackwell_na3d_forward
from physicsnemo.experimental.models.strata import layers

from screamcast import attention_layout, pixel_qkv_pack
from screamcast.cuda_timing import measure


def attention(q, k, v, persistent):
    output, _ = blackwell_na3d_forward(
        q,
        k,
        v,
        (9, 9, 9),
        (1, 1, 1),
        (1, 1, 1),
        (False, False, False),
        0.125,
        (24, 128, 128),
        (24, 128, 128),
        (24, 128, 128),
        (8, 4, 8),
        (4, 4, 8),
        persistent,
    )
    return output


def benchmark(commit, results):
    result = dict(
        agent="Cirrus-42 (AI agent)",
        commit=commit,
        time=datetime.now(timezone.utc).isoformat(),
        experiment="pixel-native-attention-persistent-component",
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        warmup=5,
        repeats=100,
        shape=[8, 393216, 2, 64],
        window=[9, 9, 9],
        scale=0.125,
        query_tile=[8, 4, 8],
        kv_tile=[4, 4, 8],
        scope="Synthetic native-packed BF16 QKV; identical native attention geometry, vary persistent scheduling only; output allocation included",
        cases=[],
    )
    Path(results).write_text(json.dumps(result, indent=2) + "\n")
    with torch.inference_mode():
        for seed in (20261020, 20261021):
            torch.manual_seed(seed)
            q, k, v = [
                torch.randn((8, 393216, 2, 64), device="cuda", dtype=torch.bfloat16)
                for _ in range(3)
            ]
            baseline = partial(attention, q, k, v, False)
            candidate = partial(attention, q, k, v, True)
            expected = baseline()
            repeat = baseline()
            actual = candidate()
            row = dict(
                seed=seed,
                native_repeat_different_bits=int(
                    (expected.view(torch.int16) != repeat.view(torch.int16)).sum()
                ),
                candidate_different_bits=int(
                    (expected.view(torch.int16) != actual.view(torch.int16)).sum()
                ),
                finite=bool(actual.isfinite().all()),
                relative_rms=float(
                    (actual.float() - expected.float()).square().mean().sqrt()
                    / expected.float().square().mean().sqrt()
                ),
            )
            del repeat, actual, expected
            row["baseline_before"] = measure(baseline, 5, 100)
            row["candidate"] = measure(candidate, 5, 100)
            row["baseline_after"] = measure(baseline, 5, 100)
            result["cases"].append(row)
            Path(results).write_text(json.dumps(result, indent=2) + "\n")
            print(
                "PIXEL_PERSISTENT",
                seed,
                row["candidate_different_bits"],
                {
                    k: row[k]["median_ms"]
                    for k in ("baseline_before", "candidate", "baseline_after")
                },
                flush=True,
            )
            del baseline, candidate, q, k, v


_NATIVE_CORE = attention_layout.attention_core


@torch.library.custom_op("cirrus42::pixel_persistent_attention", mutates_args=())
def persistent_core(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    for tensor in (q, k, v):
        if (
            tensor.shape != (8, 393216, 2, 64)
            or tensor.dtype != torch.bfloat16
            or not tensor.is_cuda
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                "Persistent pixel attention requires released native packed CUDA BF16 tensors"
            )
    return attention(q, k, v, True)


@persistent_core.register_fake
def persistent_core_fake(q, k, v):
    return torch.empty(q.shape, device=q.device, dtype=q.dtype)


def restore():
    attention_layout.attention_core = _NATIVE_CORE


def install(strata):
    selected = {
        id(module)
        for module in strata.pixel_blocks.modules()
        if isinstance(module, layers.Natten3DSelfAttention)
    }
    installed = {id(module) for module, _, _ in pixel_qkv_pack._ORIGINALS}
    if (
        strata.training
        or attention_layout._LAYOUT != (24, 128, 128)
        or attention_layout._Q_TILE != (8, 4, 8)
        or attention_layout._KV_TILE != (4, 4, 8)
        or len(selected) != 4
        or selected != installed
    ):
        raise ValueError(
            "Persistent pixel attention requires four validated packed pixel modules"
        )
    attention_layout.attention_core = persistent_core
