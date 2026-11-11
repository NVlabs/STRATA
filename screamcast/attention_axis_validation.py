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
"""Screen isotropic neighborhood axis layouts; Cirrus-42 (AI agent)."""

from datetime import datetime, timezone
from functools import partial

import torch
from natten._libnatten import blackwell_na3d_forward
from natten.token_permute.torch_impl import token_permute_torch, token_unpermute_torch

from screamcast.benchmark_statistics import latency_gate
from screamcast.cuda_timing import measure
from screamcast.gemm_schedule_validation import difference, save

Q_TILE = (8, 4, 8)
KV_TILE = (4, 4, 8)
AXES = ((1, 0, 2), (2, 1, 0))


def attention(q, k, v, shape):
    output, _ = blackwell_na3d_forward(
        q,
        k,
        v,
        (9, 9, 9),
        (1, 1, 1),
        (1, 1, 1),
        (False, False, False),
        0.125,
        shape,
        shape,
        shape,
        Q_TILE,
        KV_TILE,
        True,
    )
    return output


def pack(inputs, axes):
    order = (0, *(axis + 1 for axis in axes), 4, 5)
    return tuple(
        token_permute_torch(x.permute(order).contiguous(), tile, (1, 1, 1), True)
        for x, tile in zip(inputs, (Q_TILE, KV_TILE, KV_TILE))
    )


def restore(output, shape, axes):
    grid = token_unpermute_torch(output, shape, Q_TILE, (1, 1, 1), True)
    inverse = tuple(axes.index(axis) + 1 for axis in range(3))
    original = grid.permute(0, *inverse, 4, 5).contiguous()
    return original.reshape(-1, original.shape[-2] * 64)


def run(args):
    result = dict(
        agent="Cirrus-42 (AI agent)",
        source_commit=args.commit,
        time=datetime.now(timezone.utc).isoformat(),
        experiment="isotropic-neighborhood-axis-layout",
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        query_tile=list(Q_TILE),
        kv_tile=list(KV_TILE),
        warmup=5,
        repeats=100,
        seeds=[20261048, 20261049],
        axes=[list(a) for a in AXES],
        cases=[],
        scope="Native persistent Blackwell FNA, both actual B8/T128 semantic and pixel shapes; synthetic BF16 QKV. Isotropic nine-cube neighborhood preserves membership under axis permutation. Timing includes native core/output allocation and excludes packing/restoration; output compared after restoring original spatial order. Component evidence only; fusion of new packing and full-model gates required before selection.",
    )
    save(args.results, result)
    with torch.inference_mode():
        for name, shape, heads in (
            ("semantic", (24, 32, 32), 16),
            ("pixel", (24, 128, 128), 2),
        ):
            for axes in AXES:
                physical = tuple(shape[axis] for axis in axes)
                case = dict(
                    name=name,
                    original_shape=list(shape),
                    physical_shape=list(physical),
                    axes=list(axes),
                    heads=heads,
                    samples=[],
                )
                result["cases"].append(case)
                print("ATTENTION_AXIS_START", name, axes, physical, flush=True)
                try:
                    for seed in result["seeds"]:
                        torch.manual_seed(seed)
                        inputs = tuple(
                            torch.randn(
                                (8, *shape, heads, 64),
                                device="cuda",
                                dtype=torch.bfloat16,
                            )
                            for _ in range(3)
                        )
                        native_inputs = pack(inputs, (0, 1, 2))
                        new_inputs = pack(inputs, axes)
                        baseline = partial(attention, *native_inputs, shape)
                        candidate = partial(attention, *new_inputs, physical)
                        expected = restore(baseline(), shape, (0, 1, 2))
                        actual = restore(candidate(), physical, axes)
                        sample = dict(seed=seed, error=difference(actual, expected))
                        del actual, expected
                        sample["before"] = measure(baseline, 5, 100)
                        sample["candidate"] = measure(candidate, 5, 100)
                        sample["after"] = measure(baseline, 5, 100)
                        case["samples"].append(sample)
                        save(args.results, result)
                        print(
                            "ATTENTION_AXIS_RESULT",
                            name,
                            axes,
                            seed,
                            sample["error"],
                            {
                                k: sample[k]["median_ms"]
                                for k in ("before", "candidate", "after")
                            },
                            flush=True,
                        )
                        del baseline, candidate, native_inputs, new_inputs, inputs
                    gate = latency_gate(*case["samples"], isolated_scope="seed20261048")
                    gate["comparisons"] = {
                        key.replace("full_", "seed20261049_"): value
                        for key, value in gate["comparisons"].items()
                    }
                    case["gate"] = gate
                    case["component_eligible"] = gate["significant"] and all(
                        sample["error"]["finite"]
                        and sample["error"]["relative_rms"] <= 0.005
                        and sample["error"]["max_channel_relative_rms"] <= 0.01
                        for sample in case["samples"]
                    )
                    case["status"] = "complete"
                except Exception as error:
                    case.update(status="error", error=repr(error))
                    print("ATTENTION_AXIS_ERROR", name, axes, repr(error), flush=True)
                save(args.results, result)
    result["status"] = "complete"
    save(args.results, result)
