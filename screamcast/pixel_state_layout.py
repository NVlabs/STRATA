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
"""Screen intermediate pixel-state layouts; Cirrus-42 (AI agent)."""

from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, SliceLayout
from triton.experimental.gluon.language.extra import libdevice

from screamcast import pixel_norm2_mod, pixel_norm_mod
from screamcast.benchmark_statistics import latency_gate
from screamcast.cuda_timing import measure
from screamcast.gemm_schedule_validation import save
from screamcast.normalization_order import _combine


@gluon.jit
def norm1_row_kernel(X, SHIFT, SCALE, OUT, ROWS: gl.constexpr):
    COMPUTE: gl.constexpr = BlockedLayout([4, 1], [2, 16], [1, 8], [0, 1])
    STORE: gl.constexpr = BlockedLayout([1, 4], [1, 32], [8, 1], [1, 0])
    row = gl.program_id(0).to(gl.int64) * ROWS + gl.arange(
        0, ROWS, layout=SliceLayout(1, STORE)
    )
    channel = gl.arange(0, 128, layout=SliceLayout(0, STORE))
    loaded = gl.convert_layout(
        gl.load(
            X + row[:, None] * 128 + channel[None, :], eviction_policy="evict_last"
        ),
        COMPUTE,
    )
    average, moment, count = gl.reduce(
        (loaded, gl.zeros_like(loaded), gl.full_like(loaded, 1.0)), 1, _combine
    )
    scale = libdevice.rsqrt(moment / 128.0 + 1e-6)
    normalized = (loaded - average[:, None]) * scale[:, None]
    normalized = gl.convert_layout(normalized, STORE)
    shift = gl.load(SHIFT + row[:, None] * 768 + channel[None, :]).to(gl.float32)
    modulation = gl.load(SCALE + row[:, None] * 768 + channel[None, :]).to(gl.float32)
    value = normalized * (modulation + 1.0) + shift
    gl.store(OUT + row[:, None] * 128 + channel[None, :], value)


@gluon.jit
def norm2_row_kernel(X, ATTN, GATE, SHIFT, SCALE, OUT):
    COMPUTE: gl.constexpr = BlockedLayout([1, 4], [1, 32], [2, 1], [1, 0])
    row = gl.program_id(0).to(gl.int64) * 8 + gl.arange(
        0, 8, layout=SliceLayout(1, COMPUTE)
    )
    channel = gl.arange(0, 128, layout=SliceLayout(0, COMPUTE))
    offset = row[:, None] * 128 + channel[None, :]
    mod_offset = row[:, None] * 768 + channel[None, :]
    original = gl.load(X + offset, eviction_policy="evict_last")
    gate = gl.load(GATE + mod_offset).to(gl.float32)
    attn = gl.load(ATTN + offset).to(gl.float32)
    residual = original + gate * attn
    average, moment, count = gl.reduce(
        (residual, gl.zeros_like(residual), gl.full_like(residual, 1.0)), 1, _combine
    )
    shift = gl.load(SHIFT + mod_offset).to(gl.float32)
    scale = gl.load(SCALE + mod_offset).to(gl.float32)
    normalized = (residual - average[:, None]) * libdevice.rsqrt(
        moment[:, None] / 128.0 + 1e-6
    )
    value = normalized * (scale + 1.0) + shift
    gl.store(OUT + offset, value)


def norm1_row(x, raw, rows):
    if not x.is_contiguous():
        raise ValueError("Intermediate norm1 requires contiguous BNC state")
    out = torch.empty(x.shape, device=x.device, dtype=torch.bfloat16)
    norm1_row_kernel[(3145728 // rows,)](
        x,
        raw[:, :, :128],
        raw[:, :, 128:256],
        out,
        ROWS=rows,
        num_warps=8,
        num_stages=1,
        enable_fp_fusion=True,
    )
    return out


def norm2_row(x, raw, attn):
    if not x.is_contiguous():
        raise ValueError("Intermediate norm2 requires contiguous BNC state")
    out = torch.empty(x.shape, device=x.device, dtype=torch.bfloat16)
    norm2_row_kernel[(3145728 // 8,)](
        x,
        attn,
        raw[:, :, 256:384],
        raw[:, :, 384:512],
        raw[:, :, 512:640],
        out,
        num_warps=2,
        num_stages=1,
        enable_fp_fusion=True,
    )
    return out


def channel_update(x, attn, gate1, gate2, mlp):
    result = (x + gate1 * attn) + gate2 * mlp
    return result.transpose(1, 2).contiguous().transpose(1, 2)


def row_update(x, attn, gate1, gate2, mlp):
    return ((x + gate1 * attn) + gate2 * mlp).contiguous()


def pipeline(x, blocks, channel, row, *, middle_row, norm1_rows):
    outputs = []
    for index, (raw, attn, mlp) in enumerate(blocks):
        if middle_row and index > 0:
            norm1 = norm1_row(x, raw, norm1_rows)
            norm2 = norm2_row(x, raw, attn)
        else:
            norm1 = pixel_norm_mod.evaluate(x, raw, 32, True)
            norm2 = pixel_norm2_mod.evaluate(x, raw, attn, 8, True, 2)
        update = row if middle_row and index < 3 else channel
        x = update(x, attn, raw[:, :, 256:384], raw[:, :, 640:768], mlp)
        outputs.extend((norm1, norm2, x))
    return tuple(outputs)


def trace(call, path):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        call()
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(path))


def run(args):
    result = dict(
        agent="Cirrus-42 (AI agent)",
        source_commit=args.commit,
        time=datetime.now(timezone.utc).isoformat(),
        experiment="four-stage-pixel-state-intermediate-row-layout",
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        warmup=5,
        repeats=100,
        seeds=[20261058, 20261059],
        cases=[],
        scope="Four normalization/update stages at B8x393216x128: synthetic FP32 initial channel-major state, four BF16 modulation/attention/MLP-output sets. Native confirmed norm1 rows32/eightwarps, norm2 rows8/twowarps and compiled channel-major gated update versus contiguous BNC intermediate state, original channel-major entry/final output. Candidate norm1 tiles8/32 retain original cross-warp Welford layout/order; norm2 keeps native compute layout. No learned attention/MLP projections in this boundary component. Includes all four producer stores; no external state transpose. Require every BF16 norm and FP32 state bit identical, prescribed state strides and four adjusted timing intervals; no model acceptance.",
    )
    save(args.results, result)
    with torch.inference_mode():
        channel = torch.compile(channel_update)
        row = torch.compile(row_update)
        for rows in (8, 32):
            case = dict(norm1_rows=rows, samples=[])
            result["cases"].append(case)
            print("STATE_LAYOUT_START", rows, flush=True)
            for seed in result["seeds"]:
                torch.manual_seed(seed)
                x = torch.randn(
                    (8, 128, 393216), device="cuda", dtype=torch.float32
                ).transpose(1, 2)
                blocks = [
                    (
                        torch.randn(
                            (8, 393216, 768), device="cuda", dtype=torch.bfloat16
                        ),
                        torch.randn(
                            (8, 393216, 128), device="cuda", dtype=torch.bfloat16
                        ),
                        torch.randn(
                            (8, 393216, 128), device="cuda", dtype=torch.bfloat16
                        ),
                    )
                    for _ in range(4)
                ]
                before = partial(
                    pipeline, x, blocks, channel, row, middle_row=False, norm1_rows=32
                )
                candidate = partial(
                    pipeline, x, blocks, channel, row, middle_row=True, norm1_rows=rows
                )
                expected, actual = before(), candidate()
                errors = []
                for index, (aa, bb) in enumerate(zip(actual, expected)):
                    bits = torch.int32 if aa.dtype == torch.float32 else torch.int16
                    desired = (
                        [50331648, 128, 1]
                        if index % 3 != 2 or index < 9
                        else [50331648, 1, 393216]
                    )
                    errors.append(
                        dict(
                            index=index,
                            role=("norm1", "norm2", "state")[index % 3],
                            finite=bool(aa.isfinite().all()),
                            different_bits=int((aa.view(bits) != bb.view(bits)).sum()),
                            actual_stride=list(aa.stride()),
                            expected_stride=list(bb.stride()),
                            prescribed_stride=desired,
                        )
                    )
                sample = dict(seed=seed, outputs=errors)
                del expected, actual
                if seed == result["seeds"][0]:
                    prefix = Path(args.results).with_suffix("")
                    trace(before, str(prefix) + f"-rows{rows}-baseline.json")
                    trace(candidate, str(prefix) + f"-rows{rows}-candidate.json")
                sample["before"] = measure(before, 5, 100)
                sample["candidate"] = measure(candidate, 5, 100)
                sample["after"] = measure(before, 5, 100)
                case["samples"].append(sample)
                save(args.results, result)
                print(
                    "STATE_LAYOUT_RESULT",
                    rows,
                    seed,
                    sum(e["different_bits"] for e in errors),
                    {
                        k: sample[k]["median_ms"]
                        for k in ("before", "candidate", "after")
                    },
                    flush=True,
                )
                del before, candidate, x, blocks
            gate = latency_gate(*case["samples"], isolated_scope="seed20261058")
            gate["comparisons"] = {
                k.replace("full_", "seed20261059_"): v
                for k, v in gate["comparisons"].items()
            }
            case.update(
                gate=gate,
                component_eligible=gate["significant"]
                and all(
                    e["finite"]
                    and e["different_bits"] == 0
                    and e["actual_stride"] == e["prescribed_stride"]
                    for s in case["samples"]
                    for e in s["outputs"]
                ),
                status="complete",
            )
            save(args.results, result)
    result["status"] = "complete"
    save(args.results, result)
