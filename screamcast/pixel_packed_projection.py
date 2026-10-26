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
"""Project packed attention before residual gathers; Cirrus-42 (AI agent)."""

from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon._runtime import GluonASTSource
from triton.experimental.gluon.language import BlockedLayout, SliceLayout
from triton.experimental.gluon.language.extra import libdevice

from screamcast import pixel_axis_attention, pixel_norm2_mod, pixel_state_layout
from screamcast.benchmark_statistics import latency_gate
from screamcast.cuda_timing import measure
from screamcast.gemm_schedule_validation import save
from screamcast.normalization_order import _combine


@gluon.jit
def packed_row(row, AXIS: gl.constexpr):
    batch = row // 393216
    token = row % 393216
    d = token // 16384
    h = token // 128 % 128
    w = token % 128
    if AXIS:
        block = ((batch * 3 + d // 8) * 32 + h // 4) * 16 + w // 8
        inner = d % 8 * 32 + h % 4 * 8 + w % 8
    else:
        block = ((batch * 16 + w // 8) * 32 + h // 4) * 3 + d // 8
        inner = w % 8 * 32 + h % 4 * 8 + d % 8
    return block * 256 + inner


@gluon.jit
def normalize_packed(
    X, ATTN, GATE, SHIFT, SCALE, OUT, AXIS: gl.constexpr, ROW_INPUT: gl.constexpr
):
    COMPUTE: gl.constexpr = BlockedLayout([1, 4], [1, 32], [2, 1], [1, 0])
    if ROW_INPUT:
        LOAD: gl.constexpr = COMPUTE
    else:
        LOAD: gl.constexpr = BlockedLayout([4, 1], [2, 16], [1, 2], [0, 1])
    row = gl.program_id(0).to(gl.int64) * 8 + gl.arange(
        0, 8, layout=SliceLayout(1, LOAD)
    )
    channel = gl.arange(0, 128, layout=SliceLayout(0, LOAD))
    if ROW_INPUT:
        source = row[:, None] * 128 + channel[None, :]
    else:
        source = (
            row[:, None] // 393216 * 50331648
            + row[:, None] % 393216
            + channel[None, :] * 393216
        )
    original = gl.convert_layout(
        gl.load(X + source, eviction_policy="evict_last"), COMPUTE
    )
    cr = gl.program_id(0).to(gl.int64) * 8 + gl.arange(
        0, 8, layout=SliceLayout(1, COMPUTE)
    )
    cc = gl.arange(0, 128, layout=SliceLayout(0, COMPUTE))
    mod = cr[:, None] * 768 + cc[None, :]
    attn = gl.load(ATTN + packed_row(cr, AXIS)[:, None] * 128 + cc[None, :]).to(
        gl.float32
    )
    gate = gl.load(GATE + mod).to(gl.float32)
    residual = original + gate * attn
    average, moment, count = gl.reduce(
        (residual, gl.zeros_like(residual), gl.full_like(residual, 1.0)), 1, _combine
    )
    normalized = (residual - average[:, None]) * libdevice.rsqrt(
        moment[:, None] / 128.0 + 1e-6
    )
    shift = gl.load(SHIFT + mod).to(gl.float32)
    scale = gl.load(SCALE + mod).to(gl.float32)
    gl.store(OUT + cr[:, None] * 128 + cc[None, :], normalized * (scale + 1.0) + shift)


@gluon.jit
def update_packed(
    X,
    ATTN,
    GATE1,
    GATE2,
    MLP,
    OUT,
    AXIS: gl.constexpr,
    ROW_INPUT: gl.constexpr,
    ROW_OUTPUT: gl.constexpr,
    ROWS: gl.constexpr,
):
    COMPUTE: gl.constexpr = BlockedLayout([1, 8], [2, 16], [4, 1], [1, 0])
    COLUMN: gl.constexpr = BlockedLayout([1, 4], [ROWS, 32 // ROWS], [1, 4], [0, 1])
    if ROW_INPUT:
        LOAD: gl.constexpr = COMPUTE
    else:
        LOAD: gl.constexpr = COLUMN
    lr = gl.program_id(0).to(gl.int64) * ROWS + gl.arange(
        0, ROWS, layout=SliceLayout(1, LOAD)
    )
    lc = gl.arange(0, 128, layout=SliceLayout(0, LOAD))
    if ROW_INPUT:
        state_offset = lr[:, None] * 128 + lc[None, :]
    else:
        state_offset = (
            lr[:, None] // 393216 * 50331648
            + lr[:, None] % 393216
            + lc[None, :] * 393216
        )
    original = gl.convert_layout(gl.load(X + state_offset), COMPUTE)
    row = gl.program_id(0).to(gl.int64) * ROWS + gl.arange(
        0, ROWS, layout=SliceLayout(1, COMPUTE)
    )
    channel = gl.arange(0, 128, layout=SliceLayout(0, COMPUTE))
    offset = row[:, None] * 128 + channel[None, :]
    mod = row[:, None] * 768 + channel[None, :]
    attn = gl.load(ATTN + packed_row(row, AXIS)[:, None] * 128 + channel[None, :]).to(
        gl.float32
    )
    gate1 = gl.load(GATE1 + mod).to(gl.float32)
    residual = original + gate1 * attn
    gate2 = gl.load(GATE2 + mod).to(gl.float32)
    mlp = gl.load(MLP + offset).to(gl.float32)
    value = residual + gate2 * mlp
    if ROW_OUTPUT:
        gl.store(OUT + offset, value)
    else:
        value = gl.convert_layout(value, COLUMN)
        sr = gl.program_id(0).to(gl.int64) * ROWS + gl.arange(
            0, ROWS, layout=SliceLayout(1, COLUMN)
        )
        sc = gl.arange(0, 128, layout=SliceLayout(0, COLUMN))
        target = (
            sr[:, None] // 393216 * 50331648
            + sr[:, None] % 393216
            + sc[None, :] * 393216
        )
        gl.store(OUT + target, value)


def norm2(x, projected, raw, axis, row_input):
    out = torch.empty(x.shape, device=x.device, dtype=torch.bfloat16)
    normalize_packed[(3145728 // 8,)](
        x,
        projected,
        raw[:, :, 256:384],
        raw[:, :, 384:512],
        raw[:, :, 512:640],
        out,
        AXIS=axis,
        ROW_INPUT=row_input,
        num_warps=2,
        num_stages=1,
        enable_fp_fusion=True,
    )
    return out


def update(x, projected, raw, mlp, axis, row_input, row_output, rows=16):
    out = torch.empty(
        (8, 393216, 128) if row_output else (8, 128, 393216),
        device=x.device,
        dtype=torch.float32,
    )
    update_packed[(3145728 // rows,)](
        x,
        projected,
        raw[:, :, 256:384],
        raw[:, :, 640:768],
        mlp,
        out,
        AXIS=axis,
        ROW_INPUT=row_input,
        ROW_OUTPUT=row_output,
        ROWS=rows,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=True,
    )
    return out if row_output else out.transpose(1, 2)


def pipeline(
    x,
    blocks,
    projections,
    restore_native,
    restore_axis,
    row_update,
    channel_update,
    packed,
    update_rows=16,
):
    outputs = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for index, ((attention, raw, mlp), projection) in enumerate(
            zip(blocks, projections)
        ):
            axis = index == 3
            row_input = index > 0
            row_output = index < 3
            if packed:
                projected = projection(attention.reshape(8, 393216, 128))
                normalized = norm2(x, projected, raw, axis, row_input)
                x = update(
                    x, projected, raw, mlp, axis, row_input, row_output, update_rows
                )
            else:
                restored = (
                    restore_axis(attention) if axis else restore_native(attention)
                )
                projected = projection(restored)
                normalized = (
                    pixel_state_layout.norm2_row(x, raw, projected)
                    if row_input
                    else pixel_norm2_mod.evaluate(x, raw, projected, 8, True, 2)
                )
                updater = row_update if row_output else channel_update
                x = updater(x, projected, raw[:, :, 256:384], raw[:, :, 640:768], mlp)
            outputs.extend((normalized, x))
    return tuple(outputs)


def run(args):
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    state = checkpoint["network"]
    projections = []
    for index in range(4):
        key = f"_pixel_blocks.{index}.attn.proj."
        module = torch.nn.Linear(128, 128, device="meta").eval()
        module.weight = torch.nn.Parameter(state[key + "weight"].cuda())
        module.bias = torch.nn.Parameter(state[key + "bias"].cuda())
        projections.append(torch.compile(module))
    result = dict(
        agent="Cirrus-42 (AI agent)",
        source_commit=args.commit,
        update_rows=args.update_rows,
        time=datetime.now(timezone.utc).isoformat(),
        experiment="pixel-packed-output-projection-before-residual-gather",
        checkpoint=args.checkpoint,
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        warmup=5,
        repeats=100,
        seeds=[20261060, 20261061],
        samples=[],
        scope="Four real released128x128 FP32 attention output projections under native BF16 AMP, synthetic packed BF16 attention inputs, native-stride BF16 modulation/MLP outputs, channel-major FP32 initial state, BNC intermediate states and channel-major final state. Control restores attention before projection and uses selected norm2/update; candidate projects packed rows and gathers directly inside norm2/update, preserving original Welford compute order. Excludes attention and MLP computation, includes learned projection/bias/casts and producer stores. Require all eight normalization/state outputs bit-identical, finite and prescribed strides, plus four adjusted timing intervals; no model acceptance.",
    )
    save(args.results, result)
    with torch.inference_mode():
        restore_native = torch.compile(pixel_axis_attention.native_restore)
        restore_axis = pixel_axis_attention.restore_output
        row_update = torch.compile(pixel_state_layout.row_update)
        channel_update = torch.compile(pixel_state_layout.channel_update)
        for seed in result["seeds"]:
            torch.manual_seed(seed)
            x = torch.randn(
                (8, 128, 393216), device="cuda", dtype=torch.float32
            ).transpose(1, 2)
            blocks = [
                (
                    torch.randn(
                        (8, 393216, 2, 64), device="cuda", dtype=torch.bfloat16
                    ),
                    torch.randn((8, 393216, 768), device="cuda", dtype=torch.bfloat16),
                    torch.randn((8, 393216, 128), device="cuda", dtype=torch.bfloat16),
                )
                for _ in range(4)
            ]
            common = (
                x,
                blocks,
                projections,
                restore_native,
                restore_axis,
                row_update,
                channel_update,
            )
            before = partial(
                pipeline, *common, packed=False, update_rows=args.update_rows
            )
            candidate = partial(
                pipeline, *common, packed=True, update_rows=args.update_rows
            )
            expected, actual = before(), candidate()
            outputs = []
            for index, (a, b) in enumerate(zip(actual, expected)):
                bit_type = torch.int16 if a.dtype == torch.bfloat16 else torch.int32
                desired = [50331648, 1, 393216] if index == 7 else [50331648, 128, 1]
                outputs.append(
                    dict(
                        index=index,
                        role="state" if index % 2 else "norm2",
                        finite=bool(a.isfinite().all()),
                        different_bits=int(
                            (a.view(bit_type) != b.view(bit_type)).sum()
                        ),
                        actual_stride=list(a.stride()),
                        expected_stride=list(b.stride()),
                        prescribed_stride=desired,
                    )
                )
            sample = dict(seed=seed, outputs=outputs)
            result["samples"].append(sample)
            save(args.results, result)
            if not all(
                o["finite"]
                and o["different_bits"] == 0
                and o["actual_stride"] == o["prescribed_stride"]
                for o in outputs
            ):
                result.update(status="rejected_numerical", component_eligible=False)
                save(args.results, result)
                print("PACKED_PROJECTION_NUMERICAL_REJECT", seed, outputs, flush=True)
                return
            del expected, actual
            if seed == result["seeds"][0]:
                prefix = str(Path(args.results).with_suffix(""))
                pixel_state_layout.trace(before, prefix + "-baseline.json")
                pixel_state_layout.trace(candidate, prefix + "-candidate.json")
            sample["before"] = measure(before, 5, 100)
            sample["candidate"] = measure(candidate, 5, 100)
            sample["after"] = measure(before, 5, 100)
            save(args.results, result)
            print(
                "PACKED_PROJECTION_RESULT",
                seed,
                {k: sample[k]["median_ms"] for k in ("before", "candidate", "after")},
                flush=True,
            )
            del before, candidate, x, blocks, common
    gate = latency_gate(*result["samples"], isolated_scope="seed20261060")
    gate["comparisons"] = {
        k.replace("full_", "seed20261061_"): v for k, v in gate["comparisons"].items()
    }
    result.update(gate=gate, component_eligible=gate["significant"], status="complete")
    save(args.results, result)


def compile_only(args):
    result = dict(
        agent="Cirrus-42 (AI agent)",
        source_commit=args.commit,
        update_rows=args.update_rows,
        time=datetime.now(timezone.utc).isoformat(),
        experiment="packed-projection-gather-offline-sm103",
        pointer_alignment_bytes=16,
        gpu_execution=False,
        cases=[],
    )
    for name, axis, row_input, row_output in (
        ("entry", False, False, True),
        ("middle", False, True, True),
        ("exit", True, True, False),
    ):
        for kind, function, signature, constants, warps in (
            (
                "norm2",
                normalize_packed,
                {
                    "X": "*fp32",
                    "ATTN": "*bf16",
                    "GATE": "*bf16",
                    "SHIFT": "*bf16",
                    "SCALE": "*bf16",
                    "OUT": "*bf16",
                },
                {"AXIS": axis, "ROW_INPUT": row_input},
                2,
            ),
            (
                "update",
                update_packed,
                {
                    "X": "*fp32",
                    "ATTN": "*bf16",
                    "GATE1": "*bf16",
                    "GATE2": "*bf16",
                    "MLP": "*bf16",
                    "OUT": "*fp32",
                },
                {
                    "AXIS": axis,
                    "ROW_INPUT": row_input,
                    "ROW_OUTPUT": row_output,
                    "ROWS": args.update_rows,
                },
                4,
            ),
        ):
            signature.update({key: "constexpr" for key in constants})
            source = GluonASTSource(
                function,
                signature,
                constexprs=constants,
                attrs={(i,): [["tt.divisibility", 16]] for i in range(6)},
            )
            compiled = triton.compile(
                source,
                target=GPUTarget("cuda", 103, 32),
                options={"num_warps": warps, "num_stages": 1, "enable_fp_fusion": True},
            )
            result["cases"].append(
                dict(
                    stage=name,
                    kind=kind,
                    status="compiled",
                    cache_hash=compiled.hash,
                    shared_bytes=compiled.metadata.shared,
                    num_warps=compiled.metadata.num_warps,
                )
            )
            save(args.results, result)
    result["status"] = "complete"
    save(args.results, result)
