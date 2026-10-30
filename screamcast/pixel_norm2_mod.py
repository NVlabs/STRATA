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
"""Screen residual plus pixel norm2 memory schedules; Cirrus-42 (AI agent)."""

import json
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from types import MethodType

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, SliceLayout
from triton.experimental.gluon.language.extra import libdevice

from screamcast import pixel_norm_mod
from screamcast.cuda_timing import measure
from screamcast.normalization_order import _combine
from screamcast.pixel_norm_mod import load_reference

NATIVE_SOURCE_SHA = "5fbaa863dd89648369865602aceda76f71580c013def8a20e29bb806cefc1326"
NATIVE_NAME = "triton_red_fused__to_copy_add_mul_native_layer_norm_unbind_view_8"


@gluon.jit
def normalize_residual(
    X,
    ATTN,
    GATE,
    SHIFT,
    SCALE,
    OUT,
    ROWS: gl.constexpr,
    RETAIN: gl.constexpr,
    WARPS: gl.constexpr,
):
    LOAD: gl.constexpr = BlockedLayout([4, 1], [2, 16], [1, WARPS], [0, 1])
    COMPUTE: gl.constexpr = BlockedLayout([1, 4], [1, 32], [WARPS, 1], [1, 0])
    row = gl.program_id(0).to(gl.int64) * ROWS + gl.arange(
        0, ROWS, layout=SliceLayout(1, LOAD)
    )
    channel = gl.arange(0, 128, layout=SliceLayout(0, LOAD))
    source = (
        row[:, None] // 393216 * 50331648
        + row[:, None] % 393216
        + channel[None, :] * 393216
    )
    original = gl.convert_layout(
        gl.load(X + source, eviction_policy="evict_last"), COMPUTE
    )
    cr = gl.program_id(0).to(gl.int64) * ROWS + gl.arange(
        0, ROWS, layout=SliceLayout(1, COMPUTE)
    )
    cc = gl.arange(0, 128, layout=SliceLayout(0, COMPUTE))
    mod_offset = cr[:, None] * 768 + cc[None, :]
    attn_offset = cr[:, None] * 128 + cc[None, :]
    gate = gl.load(GATE + mod_offset).to(gl.float32)
    attn = gl.load(ATTN + attn_offset).to(gl.float32)
    residual = original + gate * attn
    average, moment, count = gl.reduce(
        (residual, gl.zeros_like(residual), gl.full_like(residual, 1.0)), 1, _combine
    )
    if not RETAIN:
        original = gl.convert_layout(
            gl.load(X + source, eviction_policy="evict_first"), COMPUTE
        )
        attn = gl.load(ATTN + attn_offset, eviction_policy="evict_first").to(gl.float32)
        gate = gl.load(GATE + mod_offset).to(gl.float32)
        residual = original + gate * attn
    shift = gl.load(SHIFT + mod_offset).to(gl.float32)
    scale = gl.load(SCALE + mod_offset).to(gl.float32)
    normalized = (residual - average[:, None]) * libdevice.rsqrt(
        moment[:, None] / 128.0 + 1e-6
    )
    value = normalized * (scale + 1.0) + shift
    gl.store(OUT + attn_offset, value)


def execute(x, attn, gate, shift, scale, rows, retain, warps):
    out = torch.empty(x.shape, device=x.device, dtype=torch.bfloat16)
    normalize_residual[(3145728 // rows,)](
        x,
        attn,
        gate,
        shift,
        scale,
        out,
        ROWS=rows,
        RETAIN=retain,
        WARPS=warps,
        num_warps=warps,
        num_stages=1,
        enable_fp_fusion=True,
    )
    return out


def evaluate(x, raw, attn, rows, retain, warps):
    return execute(
        x,
        attn,
        raw[:, :, 256:384],
        raw[:, :, 384:512],
        raw[:, :, 512:640],
        rows,
        retain,
        warps,
    )


@torch.library.custom_op("cirrus42::pixel_norm2_modulate", mutates_args=())
def norm2_modulate(
    x: torch.Tensor,
    attn: torch.Tensor,
    gate: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    if (
        x.device.type != "cuda"
        or x.dtype != torch.float32
        or x.shape != (8, 393216, 128)
        or x.stride() != (50331648, 1, 393216)
    ):
        raise ValueError("Pixel norm2 requires released FP32 channel-major states")
    if (
        attn.device != x.device
        or attn.dtype != torch.bfloat16
        or attn.shape != x.shape
        or not attn.is_contiguous()
    ):
        raise ValueError(
            "Pixel norm2 requires released contiguous BF16 attention output"
        )
    for value in (gate, shift, scale):
        if (
            value.device != x.device
            or value.dtype != torch.bfloat16
            or value.shape != x.shape
            or value.stride() != (301989888, 768, 1)
        ):
            raise ValueError("Pixel norm2 requires released BF16 modulation views")
    return execute(x, attn, gate, shift, scale, 8, True, 2)


@norm2_modulate.register_fake
def norm2_modulate_fake(x, attn, gate, shift, scale):
    return torch.empty(x.shape, device=x.device, dtype=torch.bfloat16)


_ORIGINALS = []


def _forward(
    self,
    x,
    backbone_cond,
    pixel_dhw,
    backbone_dhw,
    s_cond_bilinear=None,
    rope_tables=None,
):
    if (
        pixel_dhw != (24, 128, 128)
        or backbone_dhw != (24, 32, 32)
        or s_cond_bilinear is None
    ):
        raise ValueError("Pixel norm2 requires released bilinear conditioning geometry")
    (
        shift1,
        scale1,
        gate1,
        shift2,
        scale2,
        gate2,
    ) = self._compute_bilinear_dw_adaln_params(s_cond_bilinear)
    y = pixel_norm_mod.norm1_modulate(x, shift1, scale1)
    attn = self.attn(y, latent_dhw=pixel_dhw, rope_tables=rope_tables)
    y = norm2_modulate(x, attn, gate1, shift2, scale2)
    mlp = self.mlp(y)
    return (x + gate1 * attn) + gate2 * mlp


def restore():
    for block, had_forward, original in _ORIGINALS:
        if had_forward:
            block.forward = original
        else:
            del block.forward
    _ORIGINALS.clear()


def install(strata):
    restore()
    if (
        strata.training
        or len(strata.pixel_blocks) != 4
        or len(pixel_norm_mod._ORIGINALS) != 4
    ):
        raise ValueError("Pixel norm2 requires four validated eval norm1 pixel blocks")
    for block in strata.pixel_blocks:
        if (
            block not in [m for m, _, _ in pixel_norm_mod._ORIGINALS]
            or block.forward.__func__ is not pixel_norm_mod._forward
            or block.training
            or block.adaln_mode != "bilinear_dw"
            or block.num_adaln_params != 6
            or not isinstance(block.norm2, torch.nn.LayerNorm)
            or block.norm2.normalized_shape != (128,)
            or block.norm2.eps != 1e-6
            or block.norm2.weight is not None
            or block.norm2.bias is not None
        ):
            raise ValueError(
                "Pixel norm2 requires native nonaffine LayerNorm128 epsilon1e-6 and validated norm1"
            )
    for block in strata.pixel_blocks:
        _ORIGINALS.append((block, "forward" in block.__dict__, block.forward))
        block.forward = MethodType(_forward, block)


def native(reference, stream, x, raw, attn):
    out = torch.empty(x.shape, device=x.device, dtype=torch.bfloat16)
    reference.run(x, raw, attn, out, 3145728, 128, stream=stream)
    return out


def benchmark(commit, native_bundle, results):
    reference, metadata = load_reference(
        native_bundle,
        Path(results).parent,
        outer_sha=NATIVE_SOURCE_SHA,
        kernel_name=NATIVE_NAME,
    )
    result = dict(
        agent="Cirrus-42 (AI agent)",
        time=datetime.now(timezone.utc).isoformat(),
        commit=commit,
        experiment="pixel-residual-norm2-warp-tiles",
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        warmup=5,
        repeats=50,
        native_reference=metadata,
        cases=[],
        scope="Synthetic FP32 channel-major B8x393216x128 states, BF16 contiguous attention and BF16 B8x393216x768 modulation. Original executed X8/R128/eightwarp norm2 reference includes attention gate/residual before Welford. Preserve its contiguous-channel reduction layout and FP32 arithmetic, BF16 MLP input. Candidate retained residual, rows2/4/8/16/32, one/two/four/eight warps with identical per-row channel reduction layout. Two seeds, all allocations included.",
    )
    path = Path(results)
    path.write_text(json.dumps(result, indent=2) + "\n")
    stream = torch.cuda.current_stream().cuda_stream
    with torch.inference_mode():
        for seed in (20261020, 20261021):
            torch.manual_seed(seed)
            x = torch.randn(
                (8, 128, 393216), device="cuda", dtype=torch.float32
            ).transpose(1, 2)
            raw = torch.randn(
                (8, 393216, 768), device="cuda", dtype=torch.bfloat16
            ).mul_(0.1)
            attn = torch.randn((8, 393216, 128), device="cuda", dtype=torch.bfloat16)
            control = partial(native, reference, stream, x, raw, attn)
            expected = control()
            for rows in (2, 4, 8, 16, 32):
                for warps in (1, 2, 4, 8):
                    retain = True
                    row = dict(seed=seed, rows=rows, retain_input=retain, warps=warps)
                    try:
                        candidate = partial(evaluate, x, raw, attn, rows, retain, warps)
                        actual = candidate()
                        different = sum(
                            int((a.view(torch.int16) != b.view(torch.int16)).sum())
                            for a, b in zip(actual.split(1), expected.split(1))
                        )
                        row.update(
                            status="complete",
                            different_bits=different,
                            finite=bool(actual.isfinite().all()),
                        )
                        del actual
                        row["baseline_before"] = measure(control, 5, 50)
                        row["candidate"] = measure(candidate, 5, 50)
                        row["baseline_after"] = measure(control, 5, 50)
                        print(
                            "PIXEL_NORM2",
                            seed,
                            rows,
                            retain,
                            warps,
                            different,
                            {
                                k: row[k]["median_ms"]
                                for k in (
                                    "baseline_before",
                                    "candidate",
                                    "baseline_after",
                                )
                            },
                            flush=True,
                        )
                    except Exception as error:
                        row.update(status="error", error=repr(error))
                        result["cases"].append(row)
                        path.write_text(json.dumps(result, indent=2) + "\n")
                        raise
                    result["cases"].append(row)
                    path.write_text(json.dumps(result, indent=2) + "\n")
            del x, raw, attn, expected, control
