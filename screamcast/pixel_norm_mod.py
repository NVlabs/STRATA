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
"""Screen exact pixel LayerNorm/modulation memory schedules; Cirrus-42 (AI agent)."""

import ast
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from types import MethodType

import torch
from physicsnemo.experimental.models.strata import layers
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, SliceLayout
from triton.experimental.gluon.language.extra import libdevice

from screamcast import pixel_qkv_pack
from screamcast.cuda_timing import measure
from screamcast.normalization_order import _combine

NATIVE_SOURCE_SHA = "88350739cfefd55f185ca1c907da208b932090499e49883d6df51352caf77757"
NATIVE_NAME = "triton_red_fused__to_copy_add_mul_native_layer_norm_unbind_view_4"


def load_reference(
    bundle_path, output_dir, *, outer_sha=NATIVE_SOURCE_SHA, kernel_name=NATIVE_NAME
):
    bundle = json.loads(Path(bundle_path).read_text())
    outer = bundle["generated"][0]["source"]
    if hashlib.sha256(outer.encode()).hexdigest() != outer_sha:
        raise ValueError(
            "Native pixel normalization source differs from recorded reference"
        )
    tree = ast.parse(outer)
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == kernel_name
            for target in node.targets
        )
    )
    source = ast.literal_eval(assignment.value.args[1])
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    path = Path(output_dir) / ("native-pixel-norm-" + source_hash[:16] + ".py")
    if path.exists() and path.read_text() != source:
        raise ValueError("Reference module path contains different source")
    path.write_text(source)
    name = "cirrus42_native_pixel_norm_" + source_hash[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    reference = module.__dict__[kernel_name]
    configs = [
        c
        for c in reference.configs
        if c.kwargs.get("XBLOCK") == 8
        and c.kwargs.get("R0_BLOCK") == 128
        and c.num_warps == 8
        and c.num_stages == 1
    ]
    if len(configs) != 1:
        raise ValueError(
            "Recorded executed X8/R128/eight-warp configuration unavailable"
        )
    reference.configs = configs
    metadata = dict(
        source_path=str(path),
        source_sha256=source_hash,
        outer_sha256=outer_sha,
        xblock=8,
        rblock=128,
        num_warps=8,
        num_stages=1,
        enable_fp_fusion=True,
        executed_profile_grid=[393216, 1, 1],
        executed_profile_block=[256, 1, 1],
        executed_profile_registers=32,
        executed_profile_shared_bytes=4096,
    )
    return reference, metadata


def native(reference, stream, x, raw):
    out = torch.empty((8, 393216, 128), device=x.device, dtype=torch.bfloat16)
    reference.run(x, raw, out, 3145728, 128, stream=stream)
    return out


@gluon.jit
def normalize_modulate(X, SHIFT, SCALE, OUT, ROWS: gl.constexpr, RETAIN: gl.constexpr):
    COMPUTE: gl.constexpr = BlockedLayout([4, 1], [2, 16], [1, 8], [0, 1])
    STORE: gl.constexpr = BlockedLayout([1, 4], [1, 32], [8, 1], [1, 0])
    row = gl.program_id(0).to(gl.int64) * ROWS + gl.arange(
        0, ROWS, layout=SliceLayout(1, COMPUTE)
    )
    channel = gl.arange(0, 128, layout=SliceLayout(0, COMPUTE))
    source = (
        (row[:, None] // 393216) * 50331648
        + (row[:, None] % 393216)
        + channel[None, :] * 393216
    )
    loaded = gl.load(X + source, eviction_policy="evict_last")
    average, moment, count = gl.reduce(
        (loaded, gl.zeros_like(loaded), gl.full_like(loaded, 1.0)), 1, _combine
    )
    if not RETAIN:
        loaded = gl.load(X + source, eviction_policy="evict_first")
    scale = libdevice.rsqrt(moment / 128.0 + 1e-6)
    normalized = (loaded - average[:, None]) * scale[:, None]
    normalized = gl.convert_layout(normalized, STORE)
    out_row = gl.program_id(0).to(gl.int64) * ROWS + gl.arange(
        0, ROWS, layout=SliceLayout(1, STORE)
    )
    out_channel = gl.arange(0, 128, layout=SliceLayout(0, STORE))
    shift = gl.load(SHIFT + out_row[:, None] * 768 + out_channel[None, :]).to(
        gl.float32
    )
    modulation = gl.load(SCALE + out_row[:, None] * 768 + out_channel[None, :]).to(
        gl.float32
    )
    value = normalized * (modulation + 1.0) + shift
    gl.store(OUT + out_row[:, None] * 128 + out_channel[None, :], value)


def execute(x, shift, scale, rows, retain):
    out = torch.empty((8, 393216, 128), device=x.device, dtype=torch.bfloat16)
    normalize_modulate[(3145728 // rows,)](
        x,
        shift,
        scale,
        out,
        ROWS=rows,
        RETAIN=retain,
        num_warps=8,
        num_stages=1,
        enable_fp_fusion=True,
    )
    return out


def evaluate(x, raw, rows, retain):
    return execute(x, raw[:, :, :128], raw[:, :, 128:256], rows, retain)


@torch.library.custom_op("cirrus42::pixel_norm1_modulate", mutates_args=())
def norm1_modulate(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    if (
        x.device.type != "cuda"
        or x.dtype != torch.float32
        or x.shape != (8, 393216, 128)
        or x.stride() != (50331648, 1, 393216)
    ):
        raise ValueError("Pixel norm1 requires released FP32 channel-major states")
    for value in (shift, scale):
        if (
            value.device != x.device
            or value.dtype != torch.bfloat16
            or value.shape != x.shape
            or value.stride() != (301989888, 768, 1)
        ):
            raise ValueError("Pixel norm1 requires released BF16 modulation views")
    return execute(x, shift, scale, 32, True)


@norm1_modulate.register_fake
def norm1_modulate_fake(x, shift, scale):
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
        raise ValueError("Pixel norm1 requires released bilinear conditioning geometry")
    (
        shift1,
        scale1,
        gate1,
        shift2,
        scale2,
        gate2,
    ) = self._compute_bilinear_dw_adaln_params(s_cond_bilinear)
    y = norm1_modulate(x, shift1, scale1)
    z = self.attn(y, latent_dhw=pixel_dhw, rope_tables=rope_tables)
    x = x + gate1 * z
    y = self.norm2(x) * (1 + scale2) + shift2
    z = self.mlp(y)
    return x + gate2 * z


def restore():
    for block, had_forward, original in _ORIGINALS:
        if had_forward:
            block.forward = original
        else:
            del block.forward
    _ORIGINALS.clear()


def install(strata):
    restore()
    if strata.training or len(strata.pixel_blocks) != 4:
        raise ValueError("Pixel norm1 requires four released eval pixel blocks")
    for block in strata.pixel_blocks:
        if (
            not isinstance(block, layers.StrataPixel3DBlock)
            or block.training
            or block.adaln_mode != "bilinear_dw"
            or block.num_adaln_params != 6
            or not isinstance(block.norm1, torch.nn.LayerNorm)
            or block.norm1.normalized_shape != (128,)
            or block.norm1.eps != 1e-6
            or block.norm1.weight is not None
            or block.norm1.bias is not None
            or block.attn not in [m for m, _, _ in pixel_qkv_pack._ORIGINALS]
            or block.attn.forward.__func__ is not pixel_qkv_pack._forward
        ):
            raise ValueError(
                "Pixel norm1 requires nonaffine LayerNorm128 epsilon1e-6 and validated packed attention"
            )
    for block in strata.pixel_blocks:
        _ORIGINALS.append((block, "forward" in block.__dict__, block.forward))
        block.forward = MethodType(_forward, block)


def benchmark(commit, native_bundle, results):
    reference, metadata = load_reference(native_bundle, Path(results).parent)
    result = dict(
        agent="Cirrus-42 (AI agent)",
        time=datetime.now(timezone.utc).isoformat(),
        commit=commit,
        experiment="pixel-layernorm-modulation-extended-tiles",
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        warmup=5,
        repeats=50,
        native_reference=metadata,
        cases=[],
        scope="Synthetic channel-major FP32 B8x393216x128 pixel states and BF16 B8x393216x768 modulation projection. Reference is original generated native kernel pinned to actually executed X8/R128/eight-warps/stage1/FPfusionTrue. Candidate preserves Welford helper/order, FP32 scaling/shift and BF16 linear-input output; vary tile rows and input retention. All output allocations included; two seeds.",
    )
    Path(results).write_text(json.dumps(result, indent=2) + "\n")
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
            control = partial(native, reference, stream, x, raw)
            expected = control()
            for rows in (8, 16, 32, 64, 128):
                for retain in (False, True):
                    row = dict(seed=seed, rows=rows, retain_input=retain)
                    try:
                        candidate = partial(evaluate, x, raw, rows, retain)
                        actual = candidate()
                        different = sum(
                            int((aa.view(torch.int16) != bb.view(torch.int16)).sum())
                            for aa, bb in zip(actual.split(1), expected.split(1))
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
                            "PIXEL_NORM_MOD",
                            seed,
                            rows,
                            retain,
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
                        Path(results).write_text(json.dumps(result, indent=2) + "\n")
                        raise
                    result["cases"].append(row)
                    Path(results).write_text(json.dumps(result, indent=2) + "\n")
            del candidate, control, expected, x, raw
