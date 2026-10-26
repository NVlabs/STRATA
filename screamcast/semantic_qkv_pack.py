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
"""Fuse semantic QK RMSNorm, RoPE, and native packing; Cirrus-42 (AI agent)."""

import json
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from types import MethodType

import torch
import triton
import triton.language as tl
from natten.token_permute.torch_impl import token_permute_torch, token_unpermute_torch
from physicsnemo.experimental.models.strata import layers
from physicsnemo.nn import apply_rotary_pos_emb
from triton.language.extra.cuda import libdevice

from screamcast import semantic_attention_layout
from screamcast.cuda_timing import measure


@triton.jit
def original_index(permuted, TILE_D: tl.constexpr):
    inner_d = permuted % TILE_D
    value = permuted // TILE_D
    inner_h = value % 4
    value = value // 4
    inner_w = value % 8
    value = value // 8
    block_d = value % (24 // TILE_D)
    value = value // (24 // TILE_D)
    block_h = value % 8
    value = value // 8
    block_w = value % 4
    batch = value // 4
    return (
        batch * 24576
        + (block_d * TILE_D + inner_d) * 1024
        + (block_h * 4 + inner_h) * 32
        + block_w * 8
        + inner_w
    )


@triton.jit
def layout_probe(OUT, TILE_D: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(OUT + index, original_index(index, TILE_D))


@triton.jit
def pack_kernel(INPUT, COS, SIN, Q, K, V, ROWS: tl.constexpr):
    index = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    head = index % 16
    permuted = index // 16
    qtoken = original_index(permuted, 8)
    ktoken = original_index(permuted, 4)
    channel = tl.arange(0, 64)
    qsource = qtoken[:, None] * 3072 + head[:, None] * 64 + channel[None, :]
    ksource = ktoken[:, None] * 3072 + head[:, None] * 64 + channel[None, :] + 1024
    raw_q = tl.load(INPUT + qsource).to(tl.float32)
    raw_k = tl.load(INPUT + ksource).to(tl.float32)
    value = tl.load(INPUT + ksource + 1024)
    qsum = tl.sum(raw_q * raw_q, 1)
    ksum = tl.sum(raw_k * raw_k, 1)
    qscale = libdevice.rsqrt(qsum / 64.0 + 1e-6)
    kscale = libdevice.rsqrt(ksum / 64.0 + 1e-6)
    qnorm = raw_q * qscale[:, None]
    knorm = raw_k * kscale[:, None]
    pair = (channel ^ 1) - channel
    qneighbor = (
        tl.load(INPUT + qsource + pair[None, :]).to(tl.float32) * qscale[:, None]
    )
    kneighbor = (
        tl.load(INPUT + ksource + pair[None, :]).to(tl.float32) * kscale[:, None]
    )
    qrot = tl.where(channel[None, :] % 2 == 0, -qneighbor, qneighbor)
    krot = tl.where(channel[None, :] % 2 == 0, -kneighbor, kneighbor)
    qcos = tl.load(COS + qtoken[:, None] * 64 + channel[None, :])
    qsin = tl.load(SIN + qtoken[:, None] * 64 + channel[None, :])
    kcos = tl.load(COS + ktoken[:, None] * 64 + channel[None, :])
    ksin = tl.load(SIN + ktoken[:, None] * 64 + channel[None, :])
    qoutput = qnorm * qcos + qrot * qsin
    koutput = knorm * kcos + krot * ksin
    destination = index[:, None] * 64 + channel[None, :]
    tl.store(Q + destination, qoutput)
    tl.store(K + destination, koutput)
    tl.store(V + destination, value)


def reference(input, cos, sin):
    q, k, v = input.reshape(8, 24576, 3, 16, 64).permute(2, 0, 3, 1, 4).unbind(0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        q = torch.nn.functional.rms_norm(q, (64,), None, 1e-6)
        k = torch.nn.functional.rms_norm(k, (64,), None, 1e-6)
    q = apply_rotary_pos_emb(q, cos, sin).to(v.dtype)
    k = apply_rotary_pos_emb(k, cos, sin).to(v.dtype)
    grids = [
        t.reshape(8, 16, 24, 32, 32, 64).permute(0, 2, 3, 4, 1, 5) for t in (q, k, v)
    ]
    return tuple(
        token_permute_torch(t, tile, (1, 1, 1), True)
        for t, tile in zip(grids, ((8, 4, 8), (4, 4, 8), (4, 4, 8)))
    )


def evaluate(input, cos, sin, rows, warps, fusion):
    outputs = [
        torch.empty((8, 24576, 16, 64), device=input.device, dtype=torch.bfloat16)
        for _ in range(3)
    ]
    pack_kernel[(3145728 // rows,)](
        input, cos, sin, *outputs, ROWS=rows, num_warps=warps, enable_fp_fusion=fusion
    )
    return tuple(outputs)


def benchmark(commit, results):
    torch.manual_seed(20261020)
    result = dict(
        agent="Cirrus-42 (AI agent)",
        time=datetime.now(timezone.utc).isoformat(),
        commit=commit,
        experiment="semantic-rms-rope-native-pack-component",
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        warmup=5,
        repeats=50,
        seed=20261020,
        cases=[],
        layout_probes=[],
        projection_shape=[8, 24576, 3072],
        rope_shape=[8, 1, 24576, 64],
        scope="Synthetic BF16 projected QKV and FP32 adjacent-pair cosine/sine tables; native RMSNorm and PhysicsNeMo rotary helper, then native Torch token permutations; output allocations included; no attention/projection timing",
    )
    Path(results).write_text(json.dumps(result, indent=2) + "\n")
    with torch.inference_mode():
        ids = torch.arange(196608, device="cuda", dtype=torch.int32).reshape(
            8, 24, 32, 32, 1, 1
        )
        probe = torch.empty((196608,), device="cuda", dtype=torch.int32)
        for tile in ((8, 4, 8), (4, 4, 8)):
            expected_ids = token_permute_torch(ids, tile, (1, 1, 1), True).flatten()
            layout_probe[(192,)](probe, TILE_D=tile[0], BLOCK=1024, num_warps=4)
            different = int((probe != expected_ids).sum())
            result["layout_probes"].append(dict(tile=tile, different_ids=different))
            if different:
                Path(results).write_text(json.dumps(result, indent=2) + "\n")
                raise RuntimeError(f"Native tile layout differs for{tile}:{different}")
        del ids, probe, expected_ids
        input = torch.randn((8, 24576, 3072), device="cuda", dtype=torch.bfloat16)
        angle = torch.randn((8, 1, 24576, 32), device="cuda", dtype=torch.float32)
        cos = angle.cos().repeat_interleave(2, dim=-1).contiguous()
        sin = angle.sin().repeat_interleave(2, dim=-1).contiguous()
        del angle
        control = partial(torch.compile(reference), input, cos, sin)
        expected = control()
        for rows in (16, 32, 64, 128):
            for warps in (4, 8):
                for fusion in (True, False):
                    row = dict(rows=rows, warps=warps, fp_fusion=fusion)
                    candidate = partial(evaluate, input, cos, sin, rows, warps, fusion)
                    try:
                        actual = candidate()
                        row.update(
                            status="complete",
                            different_bits=[
                                int((a.view(torch.int16) != b.view(torch.int16)).sum())
                                for a, b in zip(actual, expected)
                            ],
                            finite=all(bool(x.isfinite().all()) for x in actual),
                        )
                        del actual
                        row["baseline_before"] = measure(control, 5, 50)
                        row["candidate"] = measure(candidate, 5, 50)
                        row["baseline_after"] = measure(control, 5, 50)
                        print(
                            "SEMANTIC_PACK",
                            rows,
                            warps,
                            fusion,
                            row["different_bits"],
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
                        print(
                            "SEMANTIC_PACK_ERROR",
                            rows,
                            warps,
                            fusion,
                            str(error)[:500],
                            flush=True,
                        )
                    result["cases"].append(row)
                    Path(results).write_text(json.dumps(result, indent=2) + "\n")


_ORIGINALS = []


@torch.library.custom_op("cirrus42::semantic_qkv_norm_rope_pack", mutates_args=())
def packed_qkv(
    raw: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if (
        raw.shape != (8, 24576, 3072)
        or raw.dtype != torch.bfloat16
        or not raw.is_cuda
        or not raw.is_contiguous()
    ):
        raise ValueError(
            "Semantic packing requires released contiguous CUDA BF16 QKV projection"
        )
    for table in (cos, sin):
        if (
            table.shape != (8, 1, 24576, 64)
            or table.dtype != torch.float32
            or table.device != raw.device
            or not table.is_contiguous()
        ):
            raise ValueError(
                "Semantic packing requires native contiguous FP32 B8 RoPE tables"
            )
    return evaluate(raw, cos, sin, 16, 4, True)


@packed_qkv.register_fake
def packed_qkv_fake(raw, cos, sin):
    return tuple(
        torch.empty((8, 24576, 16, 64), device=raw.device, dtype=torch.bfloat16)
        for _ in range(3)
    )


def _forward(self, input, latent_dhw=None, rope_tables=None):
    if latent_dhw != (24, 32, 32) or rope_tables is None:
        raise ValueError(
            "Semantic packing requires released geometry and native RoPE tables"
        )
    if (
        not torch.is_autocast_enabled("cuda")
        or torch.get_autocast_dtype("cuda") != torch.bfloat16
    ):
        raise ValueError("Semantic packing requires native BF16 autocast")
    gate = torch.sigmoid(self.gate_proj(input))
    raw = self.qkv(input)
    cos, sin = rope_tables
    q, k, v = packed_qkv(raw, cos, sin)
    output = semantic_attention_layout.attention_core(q, k, v, True)
    output = token_unpermute_torch(
        output, (24, 32, 32), (8, 4, 8), (1, 1, 1), True
    ).reshape(8, 24576, 1024)
    return self.proj_drop(self.proj(output * gate))


def restore():
    for module, had_forward, original in _ORIGINALS:
        if had_forward:
            module.forward = original
        else:
            del module.forward
    _ORIGINALS.clear()


def install(strata):
    restore()
    if (
        strata.training
        or semantic_attention_layout._LAYOUT != (24, 32, 32)
        or semantic_attention_layout._Q_TILE != (8, 4, 8)
        or semantic_attention_layout._KV_TILE != (4, 4, 8)
    ):
        raise ValueError(
            "Semantic packing requires released eval geometry and validated native tiles"
        )
    selected = []
    for module in strata.backbone.modules():
        if (
            isinstance(module, layers.Natten3DSelfAttention)
            and not module.do_depthwise_attention
        ):
            if (
                module.training
                or module.num_heads != 16
                or module.head_dim != 64
                or module.scale != 0.125
                or not module.gated_attention
                or module.attn_kernel != 9
                or module.na_dilation != 1
                or module.na3d_backend not in (None, "blackwell-fna")
            ):
                raise ValueError(
                    "Semantic packing requires released gated nine-cube sixteen-head attention"
                )
            for norm in (module.q_norm, module.k_norm):
                if (
                    not isinstance(norm, torch.nn.RMSNorm)
                    or norm.normalized_shape != (64,)
                    or norm.eps != 1e-6
                    or norm.weight is not None
                ):
                    raise ValueError(
                        "Semantic packing requires nonaffine RMS64 epsilon1e-6"
                    )
            selected.append(module)
    if len(selected) != 12:
        raise ValueError(
            "Semantic packing requires twelve released neighborhood-attention modules"
        )
    for module in selected:
        _ORIGINALS.append((module, "forward" in module.__dict__, module.forward))
        module.forward = MethodType(_forward, module)
