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
"""Pixel attention with final short axis; Cirrus-42 (AI agent)."""

from datetime import datetime, timezone
from functools import partial
from types import MethodType

import torch
import triton
import triton.language as tl
from natten._libnatten import blackwell_na3d_forward
from natten.token_permute.torch_impl import token_permute_torch, token_unpermute_torch
from physicsnemo.experimental.models.strata import layers
from triton.language.extra.cuda import libdevice

from screamcast import attention_layout, pixel_persistent, pixel_qkv_pack
from screamcast.benchmark_statistics import latency_gate
from screamcast.cuda_timing import measure
from screamcast.gemm_schedule_validation import difference, save


@triton.jit
def original_index(permuted, TILE_D: tl.constexpr):
    inner_d = permuted % TILE_D
    value = permuted // TILE_D
    inner_h = value % 4
    value = value // 4
    inner_w = value % 8
    value = value // 8
    block_d = value % (128 // TILE_D)
    value = value // (128 // TILE_D)
    block_h = value % 32
    value = value // 32
    block_w = value % 3
    batch = value // 3
    return (
        batch * 393216
        + (block_w * 8 + inner_w) * 16384
        + (block_h * 4 + inner_h) * 128
        + block_d * TILE_D
        + inner_d
    )


@triton.jit
def packed_index(original):
    batch = original // 393216
    d = original // 16384 % 24
    h = original // 128 % 128
    w = original % 128
    block = ((batch * 3 + d // 8) * 32 + h // 4) * 16 + w // 8
    return block * 256 + (d % 8) * 32 + (h % 4) * 8 + w % 8


@triton.jit
def index_probe(OUT, TD: tl.constexpr, INVERSE: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if INVERSE:
        value = packed_index(index)
    else:
        value = original_index(index, TD)
    tl.store(OUT + index, value)


@triton.jit
def pack_kernel(INPUT, Q, K, V, ROWS: tl.constexpr):
    index = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    head = index % 2
    permuted = index // 2
    qtoken = original_index(permuted, 8)
    ktoken = original_index(permuted, 4)
    channel = tl.arange(0, 64)
    qsource = qtoken[:, None] * 384 + head[:, None] * 64 + channel[None, :]
    ksource = ktoken[:, None] * 384 + head[:, None] * 64 + channel[None, :] + 128
    raw_q = tl.load(INPUT + qsource).to(tl.float32)
    raw_k = tl.load(INPUT + ksource).to(tl.float32)
    value = tl.load(INPUT + ksource + 128)
    qsum = tl.sum(raw_q * raw_q, 1)
    ksum = tl.sum(raw_k * raw_k, 1)
    qscale = libdevice.rsqrt(qsum / 64.0 + 1e-6)
    kscale = libdevice.rsqrt(ksum / 64.0 + 1e-6)
    destination = index[:, None] * 64 + channel[None, :]
    tl.store(Q + destination, raw_q * qscale[:, None])
    tl.store(K + destination, raw_k * kscale[:, None])
    tl.store(V + destination, value)


@triton.jit
def restore_kernel(INPUT, OUT, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token = index // 128
    channel = index % 128
    source = packed_index(token) * 128 + channel
    tl.store(OUT + index, tl.load(INPUT + source))


@torch.library.custom_op("cirrus42::pixel_axis_qkv", mutates_args=())
def packed_qkv(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if (
        raw.shape != (8, 393216, 384)
        or raw.dtype != torch.bfloat16
        or not raw.is_cuda
        or not raw.is_contiguous()
    ):
        raise ValueError(
            "Pixel axis packing requires contiguous CUDA BF16 released QKV"
        )
    outputs = [
        torch.empty((8, 393216, 2, 64), device=raw.device, dtype=torch.bfloat16)
        for _ in range(3)
    ]
    pack_kernel[(6291456 // 128,)](
        raw, *outputs, ROWS=128, num_warps=4, enable_fp_fusion=True
    )
    return tuple(outputs)


@packed_qkv.register_fake
def packed_qkv_fake(raw):
    return tuple(
        torch.empty((8, 393216, 2, 64), device=raw.device, dtype=torch.bfloat16)
        for _ in range(3)
    )


@torch.library.custom_op("cirrus42::pixel_axis_core", mutates_args=())
def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    for tensor in (q, k, v):
        if (
            tensor.shape != (8, 393216, 2, 64)
            or tensor.dtype != torch.bfloat16
            or not tensor.is_cuda
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                "Pixel axis core requires released contiguous packed BF16 operands"
            )
    output, _ = blackwell_na3d_forward(
        q,
        k,
        v,
        (9, 9, 9),
        (1, 1, 1),
        (1, 1, 1),
        (False, False, False),
        0.125,
        (128, 128, 24),
        (128, 128, 24),
        (128, 128, 24),
        (8, 4, 8),
        (4, 4, 8),
        True,
    )
    return output


@attention.register_fake
def attention_fake(q, k, v):
    return torch.empty_like(q)


@torch.library.custom_op("cirrus42::pixel_axis_restore", mutates_args=())
def restore_output(input: torch.Tensor) -> torch.Tensor:
    if (
        input.shape != (8, 393216, 2, 64)
        or input.dtype != torch.bfloat16
        or not input.is_cuda
        or not input.is_contiguous()
    ):
        raise ValueError(
            "Pixel axis restoration requires contiguous released packed BF16 output"
        )
    output = torch.empty((8, 393216, 128), device=input.device, dtype=input.dtype)
    restore_kernel[(402653184 // 4096,)](input, output, BLOCK=4096, num_warps=4)
    return output


@restore_output.register_fake
def restore_output_fake(input):
    return torch.empty((8, 393216, 128), device=input.device, dtype=input.dtype)


def native_restore(input):
    return token_unpermute_torch(
        input, (24, 128, 128), (8, 4, 8), (1, 1, 1), True
    ).reshape(8, 393216, 128)


def baseline(raw, compiled_restore):
    return compiled_restore(
        pixel_persistent.persistent_core(*pixel_qkv_pack.packed_qkv(raw))
    )


def candidate(raw):
    return restore_output(attention(*packed_qkv(raw)))


def run(args):
    result = dict(
        agent="Cirrus-42 (AI agent)",
        source_commit=args.commit,
        time=datetime.now(timezone.utc).isoformat(),
        experiment="registered-pixel-axis-packed-attention",
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        warmup=5,
        repeats=100,
        seeds=[20261050, 20261051],
        physical_shape=[128, 128, 24],
        layout_probes=[],
        samples=[],
        scope="Synthetic actual-shape BF16 raw QKV. Both paths include registered QK RMS64/packing, native persistent9cube attention and restoration/output allocations. Candidate fuses axis change into pack/restore; all original accuracy/timing gates required before model integration.",
    )
    save(args.results, result)
    with torch.inference_mode():
        ids = torch.arange(3145728, device="cuda", dtype=torch.int32)
        grid = ids.reshape(8, 24, 128, 128, 1, 1).permute(0, 3, 2, 1, 4, 5).contiguous()
        for td in (8, 4):
            expected = token_permute_torch(grid, (td, 4, 8), (1, 1, 1), True).flatten()
            actual = torch.empty_like(ids)
            index_probe[(3145728 // 1024,)](actual, TD=td, INVERSE=False, BLOCK=1024)
            result["layout_probes"].append(
                dict(td=td, different=int((actual != expected).sum()))
            )
        expected = (
            token_unpermute_torch(
                ids.reshape(8, 393216, 1, 1), (128, 128, 24), (8, 4, 8), (1, 1, 1), True
            )
            .permute(0, 3, 2, 1, 4, 5)
            .contiguous()
            .flatten()
        )
        index_probe[(3145728 // 1024,)](actual, TD=8, INVERSE=True, BLOCK=1024)
        result["layout_probes"].append(
            dict(direction="restore", different=int((actual != expected).sum()))
        )
        save(args.results, result)
        if any(probe["different"] for probe in result["layout_probes"]):
            raise RuntimeError(f"Axis layout probes failed: {result['layout_probes']}")
        del ids, grid, expected, actual
        compiled_restore = torch.compile(native_restore)
        for seed in result["seeds"]:
            torch.manual_seed(seed)
            raw = torch.randn((8, 393216, 384), device="cuda", dtype=torch.bfloat16)
            control = partial(baseline, raw, compiled_restore)
            changed = partial(candidate, raw)
            expected = control()
            actual = changed()
            sample = dict(
                seed=seed,
                error=difference(actual.reshape(-1, 128), expected.reshape(-1, 128)),
            )
            old_qkv = pixel_qkv_pack.packed_qkv(raw)
            new_qkv = packed_qkv(raw)
            sample["restored_qkv_different_bits"] = []
            for index, (old, new) in enumerate(zip(old_qkv, new_qkv)):
                tile = (8, 4, 8) if index == 0 else (4, 4, 8)
                old_grid = token_unpermute_torch(
                    old, (24, 128, 128), tile, (1, 1, 1), True
                )
                new_grid = (
                    token_unpermute_torch(new, (128, 128, 24), tile, (1, 1, 1), True)
                    .permute(0, 3, 2, 1, 4, 5)
                    .contiguous()
                )
                sample["restored_qkv_different_bits"].append(
                    int(
                        (old_grid.view(torch.int16) != new_grid.view(torch.int16)).sum()
                    )
                )
            del expected, actual, old_qkv, new_qkv, old_grid, new_grid
            sample["before"] = measure(control, 5, 100)
            sample["candidate"] = measure(changed, 5, 100)
            sample["after"] = measure(control, 5, 100)
            result["samples"].append(sample)
            save(args.results, result)
            print(
                "REGISTERED_PIXEL_AXIS",
                seed,
                sample["error"],
                sample["restored_qkv_different_bits"],
                {k: sample[k]["median_ms"] for k in ("before", "candidate", "after")},
                flush=True,
            )
            del raw, control, changed
        result["gate"] = latency_gate(*result["samples"], isolated_scope="seed20261050")
        result["gate"]["comparisons"] = {
            key.replace("full_", "seed20261051_"): value
            for key, value in result["gate"]["comparisons"].items()
        }
        result["component_eligible"] = result["gate"]["significant"] and all(
            sample["error"]["finite"]
            and sample["error"]["relative_rms"] <= 0.005
            and sample["error"]["max_channel_relative_rms"] <= 0.01
            and not any(sample["restored_qkv_different_bits"])
            for sample in result["samples"]
        )
        result["status"] = "complete"
        save(args.results, result)


_ORIGINALS = []


def _forward(self, input, latent_dhw=None, rope_tables=None):
    if latent_dhw != (24, 128, 128) or rope_tables is not None:
        raise ValueError(
            "Pixel axis attention requires released geometry and no RoPE tables"
        )
    if (
        not torch.is_autocast_enabled("cuda")
        or torch.get_autocast_dtype("cuda") != torch.bfloat16
    ):
        raise ValueError("Pixel axis attention requires native BF16 autocast")
    raw = self.qkv(input)
    output = restore_output(attention(*packed_qkv(raw)))
    return self.proj_drop(self.proj(output))


def restore():
    for module, had_forward, original in _ORIGINALS:
        if had_forward:
            module.forward = original
        else:
            del module.forward
    _ORIGINALS.clear()


def install(strata, *, last_only=False):
    restore()
    selected = [
        module
        for module in strata.pixel_blocks.modules()
        if isinstance(module, layers.Natten3DSelfAttention)
        and not module.do_depthwise_attention
    ]
    owned = {id(module) for module, _, _ in pixel_qkv_pack._ORIGINALS}
    if (
        strata.training
        or len(selected) != 4
        or {id(module) for module in selected} != owned
        or attention_layout.attention_core is not pixel_persistent.persistent_core
    ):
        raise ValueError(
            "Pixel axis attention requires four owned eval packed persistent pixel modules"
        )
    for module in selected:
        if (
            module.training
            or module.num_heads != 2
            or module.head_dim != 64
            or module.scale != 0.125
            or module.gated_attention
            or module.attn_kernel != 9
            or module.na_dilation != 1
            or module.na3d_backend not in (None, "blackwell-fna")
        ):
            raise ValueError(
                "Pixel axis attention requires released ungated nine-cube geometry"
            )
        for norm in (module.q_norm, module.k_norm):
            if (
                not isinstance(norm, torch.nn.RMSNorm)
                or norm.normalized_shape != (64,)
                or norm.eps != 1e-6
                or norm.weight is not None
            ):
                raise ValueError(
                    "Pixel axis attention requires native nonaffine RMS64 epsilon1e-6"
                )
    for module in selected[-1:] if last_only else selected:
        _ORIGINALS.append((module, "forward" in module.__dict__, module.forward))
        module.forward = MethodType(_forward, module)
