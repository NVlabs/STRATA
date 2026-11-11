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
"""Exact BF16 GELU table and hardware-tanh screen; Cirrus-42 (AI agent)."""

from types import MethodType

import torch
import triton
import triton.language as tl


@triton.jit
def kernel(
    input, table, output, N: tl.constexpr, MODE: tl.constexpr, BLOCK: tl.constexpr
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(input + index, index < N, other=0)
    if MODE == 0:
        bits = value.to(tl.uint16, bitcast=True).to(tl.int32)
        result = tl.load(table + bits)
    else:
        x = value.to(tl.float32)
        half = x * 0.5
        square = x * x
        cube = square * x
        scaled = cube * 0.044715
        argument = (x + scaled) * 0.7978845608028654
        tangent = tl.inline_asm_elementwise(
            "tanh.approx.f32 $0, $1;",
            constraints="=f,f",
            args=[argument],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        result = half * (tangent + 1.0)
    tl.store(output + index, result, index < N)


def evaluate(input, table, *, mode=0, block=1024, warps=4):
    if input.dtype != torch.bfloat16 or not input.is_cuda or not input.is_contiguous():
        raise ValueError("GELU screen requires contiguous CUDA BF16 input")
    output = torch.empty_like(input)
    kernel[(triton.cdiv(input.numel(), block),)](
        input,
        table,
        output,
        N=input.numel(),
        MODE=mode,
        BLOCK=block,
        num_warps=warps,
    )
    return output


_TABLES = {}
_ORIGINALS = []


def prepare_table(device):
    """Compile the native tanh GELU for all BF16 patterns outside timed regions."""
    key = str(device)
    if key not in _TABLES:
        patterns = (
            torch.arange(65536, device=device, dtype=torch.int32)
            .to(torch.int16)
            .view(torch.bfloat16)
        )
        with torch.inference_mode():
            _TABLES[key] = torch.compile(
                lambda input: torch.nn.functional.gelu(input, approximate="tanh")
            )(patterns)
    return _TABLES[key]


@torch.library.custom_op("cirrus42::gelu_lookup", mutates_args=())
def lookup(input: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    if (
        input.shape not in ((8, 24576, 4096), (8, 393216, 512))
        or input.dtype != torch.bfloat16
        or not input.is_cuda
        or not input.is_contiguous()
        or table.shape != (65536,)
        or table.dtype != torch.bfloat16
        or not table.is_contiguous()
        or table.device != input.device
    ):
        raise ValueError(
            "GELU lookup requires fixed BF16 MLP activations and a native table"
        )
    return evaluate(input, table, block=4096)


@lookup.register_fake
def lookup_fake(input, table):
    return torch.empty_like(input)


def _forward(self, input):
    return lookup(input, self._cirrus42_gelu_table)


def restore():
    for module, had_forward, original in _ORIGINALS:
        if had_forward:
            module.forward = original
        else:
            del module.forward
        del module._cirrus42_gelu_table
    _ORIGINALS.clear()


def install(strata, *, pixel=False):
    restore()
    if strata.training:
        raise ValueError("GELU lookup requires eval inference")
    blocks = list(strata.backbone.blocks)
    if len(blocks) != 24 or (pixel and len(strata.pixel_blocks) != 4):
        raise ValueError(
            "GELU lookup requires 24 backbone and optionally 4 pixel blocks"
        )
    if pixel:
        blocks.extend(strata.pixel_blocks)
    activations = [
        layer
        for block in blocks
        for layer in block.mlp.layers
        if isinstance(layer, torch.nn.GELU)
    ]
    if len(activations) != (28 if pixel else 24) or any(
        layer.approximate != "tanh" for layer in activations
    ):
        raise ValueError(
            "GELU lookup requires one tanh GELU activation in every selected block"
        )
    table = prepare_table(next(strata.parameters()).device)
    for module in activations:
        _ORIGINALS.append((module, "forward" in module.__dict__, module.forward))
        module.register_buffer("_cirrus42_gelu_table", table, persistent=False)
        module.forward = MethodType(_forward, module)


def install_pixel(strata):
    """Install the exact lookup only in four pixel MLPs for cached-backbone trials."""
    restore()
    if strata.training or len(strata.pixel_blocks) != 4:
        raise ValueError(
            "Pixel GELU lookup requires the released four-block eval decoder"
        )
    activations = [
        layer
        for block in strata.pixel_blocks
        for layer in block.mlp.layers
        if isinstance(layer, torch.nn.GELU)
    ]
    if len(activations) != 4 or any(
        layer.approximate != "tanh" for layer in activations
    ):
        raise ValueError(
            "Pixel GELU lookup requires one native tanh GELU in each pixel MLP"
        )
    table = prepare_table(next(strata.parameters()).device)
    for module in activations:
        _ORIGINALS.append((module, "forward" in module.__dict__, module.forward))
        module.register_buffer("_cirrus42_gelu_table", table, persistent=False)
        module.forward = MethodType(_forward, module)
