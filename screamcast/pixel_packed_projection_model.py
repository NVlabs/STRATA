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
"""Keep projected pixel attention packed through residuals; Cirrus-42 (AI agent)."""

from types import MethodType

import torch

from screamcast import (
    attention_layout,
    pixel_axis_attention,
    pixel_norm_mod,
    pixel_persistent,
    pixel_qkv_pack,
)
from screamcast import pixel_packed_projection as kernels
from screamcast import pixel_state_layout_model as parent

UPDATE_ROWS = 32
_ORIGINALS = []


def validate(x, attn, modulation, row_input):
    stride = (50331648, 128, 1) if row_input else (50331648, 1, 393216)
    if (
        x.shape != (8, 393216, 128)
        or x.dtype != torch.float32
        or not x.is_cuda
        or x.stride() != stride
    ):
        raise ValueError("Packed residual requires released CUDA FP32 state layout")
    if (
        attn.shape != x.shape
        or attn.dtype != torch.bfloat16
        or attn.device != x.device
        or not attn.is_contiguous()
    ):
        raise ValueError(
            "Packed residual requires contiguous CUDA BF16 projected attention"
        )
    for value in modulation:
        if (
            value.shape != x.shape
            or value.dtype != torch.bfloat16
            or value.device != x.device
            or value.stride() != (301989888, 768, 1)
        ):
            raise ValueError("Packed residual requires native BF16 modulation views")


@torch.library.custom_op("cirrus42::pixel_packed_projected_norm2", mutates_args=())
def norm2(
    x: torch.Tensor,
    attn: torch.Tensor,
    gate: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    axis: bool,
    row_input: bool,
) -> torch.Tensor:
    validate(x, attn, (gate, shift, scale), row_input)
    output = torch.empty(x.shape, device=x.device, dtype=torch.bfloat16)
    kernels.normalize_packed[(3145728 // 8,)](
        x,
        attn,
        gate,
        shift,
        scale,
        output,
        AXIS=axis,
        ROW_INPUT=row_input,
        num_warps=2,
        num_stages=1,
        enable_fp_fusion=True,
    )
    return output


@norm2.register_fake
def norm2_fake(x, attn, gate, shift, scale, axis, row_input):
    validate(x, attn, (gate, shift, scale), row_input)
    return torch.empty(x.shape, device=x.device, dtype=torch.bfloat16)


@torch.library.custom_op("cirrus42::pixel_packed_projected_update", mutates_args=())
def update(
    x: torch.Tensor,
    attn: torch.Tensor,
    gate1: torch.Tensor,
    gate2: torch.Tensor,
    mlp: torch.Tensor,
    axis: bool,
    row_input: bool,
    row_output: bool,
) -> torch.Tensor:
    validate(x, attn, (gate1, gate2), row_input)
    if (
        mlp.shape != x.shape
        or mlp.dtype != torch.bfloat16
        or mlp.device != x.device
        or not mlp.is_contiguous()
    ):
        raise ValueError("Packed update requires contiguous CUDA BF16 MLP output")
    shape = (8, 393216, 128) if row_output else (8, 128, 393216)
    output = torch.empty(shape, device=x.device, dtype=torch.float32)
    kernels.update_packed[(3145728 // UPDATE_ROWS,)](
        x,
        attn,
        gate1,
        gate2,
        mlp,
        output,
        AXIS=axis,
        ROW_INPUT=row_input,
        ROW_OUTPUT=row_output,
        ROWS=UPDATE_ROWS,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=True,
    )
    return output if row_output else output.transpose(1, 2)


@update.register_fake
def update_fake(x, attn, gate1, gate2, mlp, axis, row_input, row_output):
    validate(x, attn, (gate1, gate2), row_input)
    if (
        mlp.shape != x.shape
        or mlp.dtype != torch.bfloat16
        or mlp.device != x.device
        or not mlp.is_contiguous()
    ):
        raise ValueError("Packed update requires contiguous CUDA BF16 MLP output")
    if row_output:
        return torch.empty(x.shape, device=x.device, dtype=torch.float32)
    return torch.empty(
        (8, 128, 393216), device=x.device, dtype=torch.float32
    ).transpose(1, 2)


def attention_forward_for(axis):
    def forward(self, input, latent_dhw=None, rope_tables=None):
        if latent_dhw != (24, 128, 128) or rope_tables is not None:
            raise ValueError(
                "Packed projection requires released pixel geometry without RoPE"
            )
        if (
            not torch.is_autocast_enabled("cuda")
            or torch.get_autocast_dtype("cuda") != torch.bfloat16
        ):
            raise ValueError("Packed projection requires native BF16 autocast")
        raw = self.qkv(input)
        if axis:
            output = pixel_axis_attention.attention(
                *pixel_axis_attention.packed_qkv(raw)
            )
        else:
            output = pixel_persistent.persistent_core(*pixel_qkv_pack.packed_qkv(raw))
        return self.proj_drop(self.proj(output.reshape(8, 393216, 128)))

    return forward


def block_forward_for(row_input, row_output, axis):
    def forward(
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
            raise ValueError(
                "Packed projection requires released bilinear conditioning geometry"
            )
        (
            shift1,
            scale1,
            gate1,
            shift2,
            scale2,
            gate2,
        ) = self._compute_bilinear_dw_adaln_params(s_cond_bilinear)
        y = (
            parent.norm1(x, shift1, scale1)
            if row_input
            else pixel_norm_mod.norm1_modulate(x, shift1, scale1)
        )
        attn = self.attn(y, latent_dhw=pixel_dhw, rope_tables=rope_tables)
        y = norm2(x, attn, gate1, shift2, scale2, axis, row_input)
        mlp = self.mlp(y)
        return update(x, attn, gate1, gate2, mlp, axis, row_input, row_output)

    return forward


_ATTN_FORWARDS = (attention_forward_for(False), attention_forward_for(True))
_BLOCK_FORWARDS = (
    block_forward_for(False, True, False),
    block_forward_for(True, True, False),
    block_forward_for(True, True, False),
    block_forward_for(True, False, True),
)


def restore():
    for module, had_forward, original in reversed(_ORIGINALS):
        if had_forward:
            module.forward = original
        else:
            del module.forward
    _ORIGINALS.clear()


def install(strata):
    restore()
    if (
        strata.training
        or len(strata.pixel_blocks) != 4
        or len(parent._ORIGINALS) != 4
        or attention_layout.attention_core is not pixel_persistent.persistent_core
    ):
        raise ValueError(
            "Packed projection requires the validated eval row-state parent"
        )
    owned = {id(block) for block, _, _ in parent._ORIGINALS}
    if owned != {id(block) for block in strata.pixel_blocks} or [
        m for m, _, _ in pixel_axis_attention._ORIGINALS
    ] != [strata.pixel_blocks[-1].attn]:
        raise ValueError("Packed projection requires this model's row/last-axis owners")
    for index, block in enumerate(strata.pixel_blocks):
        attn_forward = (
            pixel_axis_attention._forward if index == 3 else pixel_qkv_pack._forward
        )
        if (
            block.training
            or not isinstance(block.forward, MethodType)
            or block.forward.__func__ is not parent._FORWARDS[index]
            or not isinstance(block.attn.forward, MethodType)
            or block.attn.forward.__func__ is not attn_forward
        ):
            raise ValueError(
                "Packed projection requires unmodified owned parent forwards"
            )
    for index, block in enumerate(strata.pixel_blocks):
        for module, forward in (
            (block.attn, _ATTN_FORWARDS[index == 3]),
            (block, _BLOCK_FORWARDS[index]),
        ):
            _ORIGINALS.append((module, "forward" in module.__dict__, module.forward))
            module.forward = MethodType(forward, module)
