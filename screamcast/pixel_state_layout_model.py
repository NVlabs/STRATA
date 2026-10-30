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
"""Install intermediate row-major pixel states; Cirrus-42 (AI agent)."""

from types import MethodType

import torch

from screamcast import pixel_norm2_mod, pixel_norm_mod, pixel_state_layout

_ORIGINALS = []


def validate(x, modulation):
    if (
        x.device.type != "cuda"
        or x.dtype != torch.float32
        or x.shape != (8, 393216, 128)
        or not x.is_contiguous()
    ):
        raise ValueError(
            "Row pixel normalization requires released contiguous CUDA FP32 BNC state"
        )
    for value in modulation:
        if (
            value.device != x.device
            or value.dtype != torch.bfloat16
            or value.shape != x.shape
            or value.stride() != (301989888, 768, 1)
        ):
            raise ValueError(
                "Row pixel normalization requires released BF16 modulation views"
            )


@torch.library.custom_op("cirrus42::pixel_state_row_norm1", mutates_args=())
def norm1(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    validate(x, (shift, scale))
    out = torch.empty(x.shape, device=x.device, dtype=torch.bfloat16)
    pixel_state_layout.norm1_row_kernel[(3145728 // 32,)](
        x, shift, scale, out, ROWS=32, num_warps=8, num_stages=1, enable_fp_fusion=True
    )
    return out


@norm1.register_fake
def norm1_fake(x, shift, scale):
    validate(x, (shift, scale))
    return torch.empty(x.shape, device=x.device, dtype=torch.bfloat16)


def validate_attention(x, attn):
    if (
        attn.device != x.device
        or attn.dtype != torch.bfloat16
        or attn.shape != x.shape
        or not attn.is_contiguous()
    ):
        raise ValueError("Row pixel norm2 requires contiguous BF16 attention output")


@torch.library.custom_op("cirrus42::pixel_state_row_norm2", mutates_args=())
def norm2(
    x: torch.Tensor,
    attn: torch.Tensor,
    gate: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    validate(x, (gate, shift, scale))
    validate_attention(x, attn)
    out = torch.empty(x.shape, device=x.device, dtype=torch.bfloat16)
    pixel_state_layout.norm2_row_kernel[(3145728 // 8,)](
        x,
        attn,
        gate,
        shift,
        scale,
        out,
        num_warps=2,
        num_stages=1,
        enable_fp_fusion=True,
    )
    return out


@norm2.register_fake
def norm2_fake(x, attn, gate, shift, scale):
    validate(x, (gate, shift, scale))
    validate_attention(x, attn)
    return torch.empty(x.shape, device=x.device, dtype=torch.bfloat16)


def forward_for(row_input, row_output):
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
                "Pixel state layout requires released bilinear conditioning geometry"
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
            norm1(x, shift1, scale1)
            if row_input
            else pixel_norm_mod.norm1_modulate(x, shift1, scale1)
        )
        attn = self.attn(y, latent_dhw=pixel_dhw, rope_tables=rope_tables)
        y = (
            norm2(x, attn, gate1, shift2, scale2)
            if row_input
            else pixel_norm2_mod.norm2_modulate(x, attn, gate1, shift2, scale2)
        )
        mlp = self.mlp(y)
        result = (x + gate1 * attn) + gate2 * mlp
        if row_output:
            return result.contiguous()
        return result.transpose(1, 2).contiguous().transpose(1, 2)

    return forward


_FORWARDS = (
    forward_for(False, True),
    forward_for(True, True),
    forward_for(True, True),
    forward_for(True, False),
)


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
        or len(pixel_norm2_mod._ORIGINALS) != 4
    ):
        raise ValueError("State layout requires four validated eval norm2 pixel blocks")
    owned = [block for block, _, _ in pixel_norm2_mod._ORIGINALS]
    for block in strata.pixel_blocks:
        if (
            block not in owned
            or not isinstance(block.forward, MethodType)
            or block.forward.__func__ is not pixel_norm2_mod._forward
            or block.training
            or block.adaln_mode != "bilinear_dw"
            or block.num_adaln_params != 6
        ):
            raise ValueError("State layout requires the owned native norm2 composition")
        for norm in (block.norm1, block.norm2):
            if (
                not isinstance(norm, torch.nn.LayerNorm)
                or norm.normalized_shape != (128,)
                or norm.eps != 1e-6
                or norm.weight is not None
                or norm.bias is not None
            ):
                raise ValueError(
                    "State layout requires native nonaffine LayerNorm128 epsilon1e-6"
                )
    for block, forward in zip(strata.pixel_blocks, _FORWARDS):
        _ORIGINALS.append((block, "forward" in block.__dict__, block.forward))
        block.forward = MethodType(forward, block)
