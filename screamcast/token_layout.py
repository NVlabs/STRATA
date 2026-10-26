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
"""Copy backbone tokens once before residual blocks; Cirrus-42 (AI agent)."""

import torch

_HANDLES = []


@torch.library.custom_op("cirrus42::contiguous_backbone_tokens", mutates_args=())
def copy_tokens(input: torch.Tensor) -> torch.Tensor:
    if input.shape != (8, 24576, 1024) or input.dtype != torch.float32:
        raise ValueError("Token layout trial requires fixed B8 FP32 backbone tokens")
    return input.clone(memory_format=torch.contiguous_format)


@copy_tokens.register_fake
def copy_tokens_fake(input):
    return torch.empty(input.shape, device=input.device, dtype=input.dtype)


def _before_first_block(module, args):
    return (copy_tokens(args[0]), *args[1:])


@torch.library.custom_op("cirrus42::channel_major_backbone_tokens", mutates_args=())
def restore_tokens(input: torch.Tensor) -> torch.Tensor:
    if input.shape != (8, 24576, 1024) or input.dtype != torch.float32:
        raise ValueError("Token layout trial requires fixed B8 FP32 backbone tokens")
    return (
        input.transpose(1, 2)
        .clone(memory_format=torch.contiguous_format)
        .transpose(1, 2)
    )


@restore_tokens.register_fake
def restore_tokens_fake(input):
    return torch.empty(
        (8, 1024, 24576), device=input.device, dtype=input.dtype
    ).transpose(1, 2)


def _after_last_block(module, args, output):
    return restore_tokens(output)


def restore():
    for handle in _HANDLES:
        handle.remove()
    _HANDLES.clear()


def install(strata):
    restore()
    if strata.training or len(strata.backbone.blocks) != 24:
        raise ValueError("Token layout trial requires eval with 24 backbone blocks")
    _HANDLES.append(
        strata.backbone.blocks[0].register_forward_pre_hook(_before_first_block)
    )
    _HANDLES.append(strata.backbone.blocks[-1].register_forward_hook(_after_last_block))
