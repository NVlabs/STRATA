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
import torch


def low_pass(x: torch.Tensor, factor: int) -> torch.Tensor:
    """Coarsen by ``factor`` via avg_pool then interpolate back."""
    if factor <= 0:
        raise ValueError(f"factor must be > 0, got {factor}")
    bsz, channels, faces, height, width = x.shape
    if height % factor != 0 or width % factor != 0:
        raise ValueError(
            f"Spatial shape ({height}, {width}) must be divisible by factor={factor}"
        )
    x2d = x.reshape(-1, 1, height, width)
    coarse = torch.nn.functional.avg_pool2d(x2d, kernel_size=factor, stride=factor)
    smooth = torch.nn.functional.interpolate(
        coarse, size=(height, width), mode="bicubic", align_corners=False
    )
    return smooth.reshape(bsz, channels, faces, height, width)
