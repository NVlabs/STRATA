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
"""Apply confirmed pixel features without a live backbone; Cirrus-42 (AI agent)."""

from screamcast import (
    attention_layout,
    column_attention,
    column_rms_attention,
    conditioning_norm,
    gelu_table,
    normalization_model,
    residual_boundary,
    semantic_qkv_pack,
    token_layout,
)
from screamcast.conditioning_spatial_reuse import install as install_spatial_reuse
from screamcast.forward_stage_benchmark import CachedConditioning

FULL_BASELINES = frozenset(
    {
        "semantic-persistent-gelu-all",
        "semantic-native-order-norm",
        "semantic-column-fused-output-gate",
        "semantic-column-rms-attention",
        "semantic-column-rms-residual-layout",
        "semantic-rope-packed",
        "semantic-rope-packed-residual-layout",
    }
)


def apply_cached_baseline(strata, name):
    """Keep captured features fixed; install only the parent configuration's decoder."""
    if not isinstance(strata.backbone, CachedConditioning):
        return False
    if name not in FULL_BASELINES and name != "cuda-fused-prepare-reuse":
        return False
    semantic_qkv_pack.restore()
    residual_boundary.restore()
    column_rms_attention.restore()
    normalization_model.restore()
    token_layout.restore()
    gelu_table.restore()
    column_attention.restore()
    attention_layout.restore()
    install_spatial_reuse(strata)
    if name in FULL_BASELINES:
        attention_layout.install(strata)
        conditioning_norm.install(strata)
        gelu_table.install_pixel(strata)
    return True
