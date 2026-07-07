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
"""Pure key-mapping tests for checkpoint_compat (CPU-only, no model build)."""

import pytest
import torch

from screamcast.checkpoint_compat import remap_legacy_state_dict

# Representative slice of the production pixeldit checkpoint's key inventory
# (semantic backbone with gated attention, bilinear_dw pixel blocks), plus the
# backbone-only (dit3d) key shapes. Values are placeholders — these tests only
# exercise key translation.
LEGACY_PIXEL_KEYS = [
    "semantic._patch_emb.proj.weight",
    "semantic._patch_emb.proj.bias",
    "semantic._blocks.0.attn.qkv.weight",
    "semantic._blocks.0.attn.qkv.bias",
    "semantic._blocks.0.attn.proj.weight",
    "semantic._blocks.0.attn.gated_attention_map.weight",
    "semantic._blocks.0.attn.gated_attention_map.bias",
    "semantic._blocks.0.mlp.fwd.0.weight",
    "semantic._blocks.0.mlp.fwd.0.bias",
    "semantic._blocks.0.mlp.fwd.3.weight",
    "semantic._blocks.23.mlp.fwd.3.bias",
    "_pixel_patch_emb.proj.weight",
    "_pixel_blocks.0.attn.qkv.weight",
    "_pixel_blocks.0.adaln_bilinear_dw_conv.weight",
    "_pixel_blocks.0.adaln_bilinear_dw_conv.bias",
    "_pixel_blocks.0.adaln_bilinear_dw_proj.weight",
    "_pixel_blocks.3.mlp.fwd.0.weight",
    "_pixel_blocks.3.mlp.fwd.3.weight",
    "_pixel_final_layer.linear.weight",
    "_pixel_final_layer.linear.bias",
]

EXPECTED_PIXEL_KEYS = [
    "strata.backbone.patch_embed.proj.weight",
    "strata.backbone.patch_embed.proj.bias",
    "strata.backbone.blocks.0.attn.qkv.weight",
    "strata.backbone.blocks.0.attn.qkv.bias",
    "strata.backbone.blocks.0.attn.proj.weight",
    "strata.backbone.blocks.0.attn.gate_proj.weight",
    "strata.backbone.blocks.0.attn.gate_proj.bias",
    "strata.backbone.blocks.0.mlp.layers.0.weight",
    "strata.backbone.blocks.0.mlp.layers.0.bias",
    "strata.backbone.blocks.0.mlp.layers.2.weight",
    "strata.backbone.blocks.23.mlp.layers.2.bias",
    "strata.pixel_patch_embed.proj.weight",
    "strata.pixel_blocks.0.attn.qkv.weight",
    "strata.pixel_blocks.0.adaln_bilinear_dw_conv.weight",
    "strata.pixel_blocks.0.adaln_bilinear_dw_conv.bias",
    "strata.pixel_blocks.0.adaln_bilinear_dw_proj.weight",
    "strata.pixel_blocks.3.mlp.layers.0.weight",
    "strata.pixel_blocks.3.mlp.layers.2.weight",
    "strata.pixel_final_layer.linear.weight",
    "strata.pixel_final_layer.linear.bias",
]

LEGACY_BACKBONE_KEYS = [
    "_patch_emb.proj.weight",
    "_blocks.0.attn.qkv.weight",
    "_blocks.0.mlp.fwd.0.weight",
    "_blocks.0.mlp.fwd.3.weight",
    "_final_layer.linear.weight",
    "_final_layer.linear.bias",
]

EXPECTED_BACKBONE_KEYS = [
    "strata.patch_embed.proj.weight",
    "strata.blocks.0.attn.qkv.weight",
    "strata.blocks.0.mlp.layers.0.weight",
    "strata.blocks.0.mlp.layers.2.weight",
    "strata.final_layer.linear.weight",
    "strata.final_layer.linear.bias",
]


def _sd(keys):
    return {k: torch.zeros(1) for k in keys}


def test_pixel_key_translation():
    new_sd, report = remap_legacy_state_dict(_sd(LEGACY_PIXEL_KEYS))
    assert report.was_legacy
    assert sorted(new_sd) == sorted(EXPECTED_PIXEL_KEYS)


def test_backbone_key_translation():
    new_sd, report = remap_legacy_state_dict(_sd(LEGACY_BACKBONE_KEYS))
    assert report.was_legacy
    assert sorted(new_sd) == sorted(EXPECTED_BACKBONE_KEYS)


def test_wrapper_prefixes_stripped_and_composed():
    # torch.compile-era production checkpoints prefix every key with _orig_mod.
    sd = _sd(["_orig_mod." + k for k in LEGACY_PIXEL_KEYS])
    new_sd, report = remap_legacy_state_dict(sd)
    assert report.was_legacy
    assert sorted(new_sd) == sorted(EXPECTED_PIXEL_KEYS)

    # Nested wrapping (DDP over compile) also normalizes.
    sd = _sd(["module._orig_mod." + LEGACY_PIXEL_KEYS[0]])
    new_sd, _ = remap_legacy_state_dict(sd)
    assert list(new_sd) == [EXPECTED_PIXEL_KEYS[0]]


def test_idempotent_on_new_format():
    new_sd, _ = remap_legacy_state_dict(_sd(LEGACY_PIXEL_KEYS))
    again, report = remap_legacy_state_dict(new_sd)
    assert not report.was_legacy
    assert report.passthrough == len(new_sd)
    assert sorted(again) == sorted(new_sd)


def test_mixed_legacy_and_new_rejected():
    sd = _sd([LEGACY_PIXEL_KEYS[0], EXPECTED_PIXEL_KEYS[1]])
    with pytest.raises(ValueError, match="mixes legacy"):
        remap_legacy_state_dict(sd)


def test_dead_heads_dropped_with_report():
    sd = _sd(LEGACY_BACKBONE_KEYS + ["_concat_skip_linear.weight"])
    new_sd, report = remap_legacy_state_dict(sd)
    assert report.dropped == ["_concat_skip_linear.weight"]
    assert "strata._concat_skip_linear.weight" not in new_sd


def test_target_model_coverage_mismatch_raises():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.strata = torch.nn.Linear(2, 2)

    sd = _sd(["semantic._blocks.0.attn.qkv.weight"])
    with pytest.raises(ValueError, match="does not match the target model"):
        remap_legacy_state_dict(sd, target_model=Tiny())


def test_lora_keys_skip_set_equality_but_translate():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.strata = torch.nn.Linear(2, 2)

    sd = _sd(
        [
            "semantic._blocks.0.attn.qkv.lora_A.weight",
            "semantic._blocks.0.attn.qkv.lora_B.weight",
        ]
    )
    new_sd, report = remap_legacy_state_dict(sd, target_model=Tiny())
    assert "strata.backbone.blocks.0.attn.qkv.lora_A.weight" in new_sd
    assert report.missing_in_target  # recorded, but not fatal for LoRA
