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
"""Check cached decoder setup without backbone metadata; Cirrus-42 (AI agent)."""

import pytest
import torch
from torch import nn

from screamcast import decoder_pixel_setup, gelu_table
from screamcast.forward_stage_benchmark import CachedConditioning


def decoder():
    model = nn.Module()
    model.pixel_blocks = nn.ModuleList()
    for _ in range(4):
        block = nn.Module()
        block.mlp = nn.Module()
        block.mlp.layers = nn.Sequential(nn.Linear(2, 2), nn.GELU(approximate="tanh"))
        model.pixel_blocks.append(block)
    model.eval()
    model.backbone = CachedConditioning(torch.arange(6.0).reshape(1, 3, 2), (1, 1, 3))
    return model


def test_cached_setup_keeps_features_and_parameter_identity(monkeypatch):
    model = decoder()
    original = model.backbone.conditioning.clone()
    cached = model.backbone
    ids = [id(p) for p in model.parameters()]
    names = tuple(model.state_dict())
    table = torch.zeros(65536, dtype=torch.bfloat16)
    monkeypatch.setattr(gelu_table, "prepare_table", lambda device: table)
    monkeypatch.setattr(
        decoder_pixel_setup, "install_spatial_reuse", lambda model: None
    )
    monkeypatch.setattr(
        decoder_pixel_setup.attention_layout, "install", lambda model: None
    )
    monkeypatch.setattr(
        decoder_pixel_setup.conditioning_norm, "install", lambda model: None
    )
    try:
        assert decoder_pixel_setup.apply_cached_baseline(
            model, "semantic-rope-packed-residual-layout"
        )
        assert model.backbone is cached
        torch.testing.assert_close(
            model.backbone.forward_tokens(None, None)[0], original
        )
        assert [id(p) for p in model.parameters()] == ids
        assert tuple(model.state_dict()) == names
        assert len(gelu_table._ORIGINALS) == 4
        assert all(
            block.mlp.layers[1].forward.__func__ is gelu_table._forward
            for block in model.pixel_blocks
        )
        assert decoder_pixel_setup.apply_cached_baseline(
            model, "cuda-fused-prepare-reuse"
        )
        assert not gelu_table._ORIGINALS
        assert all(
            "forward" not in block.mlp.layers[1].__dict__
            for block in model.pixel_blocks
        )
    finally:
        gelu_table.restore()


def test_full_backbone_or_legacy_case_uses_existing_setup():
    model = decoder()
    assert not decoder_pixel_setup.apply_cached_baseline(model, "baseline")
    model.backbone = nn.Identity()
    assert not decoder_pixel_setup.apply_cached_baseline(
        model, "semantic-rope-packed-residual-layout"
    )


def test_pixel_lookup_rejects_non_native_activation_before_patching(monkeypatch):
    model = decoder()
    model.pixel_blocks[-1].mlp.layers[1] = nn.GELU(approximate="none")
    monkeypatch.setattr(
        gelu_table,
        "prepare_table",
        lambda device: torch.zeros(65536, dtype=torch.bfloat16),
    )
    try:
        with pytest.raises(ValueError, match="tanh GELU"):
            gelu_table.install_pixel(model)
        assert not gelu_table._ORIGINALS
        assert all(
            "forward" not in block.mlp.layers[1].__dict__
            for block in model.pixel_blocks
        )
    finally:
        gelu_table.restore()
