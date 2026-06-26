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
"""Shared ACE fine-tuning utilities used by the rev2 workflow."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn as nn
from fme.ace.models.modulus.sfnonet import SphericalFourierNeuralOperatorNet
from fme.ace.stepper.single_module import load_stepper

from screamcast.ace._channels import ACE_VARIABLE_NAMES

ACE_STACK_TO_FME: dict[str, str] = {
    **{f"t{k}k": f"air_temperature_{k}" for k in range(8)},
    **{f"qtot{k}k": f"specific_total_water_{k}" for k in range(8)},
    **{f"u{k}k": f"eastward_wind_{k}" for k in range(8)},
    **{f"v{k}k": f"northward_wind_{k}" for k in range(8)},
    "sp": "PRESsfc",
    "skt": "surface_temperature",
    "u10m": "UGRD10m",
    "v10m": "VGRD10m",
    "t2m": "TMP2m",
    "q2m": "Q2m",
}

TINY_RESIDUAL_SCALE_FLOOR = 1e-6


@dataclass(frozen=True)
class ACEInputLayout:
    stack_names: list[str]
    fme_in_names: list[str]
    forcing_fme_names: list[str]

    def stack_to_fme_indices(self) -> list[tuple[int, int]]:
        fme_index = {name: i for i, name in enumerate(self.fme_in_names)}
        pairs: list[tuple[int, int]] = []
        for src_idx, stack_name in enumerate(self.stack_names):
            fme_name = ACE_STACK_TO_FME[stack_name]
            pairs.append((src_idx, fme_index[fme_name]))
        return pairs

    def forcing_to_fme_indices(self) -> list[tuple[int, int]]:
        fme_index = {name: i for i, name in enumerate(self.fme_in_names)}
        return [
            (src_idx, fme_index[name])
            for src_idx, name in enumerate(self.forcing_fme_names)
        ]


def reorder_forcing_tensor(
    forcing: torch.Tensor,
    forcing_names: list[str],
    expected_names: list[str],
) -> torch.Tensor:
    forcing_index = {name: i for i, name in enumerate(forcing_names)}
    missing = [name for name in expected_names if name not in forcing_index]
    if missing:
        raise ValueError(f"Stored ACE forcing tensor is missing channels: {missing}")
    return forcing[:, [forcing_index[name] for name in expected_names]]


def load_ace_backbone(
    checkpoint: str,
) -> tuple[nn.Module, ACEInputLayout, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    stepper = load_stepper(checkpoint)
    wrapped_step = stepper.config.step.config["wrapped_step"]["config"]
    input_layout = ACEInputLayout(
        stack_names=list(ACE_VARIABLE_NAMES),
        fme_in_names=list(wrapped_step["in_names"]),
        forcing_fme_names=list(stepper._input_only_names),
    )
    backbone = stepper.modules[0].module
    return backbone, input_layout, stepper.normalizer.means, stepper.normalizer.stds


def load_ace_out_names(checkpoint: str) -> list[str]:
    stepper = load_stepper(checkpoint)
    return list(stepper.out_names)


def _clone_tensor_dict(values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in values.items()}


def _serialize_input_layout(input_layout: ACEInputLayout) -> dict[str, list[str]]:
    return {
        "stack_names": list(input_layout.stack_names),
        "fme_in_names": list(input_layout.fme_in_names),
        "forcing_fme_names": list(input_layout.forcing_fme_names),
    }


def _deserialize_input_layout(payload: dict[str, object]) -> ACEInputLayout:
    return ACEInputLayout(
        stack_names=list(payload["stack_names"]),
        fme_in_names=list(payload["fme_in_names"]),
        forcing_fme_names=list(payload["forcing_fme_names"]),
    )


def _activation_name(backbone: nn.Module) -> str:
    activation = getattr(backbone, "activation_function", None)
    if activation is nn.GELU:
        return "gelu"
    if activation is nn.ReLU:
        return "relu"
    if activation is nn.SiLU:
        return "silu"
    raise ValueError(f"Unsupported backbone activation function {activation!r}")


def _serialize_backbone_spec(backbone: nn.Module) -> dict[str, object]:
    if not isinstance(backbone, SphericalFourierNeuralOperatorNet):
        raise TypeError(
            "Only SphericalFourierNeuralOperatorNet backbones are supported, got "
            f"{type(backbone).__name__}"
        )
    return {
        "format_version": 1,
        "class_name": type(backbone).__name__,
        "constructor": {
            "spectral_transform": backbone.spectral_transform,
            "filter_type": backbone.filter_type,
            "operator_type": backbone.operator_type,
            "img_shape": list(backbone.img_shape),
            "scale_factor": backbone.scale_factor,
            "residual_filter_factor": backbone.residual_filter_factor,
            "in_chans": backbone.in_chans,
            "out_chans": backbone.out_chans,
            "embed_dim": backbone.embed_dim,
            "num_layers": backbone.num_layers,
            "use_mlp": backbone.use_mlp,
            "activation_function": _activation_name(backbone),
            "encoder_layers": backbone.encoder_layers,
            "pos_embed": isinstance(backbone.pos_embed, nn.Parameter),
            "num_blocks": backbone.num_blocks,
            "sparsity_threshold": 0.0,
            "normalization_layer": backbone.normalization_layer,
            "hard_thresholding_fraction": backbone.hard_thresholding_fraction,
            "use_complex_kernels": True,
            "big_skip": backbone.big_skip,
            "rank": backbone.rank,
            "factorization": backbone.factorization,
            "separable": backbone.separable,
            "complex_network": backbone.complex_network,
            "complex_activation": backbone.complex_activation,
            "spectral_layers": backbone.spectral_layers,
            "checkpointing": backbone.checkpointing,
        },
    }


def _build_backbone_from_spec(payload: dict[str, object]) -> nn.Module:
    class_name = payload["class_name"]
    if class_name != "SphericalFourierNeuralOperatorNet":
        raise ValueError(f"Unsupported embedded backbone class {class_name!r}")
    constructor = dict(payload["constructor"])
    constructor["img_shape"] = tuple(constructor["img_shape"])
    return SphericalFourierNeuralOperatorNet(params=SimpleNamespace(), **constructor)


def build_embedded_ace_payload(
    backbone: nn.Module,
    input_layout: ACEInputLayout,
    normalizer_means: dict[str, torch.Tensor],
    normalizer_stds: dict[str, torch.Tensor],
    ace_out_names: list[str],
) -> dict[str, object]:
    return {
        "format_version": 2,
        "backbone": _serialize_backbone_spec(backbone),
        "input_layout": _serialize_input_layout(input_layout),
        "normalizer_means": _clone_tensor_dict(normalizer_means),
        "normalizer_stds": _clone_tensor_dict(normalizer_stds),
        "ace_out_names": list(ace_out_names),
    }


def build_coords_payload(
    *,
    lat: torch.Tensor,
    lon: torch.Tensor,
    hyam_sub: torch.Tensor,
    hybm_sub: torch.Tensor,
) -> dict[str, object]:
    return {
        "format_version": 1,
        "lat": lat.detach().cpu().clone(),
        "lon": lon.detach().cpu().clone(),
        "hyam_sub": hyam_sub.detach().cpu().clone(),
        "hybm_sub": hybm_sub.detach().cpu().clone(),
    }


def load_ace_artifacts_from_finetune_checkpoint(
    checkpoint_payload: dict[str, object],
) -> tuple[
    nn.Module,
    ACEInputLayout,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    list[str],
]:
    embedded = checkpoint_payload.get("embedded_ace")
    if embedded is None:
        raise ValueError("Checkpoint is missing required embedded_ace metadata.")
    if not isinstance(embedded, dict):
        raise TypeError(
            "checkpoint['embedded_ace'] must be a dict, got "
            f"{type(embedded).__name__}"
        )
    required = {
        "backbone",
        "input_layout",
        "normalizer_means",
        "normalizer_stds",
        "ace_out_names",
    }
    missing = sorted(required.difference(embedded))
    if missing:
        raise ValueError(f"Embedded ACE payload is missing fields: {missing}")
    backbone = _build_backbone_from_spec(embedded["backbone"])
    input_layout = _deserialize_input_layout(embedded["input_layout"])
    normalizer_means = _clone_tensor_dict(embedded["normalizer_means"])
    normalizer_stds = _clone_tensor_dict(embedded["normalizer_stds"])
    ace_out_names = list(embedded["ace_out_names"])
    return backbone, input_layout, normalizer_means, normalizer_stds, ace_out_names


def atomic_torch_save(payload: dict[str, object], output_path: str) -> None:
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".tmp_ace2scream_ckpt_", suffix=".pt", dir=output_dir
    )
    os.close(fd)
    try:
        torch.save(payload, tmp_path)
        with open(tmp_path, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp_path, output_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
