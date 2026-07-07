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
"""Checkpoint key remapping: legacy DiT/DiT_Pixel names -> Strata wrapper names.

Checkpoints trained before the physicsnemo Strata migration use the legacy
module tree (``semantic._blocks...``, ``_pixel_blocks...``, ``mlp.fwd...``).
The wrapper classes in ``strata_wrappers.py`` hold the physicsnemo model under
``self.strata``, whose module names follow physicsnemo
(``strata.backbone.blocks...``, ``strata.pixel_blocks...``,
``mlp.layers...``). :func:`remap_legacy_state_dict` translates old keys to new
ones deterministically; it is a no-op on already-migrated state dicts, so it
is safe to apply unconditionally at every network checkpoint load.

NOTE on optimizer state: this module remaps NETWORK weights only. Optimizer
state saved before the migration is index-based and does not survive the
module reordering — resuming an old run must reinitialize the optimizer.
"""

import logging
import re
from dataclasses import dataclass, field

import torch

logger = logging.getLogger(__name__)

# Wrapper prefixes injected by torch.compile / DDP / lightning; stripped
# repeatedly so nested wrapping (e.g. compile-over-DDP) also normalizes.
_WRAPPER_PREFIXES = ("_orig_mod.", "module.", "_forward_module.")

# A state dict is "legacy" iff any (wrapper-stripped) key starts with one of
# these. New-format keys all start with "strata.".
_LEGACY_HEADS = (
    "semantic.",
    "_pixel_patch_emb.",
    "_pixel_blocks.",
    "_pixel_final_layer.",
    "_patch_emb.",
    "_blocks.",
    "_final_layer.",
    "_concat_skip_linear.",
)

# Top-level tree renames: first match wins, then the search stops. The pixel
# model rules (semantic./ _pixel_*) come first; the bare _patch_emb/_blocks/
# _final_layer heads occur only in backbone-only (DiT3D) checkpoints.
_PREFIX_RULES = (
    (re.compile(r"^semantic\._patch_emb\."), "backbone.patch_embed."),
    (re.compile(r"^semantic\._blocks\."), "backbone.blocks."),
    (re.compile(r"^semantic\."), "backbone."),
    (re.compile(r"^_pixel_patch_emb\."), "pixel_patch_embed."),
    (re.compile(r"^_pixel_blocks\."), "pixel_blocks."),
    (re.compile(r"^_pixel_final_layer\."), "pixel_final_layer."),
    (re.compile(r"^_patch_emb\."), "patch_embed."),
    (re.compile(r"^_blocks\."), "blocks."),
    (re.compile(r"^_final_layer\."), "final_layer."),
)

# Intra-module renames (substring, may appear at any depth).
_SUBSTRING_RULES = (
    (".attn.gated_attention_map.", ".attn.gate_proj."),
    # Legacy MLP: fwd = Sequential(fc1, act, drop, fc2, drop) -> indices 0/3.
    # physicsnemo Mlp (drop=0): layers = Sequential(fc1, act, fc2) -> 0/2.
    (".mlp.fwd.0.", ".mlp.layers.0."),
    (".mlp.fwd.3.", ".mlp.layers.2."),
)

# Legacy keys with no counterpart in the Strata wrappers: dropped with a
# warning. The concat-skip head was never used by any shipped config; the
# pixel model's backbone is headless (include_head=False), so a semantic
# final layer — absent from checkpoints saved after it was deleted at
# construction — has nowhere to load.
_DROP_HEADS = ("_concat_skip_linear.", "semantic._final_layer.")


@dataclass
class RemapReport:
    """What :func:`remap_legacy_state_dict` did to each key."""

    was_legacy: bool = False
    renamed: dict[str, str] = field(default_factory=dict)
    dropped: list[str] = field(default_factory=list)
    passthrough: int = 0
    missing_in_target: list[str] = field(default_factory=list)
    unexpected_in_target: list[str] = field(default_factory=list)


def _strip_wrapper_prefixes(key: str) -> str:
    stripped = True
    while stripped:
        stripped = False
        for prefix in _WRAPPER_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                stripped = True
    return key


def _remap_one(key: str) -> str | None:
    """Translate one wrapper-stripped legacy key. None -> drop."""
    for head in _DROP_HEADS:
        if key.startswith(head):
            return None
    for pattern, repl in _PREFIX_RULES:
        key, n_subs = pattern.subn(repl, key)
        if n_subs:
            break
    for old, new in _SUBSTRING_RULES:
        key = key.replace(old, new)
    return "strata." + key


def remap_legacy_state_dict(
    state_dict: dict[str, torch.Tensor],
    target_model: torch.nn.Module | None = None,
) -> tuple[dict[str, torch.Tensor], RemapReport]:
    """Translate a legacy network state dict to Strata-wrapper key names.

    Always strips torch.compile / DDP wrapper prefixes first, so callers do
    not need their own ``_orig_mod.`` handling. Already-migrated (or
    fresh-model) state dicts pass through unchanged apart from that strip,
    making the function idempotent and safe to call on every load.

    Args:
        state_dict: the ``ckpt["network"]`` mapping (keys -> tensors).
        target_model: optional wrapper instance to validate coverage against.
            Keys containing LoRA adapters (``.lora_A.`` / ``.lora_B.``) skip
            the exact set-equality check (adapters are attached to the model
            after this remap), but shape checks still run on matching keys.

    Returns:
        (new_state_dict, report). Raises ValueError on mixed legacy/new keys
        or (when ``target_model`` is given, non-LoRA) coverage mismatches.
    """
    report = RemapReport()

    stripped = {_strip_wrapper_prefixes(k): v for k, v in state_dict.items()}
    if len(stripped) != len(state_dict):
        raise ValueError(
            "Duplicate keys after stripping wrapper prefixes — refusing to remap"
        )

    legacy_keys = [k for k in stripped if k.startswith(_LEGACY_HEADS)]
    report.was_legacy = bool(legacy_keys)

    if report.was_legacy and len(legacy_keys) != len(stripped):
        offenders = sorted(set(stripped) - set(legacy_keys))[:5]
        raise ValueError(
            "State dict mixes legacy and non-legacy keys; cannot remap safely. "
            f"Non-legacy examples: {offenders}"
        )

    if not report.was_legacy:
        report.passthrough = len(stripped)
        new_sd = stripped
    else:
        new_sd = {}
        for key, value in stripped.items():
            new_key = _remap_one(key)
            if new_key is None:
                report.dropped.append(key)
                logger.warning("checkpoint remap: dropping legacy key %s", key)
                continue
            if new_key in new_sd:
                raise ValueError(f"Remap collision: {key} -> {new_key}")
            report.renamed[key] = new_key
            new_sd[new_key] = value

    if target_model is not None:
        target_sd = target_model.state_dict()
        has_lora = any(".lora_A." in k or ".lora_B." in k for k in new_sd)
        report.missing_in_target = sorted(set(new_sd) - set(target_sd))
        report.unexpected_in_target = sorted(set(target_sd) - set(new_sd))
        if not has_lora and (report.missing_in_target or report.unexpected_in_target):
            raise ValueError(
                "Remapped state dict does not match the target model.\n"
                f"  keys not in model (first 5): {report.missing_in_target[:5]}\n"
                f"  model keys not in ckpt (first 5): {report.unexpected_in_target[:5]}"
            )
        for key in new_sd.keys() & target_sd.keys():
            if new_sd[key].shape != target_sd[key].shape:
                raise ValueError(
                    f"Shape mismatch for {key}: checkpoint "
                    f"{tuple(new_sd[key].shape)} vs model {tuple(target_sd[key].shape)}"
                )

    if report.was_legacy:
        logger.info(
            "checkpoint remap: translated %d legacy keys (%d dropped)",
            len(report.renamed),
            len(report.dropped),
        )
    return new_sd, report
