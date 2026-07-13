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
"""Freeze the production models' parameter iteration order.

Optimizer state in pre-migration checkpoints maps onto parameters BY INDEX
(torch param groups are positional), so any change to ``named_parameters()``
order — a physicsnemo pin bump reordering a block's submodules, a new module
registered before ``self.strata`` in the wrappers — silently attaches Adam
moments to the wrong parameters when such a checkpoint resumes. Weight
loading would still succeed (keys remap by name), making the corruption
near-impossible to trace. These tests pin the order to committed manifests.

Regenerate the manifests ONLY alongside an intentional architecture change
that already invalidates old optimizer states::

    SCREAMCAST_REGEN_PARAM_ORDER=1 pytest tests/test_param_order.py

The models are built full-size on the meta device (no GPU, no weights) with
``grid_type="healpix"`` — the grid backend only affects geometry buffers,
never the parameter tree, and this avoids the cubesphere latlon ``.nc``
dependency (same trick as test_characterization_inference).
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

REF_DIR = Path(__file__).parent / "regression_data"
REGEN = os.environ.get("SCREAMCAST_REGEN_PARAM_ORDER") == "1"

PRODUCTION_CONFIGS = [
    "sweep1_nodilation_gated_tile64_kernel3_lr1em4_dim1024_hpatch4_depth32_r3_cos",
    "pixeldit_sem1024d24l_pix128d4l_bilineardwgeluproject_unfreeze_3src_const1em5_t128",
]


def _build_full_size(name: str) -> torch.nn.Module:
    """Build a production config at full size on the meta device."""
    import train as train_module
    from train_configs import CONFIGS

    cfg = CONFIGS[name]
    data = cfg.data
    in_channels = len(data.variables_prognostic) + len(data.variables_forcing)
    out_channels = len(data.variables_prognostic) + len(data.variables_diagnostic)
    depth_levels = len(np.r_[data.level_start : data.level_end : data.plevel])
    if cfg.dit.do_rotate_wind:
        wind = (
            data.variables_prognostic.index("U"),
            data.variables_prognostic.index("V"),
        )
    else:
        wind = None
    common = dict(
        in_channels=in_channels,
        out_channels=out_channels,
        nside=1024,
        tile_size=data.tile_size,
        dit_cfg=cfg.dit,
        do_bf16_mixed=cfg.compute.do_bf16_mixed,
        depth_levels=depth_levels,
        wind_channel_indices=wind,
        grid_type="healpix",  # grid backend never affects the parameter tree
        cubesphere_latlon_path=None,
    )
    with torch.device("meta"):
        if cfg.experiment.model_type == "pixeldit":
            return train_module.build_strata(pixel_cfg=cfg.pixel_dit, **common)
        return train_module.build_backbone(**common)


@pytest.mark.parametrize("name", PRODUCTION_CONFIGS)
def test_parameter_order_frozen(name: str) -> None:
    pytest.importorskip("train_configs", reason="needs the screamcast training env")
    model = _build_full_size(name)
    names = [n for n, _ in model.named_parameters()]
    assert len(names) > 0

    ref_path = REF_DIR / f"{name}.param_order.txt"
    if REGEN:
        REF_DIR.mkdir(parents=True, exist_ok=True)
        ref_path.write_text("\n".join(names) + "\n")
        return
    if not ref_path.exists():
        pytest.skip(
            f"no committed manifest {ref_path.name}; generate with "
            f"SCREAMCAST_REGEN_PARAM_ORDER=1 pytest {Path(__file__).name}"
        )
    ref = ref_path.read_text().splitlines()
    if names != ref:
        added = sorted(set(names) - set(ref))
        removed = sorted(set(ref) - set(names))
        first_move = next(
            (i for i, (a, b) in enumerate(zip(names, ref)) if a != b), None
        )
        raise AssertionError(
            f"{name}: named_parameters() order changed "
            f"(first divergence at index {first_move}, "
            f"{len(added)} added, {len(removed)} removed). Pre-migration "
            f"optimizer states map by index and would now attach to the "
            f"wrong parameters. If this reorder is intentional and old "
            f"optimizer states are being abandoned anyway, regenerate with "
            f"SCREAMCAST_REGEN_PARAM_ORDER=1."
        )


def test_manifest_committed() -> None:
    """Fail (not skip) when a manifest is missing in a full environment.

    The order guard is only as good as its committed baseline; this makes a
    deleted/forgotten manifest visible in CI instead of silently skipping.
    """
    missing = [
        f"{name}.param_order.txt"
        for name in PRODUCTION_CONFIGS
        if not (REF_DIR / f"{name}.param_order.txt").exists()
    ]
    if REGEN:
        pytest.skip("regenerating")
    assert not missing, (
        f"missing committed param-order manifests: {missing}; generate with "
        f"SCREAMCAST_REGEN_PARAM_ORDER=1 pytest tests/test_param_order.py "
        f"and commit the files"
    )
