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
"""Named training experiment configurations (CONFIGS dict).

Usage: python train.py <experiment_name>


Add new experiments to CONFIGS. Use dataclasses.replace() to branch off existing ones:

    CONFIGS["my_variant"] = dataclasses.replace(
        CONFIGS["some_base"],
        training=dataclasses.replace(CONFIGS["some_base"].training, lr=2e-4),
    )

Cluster-specific paths come from environment variables (set in .env):
  - PROJECT_ROOT    : root dir for experiment outputs
                      rundir:     $PROJECT_ROOT/<exp>/output
                      resume_from: $PROJECT_ROOT/<other_exp>/output/best.pth
  - AUX_DATA_ROOT   : directory for auxiliary data files (latlon .nc, z coords, etc.)
  - ZARR_ROOT       : root path for zarr datasets (S3 or lustre, cluster-dependent)

Zarr paths: for the standard single-source case, do NOT set DataConfig.train_main_zarr_paths
— dali_ext_src.py falls back to SCREAM_MAIN_ZARR_PATH/SCREAM_AUX_ZARR_PATH automatically.
Only set train_main_zarr_paths for multi-source training, e.g.:
    train_main_zarr_paths=(
        os.path.join(os.environ["ZARR_ROOT"], "dataset1.zarr"),
        os.path.join(os.environ["ZARR_ROOT"], "dataset2.zarr"),
    )
"""

import dataclasses
import os

from screamcast.config import (
    ComputeConfig,
    DataConfig,
    DebugConfig,
    DiTConfig,
    ExperimentConfig,
    PipelineConfig,
    PixelDiTConfig,
    TrainConfig,
    TrainingConfig,
)

CONFIGS: dict[str, TrainConfig] = {}

_ROOT = os.environ.get("PROJECT_ROOT", ".")
_AUX = os.environ.get("AUX_DATA_ROOT", "data")


def _ckpt(exp: str, ckpt_name: str = "best.pth") -> str:
    """Path to a checkpoint for a given experiment (defaults to best.pth)."""
    return os.path.join(_ROOT, exp, "output", ckpt_name)


def add_multistep_finetune_configs(
    base_exp: str,
    lr: float = 1e-5,
    max_validations: int = 5,
    validate_steps: int = 400,
    validate_min_samples: int = 1024,
    backup_every_steps: int = 100,
    ckpt_name: str = "best.pth",
    name_suffix: str = "",
) -> None:
    """Register 2-step and 4-step finetune configs for a base 1-step experiment.

    Adds CONFIGS[f"{base_exp}_2stepft{_suffix}"] and CONFIGS[f"{base_exp}_4stepft{_suffix}"]
    where _suffix is f"_{name_suffix}" if name_suffix is set, else "".

    Both use a constant LR schedule (lr=1e-5) and exit after max_validations validation rounds.

    Args:
        ckpt_name: checkpoint filename from base_exp to resume 2-step from (default "best.pth").
        name_suffix: appended to exp names to distinguish runs from different checkpoints,
            e.g. name_suffix="step1610k" -> "<base>_2stepft_step1610k".
    """
    base = CONFIGS[base_exp]
    _ft_training = dataclasses.replace(
        base.training,
        lr=lr,
        scheduler_type="constant",
        validate_steps=validate_steps,
        validate_min_samples=validate_min_samples,
        backup_every_steps=backup_every_steps,
        max_validations=max_validations,
    )
    _ft_debug = dataclasses.replace(
        base.debug,
        print_steps=5,
        skip_india_ocean_eval=True,
        use_full_test_set=True,
        recreate_dataloader_each_validation=True,
    )

    _suffix = f"_{name_suffix}" if name_suffix else ""
    name_2step = f"{base_exp}_2stepft{_suffix}"
    CONFIGS[name_2step] = dataclasses.replace(
        base,
        experiment=dataclasses.replace(
            base.experiment,
            name=name_2step,
            rundir=os.path.join(_ROOT, name_2step, "output"),
            resume_from=_ckpt(base_exp, ckpt_name),
            reset_scheduler_state=True,
            reset_best_valid_loss=True,
        ),
        training=dataclasses.replace(_ft_training, num_steps=2),
        debug=_ft_debug,
    )

    name_4step = f"{base_exp}_4stepft{_suffix}"
    CONFIGS[name_4step] = dataclasses.replace(
        CONFIGS[name_2step],
        experiment=dataclasses.replace(
            CONFIGS[name_2step].experiment,
            name=name_4step,
            rundir=os.path.join(_ROOT, name_4step, "output"),
            resume_from=_ckpt(name_2step),
        ),
        training=dataclasses.replace(_ft_training, num_steps=4),
    )


# ---------------------------------------------------------------------------
# example — small mock experiment for testing the pipeline
# ---------------------------------------------------------------------------
CONFIGS["example"] = TrainConfig(
    experiment=ExperimentConfig(
        name="example",
        rundir="output",
        model_type="dit3d",
    ),
    dit=DiTConfig(
        embed_dim=128,
        n_layers=2,
        num_heads=8,
        do_interleaved_dilation=True,
        do_alt_depthwise_attn=True,
        gated_attention=True,
        qk_norm=True,
        qk_norm_elementwise_affine=True,
        index_is_latlon=True,
    ),
    pipeline=PipelineConfig(
        target_type="mixed_state_diff",
        do_qv_softplus=True,
        do_precip_relu=True,
    ),
    compute=ComputeConfig(do_bf16_mixed=True),
    training=TrainingConfig(
        lr=5e-4,
        epochs=1,
        batch_size=1,
        optimizer="adamw",
        beta2=0.99,
        scheduler_type="dampened_cosine_with_hard_restarts",
        cosine_t_max=16,
        cosine_n_cycles=8,
        cosine_decay=0.8,
        warmup_steps=20,
        loss_type="smooth_l1",
        grad_clip_max_norm=0.05,
        validate_steps=200,
        validate_min_samples=20,
        max_steps_per_epoch=30000,
        num_workers=1,
        num_workers_inference=1,
    ),
    data=DataConfig(
        grid_type="cubesphere",
        tile_size=32,
        level_start=19,
        level_end=32,
        plevel=4,
        split_by_time=True,
        train_end_index=144,
        test_start_index=270,
        test_end_index=280,
        test_stride=2,
        variables_prognostic=("U", "omega", "qv", "T_2m"),
        variables_forcing=("coszr", "phis", "sgh30", "landfrac"),
        variables_diagnostic=("precip_ice_surf_mass_flux",),
    ),
    debug=DebugConfig(mock=True, skip_india_ocean_eval=False, wandb_disabled=False),
)

# ---------------------------------------------------------------------------
# example_pixeldit — two-stage PixelDiT variant of the mock example
# ---------------------------------------------------------------------------
CONFIGS["example_pixeldit"] = dataclasses.replace(
    CONFIGS["example"],
    experiment=dataclasses.replace(
        CONFIGS["example"].experiment,
        name="example_pixeldit",
        model_type="pixeldit",
    ),
    pixel_dit=PixelDiTConfig(
        embed_dim=64,
        n_layers=2,
        num_heads=4,
        attn_kernel=3,
        gated_attention=True,
        qk_norm=True,
    ),
    debug=dataclasses.replace(
        CONFIGS["example"].debug,
        wandb_disabled=True,
    ),
)

# ---------------------------------------------------------------------------
# pytest — fast mock config for test_training_launch.py
# ---------------------------------------------------------------------------
CONFIGS["pytest"] = dataclasses.replace(
    CONFIGS["example"],
    training=dataclasses.replace(
        CONFIGS["example"].training,
        max_steps_per_epoch=2,
        validate_min_samples=2,
        no_save=True,
    ),
    debug=dataclasses.replace(
        CONFIGS["example"].debug,
        wandb_disabled=True,
    ),
)

# ---------------------------------------------------------------------------
# example_diagomega — mock experiment for testing diagnostic-omega pipeline
# ---------------------------------------------------------------------------
CONFIGS["example_diagomega"] = dataclasses.replace(
    CONFIGS["example"],
    experiment=dataclasses.replace(
        CONFIGS["example"].experiment,
        name="example_diagomega",
    ),
    pipeline=dataclasses.replace(
        CONFIGS["example"].pipeline,
        variables_input_zeroed=("omega",),
    ),
    debug=dataclasses.replace(
        CONFIGS["example"].debug,
        wandb_disabled=True,
    ),
)

# ---------------------------------------------------------------------------
# pytest_diagomega — fast mock config for testing diagnostic-omega path
# ---------------------------------------------------------------------------
CONFIGS["pytest_diagomega"] = dataclasses.replace(
    CONFIGS["pytest"],
    experiment=dataclasses.replace(
        CONFIGS["pytest"].experiment,
        name="pytest_diagomega",
    ),
    pipeline=dataclasses.replace(
        CONFIGS["pytest"].pipeline,
        variables_input_zeroed=("omega",),
    ),
)


# ---------------------------------------------------------------------------
# sweep1 — DiT architecture sweep (cosine LR finetunes from pretrained r3 ckpts)
#
# All four experiments share the same data, pipeline, compute, and most
# training settings.  Each branches off its own pretrained checkpoint and
# varies only the DiT architecture and a few scheduling knobs.
# We reconstructed the configs below from original bash scripts.
#
# Variants:
#   kernel3_hpatch4_dim1024  — baseline: 1024d/32L/k3/hpatch4
#   kernel3_hpatch2_dim1024  — narrower horizontal patch (hpatch2, cosine_t_max=100)
#   kernel3_hpatch4_dim768   — lighter model: 768d/14L
#   kernel9_hpatch4_dim1024  — wider receptive field (k9), lower lr, branches from
#                              kernel3_hpatch4_dim1024 cos ckpt
# ---------------------------------------------------------------------------

# Shared base — use kernel3/hpatch4/dim1024 as the reference shape; each
# variant overrides only what differs.
_SWEEP1_BASE = TrainConfig(
    experiment=ExperimentConfig(
        name="",  # overridden per variant
        rundir="output",
        model_type="dit3d",
        reset_scheduler_state=True,
    ),
    dit=DiTConfig(
        embed_dim=1024,
        n_layers=32,
        num_heads=16,
        attn_kernel=3,
        patch_size_horiz=4,
        patch_size_vert=1,
        do_rope_2d=False,
        do_rope_2d_stereographic=True,
        do_rotate_wind=True,
        qk_norm=True,
        gated_attention=True,
        do_interleaved_dilation=False,
        index_is_latlon=True,
        do_alt_depthwise_attn=True,
    ),
    pipeline=PipelineConfig(target_type="mixed_state_diff"),
    compute=ComputeConfig(do_bf16_mixed=True, do_torch_compile=True),
    training=TrainingConfig(
        lr=1e-4,
        epochs=50,
        batch_size=1,
        loss_type="smooth_l1",
        optimizer="adamw",
        beta2=0.999,
        scheduler_type="dampened_cosine_with_hard_restarts",
        cosine_t_max=200,
        cosine_n_cycles=10,
        cosine_decay=0.5,
        do_tf32=True,
        num_workers=2,
        num_workers_inference=2,
        validate_steps=10000,
        backup_every_steps=2000,
        validate_min_samples=1024,
        fit_batches_for_norm=1280,
        save_ckpt_every_epoch=True,
        num_steps=1,
        multistep_training_mode="final_only",
        use_time_variation_seed=True,
        do_backtest_inference=False,
    ),
    data=DataConfig(
        grid_type="cubesphere",
        tile_size=64,
        level_start=8,
        level_end=32,
        plevel=1,
        cross_face_tiles=False,
        split_by_time=True,
        train_start_index=144,
        train_end_index=1435,
        test_start_index=1872,
        test_end_index=2016,
        test_stride=72,
        inference_start_index=1872,
        latlon_path=os.path.join(_AUX, "latlon_ne1024pg2.nc"),
        variables_prognostic=(
            "PotentialTemperature",
            "U",
            "V",
            "z_mid",
            "omega",
            "qv",
            "T_2m",
        ),
        variables_prognostic_state=("omega",),
        variables_forcing=("coszr", "phis", "sgh30", "landfrac"),
        variables_diagnostic=(
            "precip_ice_surf_mass_flux",
            "precip_liq_surf_mass_flux",
            "ps",
        ),
    ),
    debug=DebugConfig(
        use_full_test_set=True,
        recreate_dataloader_each_validation=True,
    ),
)

# kernel3 / hpatch4 / dim1024  — matches base exactly, just sets name + resume
_S1_PRE_K3_HP4_D1024 = (
    "sweep1_nodilation_gated_tile64_kernel3_lr1em4_dim1024_hpatch4_depth32_r3"
)
_S1_K3_HP4_D1024 = (
    "sweep1_nodilation_gated_tile64_kernel3_lr1em4_dim1024_hpatch4_depth32_r3_cos"
)
CONFIGS[_S1_K3_HP4_D1024] = dataclasses.replace(
    _SWEEP1_BASE,
    experiment=dataclasses.replace(
        _SWEEP1_BASE.experiment,
        name=_S1_K3_HP4_D1024,
        resume_from=_ckpt(_S1_PRE_K3_HP4_D1024, "latest.pth"),
    ),
)

# kernel3 / hpatch2 / dim1024  — narrower horizontal patch, shorter cosine cycle
_S1_PRE_K3_HP2_D1024 = (
    "sweep1_nodilation_gated_tile64_kernel3_lr1em4_dim1024_hpatch2_depth32_r3"
)
_S1_K3_HP2_D1024 = (
    "sweep1_nodilation_gated_tile64_kernel3_lr1em4_dim1024_hpatch2_depth32_r3_cos"
)
CONFIGS[_S1_K3_HP2_D1024] = dataclasses.replace(
    _SWEEP1_BASE,
    experiment=dataclasses.replace(
        _SWEEP1_BASE.experiment,
        name=_S1_K3_HP2_D1024,
        resume_from=_ckpt(_S1_PRE_K3_HP2_D1024, "latest.pth"),
    ),
    dit=dataclasses.replace(
        _SWEEP1_BASE.dit,
        patch_size_horiz=2,
    ),
    training=dataclasses.replace(
        _SWEEP1_BASE.training,
        cosine_t_max=100,
    ),
)

# kernel3 / hpatch4 / dim768  — lighter 768d/14L model
_S1_PRE_K3_HP4_D768 = (
    "sweep1_nodilation_gated_tile64_kernel3_lr1em4_dim768_hpatch4_depth14_r3"
)
_S1_K3_HP4_D768 = (
    "sweep1_nodilation_gated_tile64_kernel3_lr1em4_dim768_hpatch4_depth14_r3_cos"
)
CONFIGS[_S1_K3_HP4_D768] = dataclasses.replace(
    _SWEEP1_BASE,
    experiment=dataclasses.replace(
        _SWEEP1_BASE.experiment,
        name=_S1_K3_HP4_D768,
        resume_from=_ckpt(_S1_PRE_K3_HP4_D768, "latest.pth"),
    ),
    dit=dataclasses.replace(
        _SWEEP1_BASE.dit,
        embed_dim=768,
        n_layers=14,
        num_heads=12,
    ),
    training=dataclasses.replace(
        _SWEEP1_BASE.training,
        validate_steps=40000,
        backup_every_steps=8000,
        num_workers=4,
        cosine_decay=0.8,
    ),
)

# ---------------------------------------------------------------------------
# pixeldit_sem1024d24l_pix128d4l
#   Semantic stage: DiT 1024d / 24L / patch_horiz=4 / kernel=9 / stereographic RoPE
#   Pixel stage:    128d / 4L / kernel=9 / qk_norm
# ---------------------------------------------------------------------------
_EXP_PIXELDIT_PROD = "pixeldit_sem1024d24l_pix128d4l"
_NORM_REF = _ckpt("sweep1_get_normalization_24levels", "latest.pth")

CONFIGS[_EXP_PIXELDIT_PROD] = TrainConfig(
    experiment=ExperimentConfig(
        name=_EXP_PIXELDIT_PROD,
        rundir="output",
        model_type="pixeldit",
    ),
    dit=DiTConfig(
        embed_dim=1024,
        n_layers=24,
        num_heads=16,
        attn_kernel=9,
        patch_size_horiz=4,
        patch_size_vert=1,
        do_rope_2d=False,
        do_rope_2d_stereographic=True,
        do_rotate_wind=True,
        qk_norm=True,
        gated_attention=True,
        do_alt_depthwise_attn=True,
        do_interleaved_dilation=False,
        index_is_latlon=True,
    ),
    pixel_dit=PixelDiTConfig(
        embed_dim=128,
        n_layers=4,
        attn_kernel=9,
        qk_norm=True,
    ),
    pipeline=PipelineConfig(target_type="mixed_state_diff"),
    compute=ComputeConfig(do_bf16_mixed=True, do_torch_compile=True),
    training=TrainingConfig(
        lr=5e-4,
        epochs=50,
        batch_size=1,
        optimizer="adamw",
        beta2=0.999,
        loss_type="smooth_l1",
        scheduler_type="dampened_cosine_with_hard_restarts",
        warmup_steps=2000,
        cosine_t_max=100,
        cosine_n_cycles=5,
        cosine_decay=0.3162,
        do_tf32=True,
        num_workers=2,
        num_workers_inference=2,
        num_steps=1,
        multistep_training_mode="final_only",
        validate_steps=20000,
        backup_every_steps=2000,
        validate_min_samples=1024,
        fit_batches_for_norm=1280,
        save_ckpt_every_epoch=True,
        use_time_variation_seed=True,
        do_backtest_inference=False,
        normalization_ref_path_input=_NORM_REF,
        normalization_ref_path_target=_NORM_REF,
    ),
    data=DataConfig(
        grid_type="cubesphere",
        tile_size=64,
        level_start=8,
        level_end=32,
        plevel=1,
        cross_face_tiles=False,
        split_by_time=True,
        train_start_index=144,
        train_end_index=1435,
        test_start_index=1872,
        test_end_index=2016,
        test_stride=48,
        inference_start_index=1872,
        latlon_path=os.path.join(_AUX, "latlon_ne1024pg2.nc"),
        variables_prognostic=(
            "PotentialTemperature",
            "U",
            "V",
            "z_mid",
            "omega",
            "qv",
            "T_2m",
        ),
        variables_prognostic_state=("omega",),
        variables_forcing=("coszr", "phis", "sgh30", "landfrac"),
        variables_diagnostic=(
            "precip_ice_surf_mass_flux",
            "precip_liq_surf_mass_flux",
            "ps",
        ),
    ),
    debug=DebugConfig(
        use_full_test_set=True,
        recreate_dataloader_each_validation=False,
        wandb_disabled=False,
    ),
)

# ---------------------------------------------------------------------------
# pixeldit_sem1024d24l_pix128d4l_3src
#   Resumes from pixeldit_sem1024d24l_pix128d4l with 3-source zarr training data
#   (sdecadal + sdy2 + sdy1), warmup→constant LR at 1e-4, backtest inference on sdy2.
# ---------------------------------------------------------------------------
_EXP_PIXELDIT_PROD = "pixeldit_sem1024d24l_pix128d4l"
_EXP_PIXELDIT_3SRC = "pixeldit_sem1024d24l_pix128d4l_3src"
_ZARR_ROOT = os.environ.get("ZARR_ROOT", "s3://SCREAM_zarrv3")
_ZARR_SDECADAL = os.path.join(
    _ZARR_ROOT,
    "sdecadal.ne1024pg2_ne1024pg2.F20TR-SCREAMv1.c10-sep11.out10min.cubesphere.zarr",
)
_ZARR_SDY2 = os.path.join(
    _ZARR_ROOT,
    "sdy2.ne1024pg2_ne1024pg2.F2010-SCREAMv1.c10-sep11-f602da2b98.out10min-4lev.cubesphere.zarr",
)
_ZARR_SDY1 = os.path.join(
    _ZARR_ROOT,
    "sdy1.ne1024pg2_ne1024pg2.F2010-SCREAMv1.c10-sep11-f602da2b98.out10min-4lev.cubesphere.zarr",
)

CONFIGS[_EXP_PIXELDIT_3SRC] = dataclasses.replace(
    CONFIGS[_EXP_PIXELDIT_PROD],
    experiment=dataclasses.replace(
        CONFIGS[_EXP_PIXELDIT_PROD].experiment,
        name=_EXP_PIXELDIT_3SRC,
        resume_from=_ckpt(_EXP_PIXELDIT_PROD),
        reset_scheduler_state=True,
        reset_best_valid_loss=True,
    ),
    training=dataclasses.replace(
        CONFIGS[_EXP_PIXELDIT_PROD].training,
        lr=1e-4,
        scheduler_type="warmup_constant",
        warmup_steps=10000,
        num_workers=2,
        num_workers_inference=2,
        do_backtest_inference=True,
        validate_steps=20000,
    ),
    data=dataclasses.replace(
        CONFIGS[_EXP_PIXELDIT_PROD].data,
        split_by_time=True,
        train_start_index=144,
        train_end_index=1435,
        train_main_zarr_paths=(_ZARR_SDECADAL, _ZARR_SDY2, _ZARR_SDY1),
        train_aux_zarr_paths=(_ZARR_SDECADAL, _ZARR_SDY2, _ZARR_SDY1),
        train_start_indices=(144, 144, 144),
        train_end_indices=(1435, 720, 720),
        train_zarr_weights=(0.5, 1.0, 1.0),
        test_start_index=1872,
        test_end_index=2015,
        test_stride=36,
        inference_start_index=1872,
        backtest_main_zarr_path=_ZARR_SDY2,
        backtest_aux_zarr_path=_ZARR_SDY2,
        backtest_start_index=864,
        backtest_end_index=1007,
        backtest_stride=36,
    ),
    debug=dataclasses.replace(
        CONFIGS[_EXP_PIXELDIT_PROD].debug,
        skip_india_ocean_eval=True,
        recreate_dataloader_each_validation=False,
    ),
)

_EXP_PIXELDIT_3SRC_COS = "pixeldit_sem1024d24l_pix128d4l_3src_lr5e5cos"
CONFIGS[_EXP_PIXELDIT_3SRC_COS] = dataclasses.replace(
    CONFIGS[_EXP_PIXELDIT_3SRC],
    experiment=dataclasses.replace(
        CONFIGS[_EXP_PIXELDIT_3SRC].experiment,
        name=_EXP_PIXELDIT_3SRC_COS,
    ),
    training=dataclasses.replace(
        CONFIGS[_EXP_PIXELDIT_3SRC].training,
        lr=5e-5,
        scheduler_type="dampened_cosine_with_hard_restarts",
        cosine_t_max=100,
        cosine_n_cycles=5,
        cosine_decay=0.3162,
        grad_clip_max_norm=0.5,
    ),
    debug=dataclasses.replace(
        CONFIGS[_EXP_PIXELDIT_3SRC].debug,
        recreate_dataloader_each_validation=True,
    ),
)

_EXP_PIXELDIT_3SRC_COS_QVFIX = "pixeldit_sem1024d24l_pix128d4l_3src_lr5e5cos_qvfix"
CONFIGS[_EXP_PIXELDIT_3SRC_COS_QVFIX] = dataclasses.replace(
    CONFIGS[_EXP_PIXELDIT_3SRC_COS],
    experiment=dataclasses.replace(
        CONFIGS[_EXP_PIXELDIT_3SRC_COS].experiment,
        name=_EXP_PIXELDIT_3SRC_COS_QVFIX,
        resume_from=_ckpt(_EXP_PIXELDIT_3SRC_COS),
        reset_scheduler_state=True,
        reset_best_valid_loss=True,
    ),
    pipeline=dataclasses.replace(
        CONFIGS[_EXP_PIXELDIT_3SRC_COS].pipeline,
        do_qv_softplus=True,
        do_precip_relu=True,
    ),
    training=dataclasses.replace(
        CONFIGS[_EXP_PIXELDIT_3SRC_COS].training,
        lr=5e-6,
        scheduler_type="warmup_constant",
        grad_clip_max_norm=0.5,
    ),
    debug=dataclasses.replace(
        CONFIGS[_EXP_PIXELDIT_3SRC_COS].debug,
        recreate_dataloader_each_validation=True,
    ),
)

_EXP_PIXELDIT_3SRC_COS_QVFIX_WSPINUP = (
    "pixeldit_sem1024d24l_pix128d4l_3src_lr5e5cos_qvfix_wspinup"
)
CONFIGS[_EXP_PIXELDIT_3SRC_COS_QVFIX_WSPINUP] = dataclasses.replace(
    CONFIGS[_EXP_PIXELDIT_3SRC_COS_QVFIX],
    experiment=dataclasses.replace(
        CONFIGS[_EXP_PIXELDIT_3SRC_COS_QVFIX].experiment,
        name=_EXP_PIXELDIT_3SRC_COS_QVFIX_WSPINUP,
    ),
    training=dataclasses.replace(
        CONFIGS[_EXP_PIXELDIT_3SRC_COS_QVFIX].training,
        lr=1e-5,
        scheduler_type="warmup_constant",
        batch_size=2,
        validate_steps=10000,
        grad_clip_max_norm=0.5,
    ),
    data=dataclasses.replace(
        CONFIGS[_EXP_PIXELDIT_3SRC_COS_QVFIX].data,
        train_start_index=2,
        train_end_index=1435,
        train_main_zarr_paths=(_ZARR_SDECADAL, _ZARR_SDY2, _ZARR_SDY1),
        train_aux_zarr_paths=(_ZARR_SDECADAL, _ZARR_SDY2, _ZARR_SDY1),
        train_start_indices=(2, 2, 144),
        train_end_indices=(1435, 720, 720),
        train_zarr_weights=(1.0, 1.0, 1.0),
        test_start_index=1872,
        test_end_index=2015,
        test_stride=36,
        inference_start_index=1872,
        backtest_main_zarr_path=_ZARR_SDY1,
        backtest_aux_zarr_path=_ZARR_SDY1,
        backtest_start_index=2,
        backtest_end_index=144,
        backtest_stride=36,
    ),
)

# ---------------------------------------------------------------------------
# pixeldit_sem1024d24l_pix128d4l_bilineardwgeluproject
#   bilinear upsample → DWConv2d(1×5×5, replicate, identity-init) → GeLU → Linear at pixel res.
#   Combines smooth boundary conv with full-resolution GeLU projection.
# ---------------------------------------------------------------------------
_EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT = (
    "pixeldit_sem1024d24l_pix128d4l_bilineardwgeluproject"
)
CONFIGS[_EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT] = dataclasses.replace(
    CONFIGS[_EXP_PIXELDIT_PROD],
    experiment=dataclasses.replace(
        CONFIGS[_EXP_PIXELDIT_PROD].experiment,
        name=_EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT,
    ),
    pipeline=dataclasses.replace(
        CONFIGS[_EXP_PIXELDIT_PROD].pipeline,
        do_qv_softplus=True,
        do_precip_relu=True,
    ),
    training=dataclasses.replace(
        CONFIGS[_EXP_PIXELDIT_PROD].training,
        grad_clip_max_norm=0.5,
    ),
    pixel_dit=dataclasses.replace(
        CONFIGS[_EXP_PIXELDIT_PROD].pixel_dit,
        use_bilinear_dw_gelu_project_adaln=True,
    ),
)

_EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_FREEZESEM = (
    "pixeldit_sem1024d24l_pix128d4l_bilineardwgeluproject_freezesem"
)
CONFIGS[_EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_FREEZESEM] = dataclasses.replace(
    CONFIGS[_EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT],
    experiment=dataclasses.replace(
        CONFIGS[_EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT].experiment,
        name=_EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_FREEZESEM,
        resume_from=_ckpt(_EXP_PIXELDIT_3SRC_COS_QVFIX_WSPINUP),
        skip_optimizer_reloading=True,
        reset_best_valid_loss=True,
        reset_scheduler_state=True,
    ),
    pixel_dit=dataclasses.replace(
        CONFIGS[_EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT].pixel_dit,
        freeze_semantic=True,
    ),
)

_EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_UNFREEZE_3SRC = (
    "pixeldit_sem1024d24l_pix128d4l_bilineardwgeluproject_unfreeze_3src"
)
CONFIGS[_EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_UNFREEZE_3SRC] = dataclasses.replace(
    CONFIGS[_EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_FREEZESEM],
    experiment=dataclasses.replace(
        CONFIGS[_EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_FREEZESEM].experiment,
        name=_EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_UNFREEZE_3SRC,
        resume_from=_ckpt(_EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_FREEZESEM),
        skip_optimizer_reloading=False,
        reset_best_valid_loss=True,
        reset_scheduler_state=True,
    ),
    data=CONFIGS[_EXP_PIXELDIT_3SRC].data,
    training=dataclasses.replace(
        CONFIGS[_EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_FREEZESEM].training,
        lr=5e-5,
        scheduler_type="dampened_cosine_with_hard_restarts",
        cosine_t_max=100,
        cosine_n_cycles=5,
        cosine_decay=0.3162,
        grad_clip_max_norm=0.5,
    ),
    debug=dataclasses.replace(
        CONFIGS[_EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_FREEZESEM].debug,
        recreate_dataloader_each_validation=True,
    ),
    pixel_dit=dataclasses.replace(
        CONFIGS[_EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_FREEZESEM].pixel_dit,
        freeze_semantic=False,
    ),
)

_EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_UNFREEZE_3SRC_CONST = (
    "pixeldit_sem1024d24l_pix128d4l_bilineardwgeluproject_unfreeze_3src_const1em5"
)
CONFIGS[
    _EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_UNFREEZE_3SRC_CONST
] = dataclasses.replace(
    CONFIGS[_EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_UNFREEZE_3SRC],
    experiment=dataclasses.replace(
        CONFIGS[_EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_UNFREEZE_3SRC].experiment,
        name=_EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_UNFREEZE_3SRC_CONST,
        resume_from=_ckpt(_EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_UNFREEZE_3SRC),
        reset_scheduler_state=True,
        reset_best_valid_loss=True,
    ),
    training=dataclasses.replace(
        CONFIGS[_EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_UNFREEZE_3SRC].training,
        lr=1e-5,
        scheduler_type="constant",
    ),
)

_EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_UNFREEZE_3SRC_CONST_T128 = (
    "pixeldit_sem1024d24l_pix128d4l_bilineardwgeluproject_unfreeze_3src_const1em5_t128"
)
CONFIGS[
    _EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_UNFREEZE_3SRC_CONST_T128
] = dataclasses.replace(
    CONFIGS[_EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_UNFREEZE_3SRC_CONST],
    experiment=dataclasses.replace(
        CONFIGS[_EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_UNFREEZE_3SRC_CONST].experiment,
        name=_EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_UNFREEZE_3SRC_CONST_T128,
        resume_from=_ckpt(_EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_UNFREEZE_3SRC_CONST),
        reset_best_valid_loss=True,
        reset_scheduler_state=True,
        skip_optimizer_reloading=True,
    ),
    training=dataclasses.replace(
        CONFIGS[_EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_UNFREEZE_3SRC_CONST].training,
        lr=1e-5,
        scheduler_type="warmup_constant",
        validate_steps=5000,
        backup_every_steps=500,
    ),
    data=dataclasses.replace(
        CONFIGS[_EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_UNFREEZE_3SRC_CONST].data,
        tile_size=128,
    ),
    dit=dataclasses.replace(
        CONFIGS[_EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_UNFREEZE_3SRC_CONST].dit,
        do_activation_checkpointing=1.0,
    ),
)

# ---------------------------------------------------------------------------
# ..._const1em5_t128_dealias
#   Resume from the T128 production checkpoint and only change one thing:
#   enable the dealiased patch embedding with the default (1,4,6,4,1) filter.
#   Keeps T128's warmup_constant lr=1e-5 schedule. Per the dealiased-embed
#   pattern elsewhere in this file, skip_optimizer_reloading=True because the
#   proj weights transfer exactly (only the stride/padding scheme changes).
# ---------------------------------------------------------------------------
_EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_UNFREEZE_3SRC_CONST_T128_DEALIAS = "pixeldit_sem1024d24l_pix128d4l_bilineardwgeluproject_unfreeze_3src_const1em5_t128_dealias"
CONFIGS[
    _EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_UNFREEZE_3SRC_CONST_T128_DEALIAS
] = dataclasses.replace(
    CONFIGS[_EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_UNFREEZE_3SRC_CONST_T128],
    experiment=dataclasses.replace(
        CONFIGS[
            _EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_UNFREEZE_3SRC_CONST_T128
        ].experiment,
        name=_EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_UNFREEZE_3SRC_CONST_T128_DEALIAS,
        resume_from=_ckpt(
            _EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_UNFREEZE_3SRC_CONST_T128
        ),
        reset_best_valid_loss=True,
        reset_scheduler_state=True,
        skip_optimizer_reloading=True,
    ),
    dit=dataclasses.replace(
        CONFIGS[_EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_UNFREEZE_3SRC_CONST_T128].dit,
        use_dealiased_patch_embed=True,
        dealias_resample_filter=(1, 4, 6, 4, 1),
    ),
    pixel_dit=dataclasses.replace(
        CONFIGS[
            _EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_UNFREEZE_3SRC_CONST_T128
        ].pixel_dit,
        use_chunked_depthwise_conv=False,
    ),
)

# ---------------------------------------------------------------------------
# ..._const1em5_t128_dealias_r2
#   Continuation of the _t128_dealias run under a new experiment name.
#   Architecture, data, and schedule are unchanged; resumes from the latest
#   _t128_dealias checkpoint. reset_best_valid_loss and reset_scheduler_state
#   are inherited as True from the parent _t128_dealias config.
# ---------------------------------------------------------------------------
_EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_UNFREEZE_3SRC_CONST_T128_DEALIAS_R2 = (
    _EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_UNFREEZE_3SRC_CONST_T128_DEALIAS + "_r2"
)
CONFIGS[
    _EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_UNFREEZE_3SRC_CONST_T128_DEALIAS_R2
] = dataclasses.replace(
    CONFIGS[_EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_UNFREEZE_3SRC_CONST_T128_DEALIAS],
    experiment=dataclasses.replace(
        CONFIGS[
            _EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_UNFREEZE_3SRC_CONST_T128_DEALIAS
        ].experiment,
        name=_EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_UNFREEZE_3SRC_CONST_T128_DEALIAS_R2,
        resume_from=_ckpt(
            _EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_UNFREEZE_3SRC_CONST_T128_DEALIAS
        ),
    ),
)

# ---------------------------------------------------------------------------
# ..._bilineardwgeluproject_3src_t64_pixstereorope_const1em4
#   From-scratch variant of the T128 production layout: same dit/pixel_dit
#   architecture, same 3-source zarr training data, but trained with
#   tile_size=64 and stereographic 2-D RoPE turned on in the pixel pathway
#   (do_rope_2d_stereographic on PixelDiTConfig). resume_from=None and the
#   reset_* flags are cleared. Normalization refs (_NORM_REF) are inherited
#   unchanged from PROD via the T128 base, so the model uses the precomputed
#   norm files rather than refitting.
#   Training: warmup_constant, lr=1e-4, warmup_steps=5000.
# ---------------------------------------------------------------------------
_EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_3SRC_T64_PIXSTEREO_CONST1EM4 = "pixeldit_sem1024d24l_pix128d4l_bilineardwgeluproject_3src_t64_pixstereorope_const1em4"
CONFIGS[
    _EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_3SRC_T64_PIXSTEREO_CONST1EM4
] = dataclasses.replace(
    CONFIGS[_EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_UNFREEZE_3SRC_CONST_T128],
    experiment=dataclasses.replace(
        CONFIGS[
            _EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_UNFREEZE_3SRC_CONST_T128
        ].experiment,
        name=_EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_3SRC_T64_PIXSTEREO_CONST1EM4,
        resume_from=None,
        reset_best_valid_loss=False,
        reset_scheduler_state=False,
        skip_optimizer_reloading=False,
    ),
    data=dataclasses.replace(
        CONFIGS[_EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_UNFREEZE_3SRC_CONST_T128].data,
        tile_size=64,
        train_zarr_weights=(1.0, 1.0, 1.0),
        test_stride=48,
    ),
    training=dataclasses.replace(
        CONFIGS[
            _EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_UNFREEZE_3SRC_CONST_T128
        ].training,
        lr=1e-4,
        scheduler_type="warmup_constant",
        warmup_steps=5000,
        validate_steps=20000,
    ),
    pixel_dit=dataclasses.replace(
        CONFIGS[
            _EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_UNFREEZE_3SRC_CONST_T128
        ].pixel_dit,
        do_rope_2d_stereographic=True,
        use_chunked_depthwise_conv=False,
    ),
    dit=dataclasses.replace(
        CONFIGS[_EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_UNFREEZE_3SRC_CONST_T128].dit,
        do_activation_checkpointing=0.0,
    ),
)

# ---------------------------------------------------------------------------
# ..._bilineardwgeluproject_3src_t64_axialrope_nowindrot_const1em4
#   From-scratch variant that uses row/column (axial) 2-D RoPE on BOTH the
#   semantic DiT blocks and the pixel blocks, with wind rotation disabled
#   (since axial RoPE is grid-local and not tied to the stereographic tangent
#   plane that the wind rotation aligns to).
#     - dit.do_rope_2d=True, dit.do_rope_2d_stereographic=False
#     - dit.do_rotate_wind=False
#     - pixel_dit.do_rope_2d=True
#   Same 3-source zarr data, tile_size=64, warmup_constant lr=1e-4 warmup=5000.
#   Norm refs are inherited from PROD via the T128 base.
# ---------------------------------------------------------------------------
_EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_3SRC_T64_AXIALROPE_NOWINDROT_CONST1EM4 = "pixeldit_sem1024d24l_pix128d4l_bilineardwgeluproject_3src_t64_axialrope_nowindrot_const1em4"
CONFIGS[
    _EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_3SRC_T64_AXIALROPE_NOWINDROT_CONST1EM4
] = dataclasses.replace(
    CONFIGS[_EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_UNFREEZE_3SRC_CONST_T128],
    experiment=dataclasses.replace(
        CONFIGS[
            _EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_UNFREEZE_3SRC_CONST_T128
        ].experiment,
        name=_EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_3SRC_T64_AXIALROPE_NOWINDROT_CONST1EM4,
        resume_from=None,
        reset_best_valid_loss=False,
        reset_scheduler_state=False,
        skip_optimizer_reloading=False,
    ),
    data=dataclasses.replace(
        CONFIGS[_EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_UNFREEZE_3SRC_CONST_T128].data,
        tile_size=64,
        train_zarr_weights=(1.0, 1.0, 1.0),
        test_stride=48,
    ),
    training=dataclasses.replace(
        CONFIGS[
            _EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_UNFREEZE_3SRC_CONST_T128
        ].training,
        lr=1e-4,
        scheduler_type="warmup_constant",
        warmup_steps=5000,
        validate_steps=20000,
    ),
    dit=dataclasses.replace(
        CONFIGS[_EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_UNFREEZE_3SRC_CONST_T128].dit,
        do_rope_2d=True,
        do_rope_2d_stereographic=False,
        do_rotate_wind=False,
        do_activation_checkpointing=0.0,
    ),
    pixel_dit=dataclasses.replace(
        CONFIGS[
            _EXP_PIXELDIT_BILINEAR_DW_GELU_PROJECT_UNFREEZE_3SRC_CONST_T128
        ].pixel_dit,
        do_rope_2d=True,
        use_chunked_depthwise_conv=False,
    ),
)

# ---------------------------------------------------------------------------
# pixeldit_sem768d12l_pix64d2l
#   Lighter variant: DiT 768d / 12L / patch_horiz=4 / kernel=9 / stereographic RoPE
#   Pixel stage:     64d / 2L / kernel=9 / qk_norm
# ---------------------------------------------------------------------------
_EXP_PIXELDIT_LIGHT = "pixeldit_sem768d12l_pix64d2l"

# ---------------------------------------------------------------------------
# pixeldit_sem1024d24l_pix128d4l_diagomega
#   Same as pixeldit_sem1024d24l_pix128d4l but omega input is zeroed (diagnostic omega).
#   Finetuned from the pretrained prod model.
# ---------------------------------------------------------------------------
_EXP_PIXELDIT_PROD_DIAGOMEGA = "pixeldit_sem1024d24l_pix128d4l_diagomega"

CONFIGS[_EXP_PIXELDIT_PROD_DIAGOMEGA] = dataclasses.replace(
    CONFIGS[_EXP_PIXELDIT_PROD],
    experiment=dataclasses.replace(
        CONFIGS[_EXP_PIXELDIT_PROD].experiment,
        name=_EXP_PIXELDIT_PROD_DIAGOMEGA,
        resume_from=_ckpt(_EXP_PIXELDIT_PROD),
        reset_scheduler_state=True,
        reset_best_valid_loss=True,
    ),
    pipeline=dataclasses.replace(
        CONFIGS[_EXP_PIXELDIT_PROD].pipeline,
        variables_input_zeroed=("omega",),
    ),
)

CONFIGS[_EXP_PIXELDIT_LIGHT] = dataclasses.replace(
    CONFIGS[_EXP_PIXELDIT_PROD],
    experiment=dataclasses.replace(
        CONFIGS[_EXP_PIXELDIT_PROD].experiment,
        name=_EXP_PIXELDIT_LIGHT,
    ),
    dit=dataclasses.replace(
        CONFIGS[_EXP_PIXELDIT_PROD].dit,
        embed_dim=768,
        n_layers=12,
        num_heads=12,
    ),
    pixel_dit=dataclasses.replace(
        CONFIGS[_EXP_PIXELDIT_PROD].pixel_dit,
        embed_dim=64,
        n_layers=2,
    ),
    training=dataclasses.replace(
        CONFIGS[_EXP_PIXELDIT_PROD].training,
        cosine_t_max=200,
        cosine_n_cycles=4,
        num_workers=6,
    ),
)

# ---------------------------------------------------------------------------
# fixed_cube — old pre-config-system experiments (tile32, kernel5, dim1024,
# interleaved dilation, 8 heads).  Minimal TrainConfig entries so that the
# rollout eval script can reconstruct the model from old checkpoints that
# don't embed a train_config dict.
#
# Two variants share the same architecture; only training schedule differs:
#   _lr5em4_4stepft_paddedloader  — cosine lr=5e-4, single-source
#   _lr5em5cos_3source_duopad_4stepft — cosine lr=5e-5, 3-source
# ---------------------------------------------------------------------------
_FIXED_CUBE_BASE = TrainConfig(
    experiment=ExperimentConfig(
        name="",
        rundir="output",
        model_type="dit3d",
    ),
    dit=DiTConfig(
        embed_dim=1024,
        n_layers=32,
        num_heads=8,
        attn_kernel=5,
        do_rope_2d=False,
        do_rope_2d_stereographic=True,
        do_rotate_wind=True,
        qk_norm=True,
        gated_attention=True,
        do_interleaved_dilation=True,
        do_alt_depthwise_attn=True,
        index_is_latlon=True,
    ),
    pipeline=PipelineConfig(target_type="mixed_state_diff"),
    compute=ComputeConfig(do_bf16_mixed=True, do_torch_compile=True),
    data=DataConfig(
        grid_type="cubesphere",
        tile_size=32,
        level_start=13,
        level_end=32,
        plevel=1,
        latlon_path=os.path.join(_AUX, "latlon_ne1024pg2.nc"),
        variables_prognostic=(
            "PotentialTemperature",
            "U",
            "V",
            "z_mid",
            "omega",
            "qv",
            "T_2m",
        ),
        variables_prognostic_state=("omega",),
        variables_forcing=("coszr", "phis", "sgh30", "landfrac"),
        variables_diagnostic=(
            "precip_ice_surf_mass_flux",
            "precip_liq_surf_mass_flux",
            "ps",
        ),
    ),
)

_FC_DUOPAD = "fixed_cube_dit3d_base_control_gated_tile32_32layers_dilated_kernel5_dim1024_lr5em4_4stepft_paddedloader"
CONFIGS[_FC_DUOPAD] = dataclasses.replace(
    _FIXED_CUBE_BASE,
    experiment=dataclasses.replace(_FIXED_CUBE_BASE.experiment, name=_FC_DUOPAD),
)

_FC_3SRC = "fixed_cube_dit3d_base_control_gated_tile32_32layers_dilated_kernel5_dim1024_lr5em5cos_3source_duopad_4stepft"
CONFIGS[_FC_3SRC] = dataclasses.replace(
    _FIXED_CUBE_BASE,
    experiment=dataclasses.replace(_FIXED_CUBE_BASE.experiment, name=_FC_3SRC),
)

# ---------------------------------------------------------------------------
# sweep1 dim768/14L warmup-then-constant — sweep over scheduler on lighter model
#
# Architecture: 768d / 14L / 12 heads / kernel3 / hpatch4 (same as _S1_K3_HP4_D768
# but trained from scratch with warmup_constant instead of cosine).
# Baseline lr=1e-4 with 20k warmup steps; constant thereafter.
# ---------------------------------------------------------------------------
_S1_K3_HP4_D768_WARMUP20K = "sweep1_k3_hp4_d768_14l_warmup20k"
CONFIGS[_S1_K3_HP4_D768_WARMUP20K] = dataclasses.replace(
    _SWEEP1_BASE,
    experiment=dataclasses.replace(
        _SWEEP1_BASE.experiment,
        name=_S1_K3_HP4_D768_WARMUP20K,
    ),
    dit=dataclasses.replace(
        _SWEEP1_BASE.dit,
        embed_dim=768,
        n_layers=14,
        num_heads=12,
    ),
    training=dataclasses.replace(
        _SWEEP1_BASE.training,
        lr=1e-4,
        scheduler_type="warmup_constant",
        warmup_steps=20000,
        num_workers=2,  # dim=768
    ),
)

for _lr_tag, _lr_val in (("5em4", 5e-4), ("1em3", 1e-3), ("2em3", 2e-3)):
    _name = f"sweep1_k3_hp4_d768_14l_warmup20k_lr{_lr_tag}"
    CONFIGS[_name] = dataclasses.replace(
        CONFIGS[_S1_K3_HP4_D768_WARMUP20K],
        experiment=dataclasses.replace(
            CONFIGS[_S1_K3_HP4_D768_WARMUP20K].experiment,
            name=_name,
        ),
        training=dataclasses.replace(
            CONFIGS[_S1_K3_HP4_D768_WARMUP20K].training,
            lr=_lr_val,
        ),
    )

# ---------------------------------------------------------------------------
# sweep1 batch-size scaling sweep — linear LR scaling rule
#
#   Base: 768d/14L/k3/hp4, warmup_constant.
#   Linear scaling: 2× global batch → 2× LR (from 5e-4 @ gbs=8).
#
#   All runs on 1 node (8 GPUs). gbs≥32 use activation checkpointing + higher per-GPU bs.
#
#     gbs  = bs × 8 GPU | LR      | act. ckpt
#       8  = 1  × 8     | 5e-4    | no
#      16  = 2  × 8     | 1e-3    | no
#      32  = 4  × 8     | 2e-3    | yes (1.0)
#      64  = 8  × 8     | 4e-3    | yes (1.0)
#     128  = 16 × 8     | 8e-3    | yes (1.0)
#     256  = 32 × 8     | 1.6e-2  | yes (1.0)  # may OOM
# ---------------------------------------------------------------------------
_BSWEEP_BASE_NAME = "sweep1_k3_hp4_d768_14l_warmup20k_lr5em4"
_BSWEEP_NORM_REF = _ckpt("sweep1_get_normalization_24levels", "latest.pth")
_BSWEEP_TRAINING = dataclasses.replace(
    CONFIGS[_BSWEEP_BASE_NAME].training,
    normalization_ref_path_input=_BSWEEP_NORM_REF,
    normalization_ref_path_target=_BSWEEP_NORM_REF,
)
_BSWEEP_DIT = CONFIGS[_BSWEEP_BASE_NAME].dit
_BSWEEP_DIT_CKPT = dataclasses.replace(_BSWEEP_DIT, do_activation_checkpointing=1.0)

#                     suffix,   bs, lr,    act_ckpt
_BSWEEP_VARIANTS = (
    ("gbs8", 1, 5e-4, False),
    ("gbs16", 2, 1e-3, False),
    ("gbs32", 4, 2e-3, True),
    ("gbs64", 8, 4e-3, True),  # linear scaling (unstable)
    ("gbs64_lr2em3", 8, 2e-3, True),  # same lr as gbs32
    ("gbs64_lr1em3", 8, 1e-3, True),  # same lr as gbs16
    ("gbs64_lr5em4", 8, 5e-4, True),  # same lr as gbs8
    ("gbs128", 16, 8e-3, True),
    ("gbs256", 32, 1.6e-2, True),
)

for _suffix, _bs, _lr, _act_ckpt in _BSWEEP_VARIANTS:
    _name = f"sweep1_k3_hp4_d768_bsweep_{_suffix}"
    CONFIGS[_name] = dataclasses.replace(
        CONFIGS[_BSWEEP_BASE_NAME],
        experiment=dataclasses.replace(
            CONFIGS[_BSWEEP_BASE_NAME].experiment,
            name=_name,
        ),
        dit=_BSWEEP_DIT_CKPT if _act_ckpt else _BSWEEP_DIT,
        training=dataclasses.replace(
            _BSWEEP_TRAINING,
            batch_size=_bs,
            lr=_lr,
        ),
    )

# ---------------------------------------------------------------------------
# Multi-step finetune variants: 2-step and 4-step for both pixeldit models
# ---------------------------------------------------------------------------
# define expname_2stepft and expname_4stepft for the following experiments
add_multistep_finetune_configs("pixeldit_sem1024d24l_pix128d4l")
add_multistep_finetune_configs("pixeldit_sem768d12l_pix64d2l")
for _sfx in ("_2stepft", "_4stepft"):
    _k = f"pixeldit_sem768d12l_pix64d2l{_sfx}"
    CONFIGS[_k] = dataclasses.replace(
        CONFIGS[_k],
        training=dataclasses.replace(CONFIGS[_k].training, num_workers=2),
    )

add_multistep_finetune_configs(
    "pixeldit_sem1024d24l_pix128d4l_bilineardwgeluproject_unfreeze_3src_const1em5_t128",
    ckpt_name="best.pth",
)
