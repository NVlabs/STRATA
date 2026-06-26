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
"""
Dataclasses grouping related parameters for screamcast models and pipelines.

These serve as the canonical parameter interface for training (train.py) and
rollout inference.
Experiments are defined as named entries in train_configs.py.

Usage::

    from train_configs import CONFIGS
    cfg = CONFIGS["my_experiment"]   # TrainConfig
    dit_cfg = cfg.dit                # DiTConfig
    training_cfg = cfg.training      # TrainingConfig
    data_cfg = cfg.data              # DataConfig
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import dotenv
import yaml

from screamcast.dali_ext_src import output_channels

dotenv.load_dotenv()

__all__ = [
    "output_channels",
    "DiTConfig",
    "PixelDiTConfig",
    "TrainingConfig",
    "PipelineConfig",
    "ComputeConfig",
    "DataConfig",
    "ExperimentConfig",
    "DebugConfig",
    "TrainConfig",
    "load_train_config_from_yaml",
    "configs_to_flat_dict",
]


def _from_dict(cls: type, d: dict[str, Any]) -> Any:
    """Construct a dataclass from a raw dict, ignoring unknown keys.

    Lists are coerced to tuples for fields typed as ``tuple[...]``.
    """
    valid = {f.name for f in dataclasses.fields(cls)}
    kwargs: dict[str, Any] = {k: v for k, v in d.items() if k in valid}
    for f in dataclasses.fields(cls):
        if (
            f.name in kwargs
            and "tuple" in str(f.type)
            and isinstance(kwargs[f.name], list)
        ):
            kwargs[f.name] = tuple(kwargs[f.name])
    return cls(**kwargs)


@dataclass
class DiTConfig:
    """Architecture parameters for the DiT3D model."""

    # Patch size
    patch_size: int = 1
    patch_size_vert: int = -1  # -1 → falls back to patch_size
    patch_size_horiz: int = -1  # -1 → falls back to patch_size

    # Core dimensions
    embed_dim: int = 768
    n_layers: int = 12
    num_heads: int = 16

    # QK normalization
    qk_norm: bool = False  # if True, will apply RMSNorm(elementwise_affine=qk_norm_elementwise_affine) to q and k before attention dot products
    qk_norm_elementwise_affine: bool = False

    # Attention (NA3D kernel, dilation, gating, depthwise, backend)
    attn_kernel: int | tuple[int, int, int] = 3
    do_interleaved_dilation: bool = False
    na_dilations: int = 3
    na3d_backend: str = ""  # "" → natten default. Options: "cutlass-fna", "hopper-fna", "blackwell-fna", "flex-fna"
    gated_attention: bool = False
    do_alt_depthwise_attn: bool = (
        False  # if True, will use depthwise attention every other DiTBlock
    )

    # Positional encoding
    do_rope_2d: bool = False
    do_rope_2d_stereographic: bool = False
    do_rotate_wind: bool = False
    index_is_latlon: bool = False

    # Dealiased patch embedding (shift-invariant PatchEmbed3D)
    use_dealiased_patch_embed: bool = False
    dealias_resample_filter: tuple[int, ...] = field(
        default_factory=lambda: (1, 4, 6, 4, 1)
    )  # Bin-5 default; good for 4× spatial stride

    # Activation checkpointing
    do_activation_checkpointing: float = 0.0  # 0.0=none, 1.0=all blocks

    def __post_init__(self) -> None:
        if not (0.0 <= self.do_activation_checkpointing <= 1.0):
            raise ValueError(
                f"do_activation_checkpointing must be in [0, 1], got {self.do_activation_checkpointing}"
            )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DiTConfig":
        return _from_dict(cls, d)


@dataclass
class PixelDiTConfig:
    """Architecture parameters for the PixelDiT pixel pathway (Stage 2)."""

    embed_dim: int = 128
    n_layers: int = 4
    num_heads: int | None = None  # None → auto (embed_dim // 64)
    mlp_ratio: float = 4.0
    attn_kernel: int = 7
    gated_attention: bool = False
    qk_norm: bool = False
    qk_norm_elementwise_affine: bool = False
    freeze_semantic: bool = False
    freeze_pixel_blocks: bool = False
    use_bilinear_dw_gelu_project_adaln: bool = False
    use_chunked_depthwise_conv: bool = True
    first_block_only_adaln: bool = False
    do_rope_2d: bool = False
    do_rope_2d_stereographic: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PixelDiTConfig":
        return _from_dict(cls, d)


@dataclass
class TrainingConfig:
    """Hyperparameters and settings for the training loop."""

    lr: float = 1e-4
    loss_lr: float = 0.1
    epochs: int = 1
    batch_size: int = 4
    num_workers: int = 2
    num_workers_inference: int = 1
    optimizer: str = "adam"
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 1e-5
    seed: int = 42
    scheduler_type: str = "constant"
    warmup_steps: int = 0
    cosine_t_max: int = 0
    cosine_n_cycles: float = 8
    cosine_decay: float = 0.8
    num_steps: int = 1  # 1=single-step, >1=multi-step rollout training
    multistep_training_mode: str = "final_only"
    do_tf32: bool = False
    grad_clip_max_norm: float = float("inf")
    backup_every_steps: int = 0
    loss_type: str = "smooth_l1"
    resume_from: str | None = None
    reset_best_valid_loss: bool = False
    reset_scheduler_state: bool = False
    skip_optimizer_reloading: bool = False
    fit_batches_for_norm: int = 20
    normalization_ref_path_input: str | None = None
    normalization_ref_path_target: str | None = None
    pooled_loss: str | None = None  # e.g. "4:0.2,16:0.1"
    pooled_loss_channel_weights_csv: str | None = None
    max_steps_per_epoch: int | None = None
    validate_steps: int | None = None
    validate_min_samples: int = 1000
    exit_after_first_validation: bool = False
    max_validations: int | None = None
    no_save: bool = False
    save_ckpt_every_epoch: bool = False
    use_time_variation_seed: bool = False
    do_backtest_inference: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainingConfig":
        return _from_dict(cls, d)


@dataclass
class PipelineConfig:
    """Prediction mode and output constraints shared between training and rollout.

    Grouping these together ensures that training and rollout always agree on
    what the model is predicting and how its outputs are post-processed.
    """

    target_type: str = "diff"  # "diff", "state", "mixed_state_diff", etc.
    do_qv_softplus: bool = False  # softplus non-negativity on qv tendency
    do_precip_relu: bool = False  # ReLU non-negativity on precip diagnostic outputs
    variables_input_zeroed: tuple[str, ...] = field(
        default_factory=tuple
    )  # zero these input channels post-normalization

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PipelineConfig":
        return _from_dict(cls, d)


@dataclass
class ComputeConfig:
    """Low-level compute settings shared between training and rollout.

    These affect how the model runs (precision, compilation) but not what it
    computes. Set once in the YAML and both scripts pick them up.
    """

    do_bf16_mixed: bool = False
    do_torch_compile: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ComputeConfig":
        return _from_dict(cls, d)


@dataclass
class DataConfig:
    """Dataset paths, grid configuration, variable lists, and tiling parameters."""

    # Grid
    grid_type: str = "cubesphere"  # "healpix" or "cubesphere"
    cubesphere_ne: int = 1024  # cubesphere resolution (ne1024pg2 → ne=1024)
    cubesphere_npg: int = 2  # nodes-per-gauss-point (ne1024pg2 → npg=2)
    tile_size: int = 64
    nside: int = 2048
    level_start: int = 8
    level_end: int = 32
    plevel: int = 1

    # Tiling
    cross_face_tiles: bool = False  # if True, will use cross-face tiles for training
    use_duo_padding: bool = (
        False  # if True, will use duo-padding for constructing cross-face tiles
    )
    duo_padding_scrip_src_path: str | None = None  # required when use_duo_padding=True
    duo_padding_scrip_tgt_path: str | None = None  # required when use_duo_padding=True
    skip_corner_tiles: bool = False
    balance_cross_face: bool = False

    # Variables — no defaults; must be set explicitly per experiment
    variables_prognostic: tuple[str, ...] = field(default_factory=tuple)
    variables_prognostic_state: tuple[str, ...] = field(default_factory=tuple)
    variables_forcing: tuple[str, ...] = field(default_factory=tuple)
    variables_diagnostic: tuple[str, ...] = field(default_factory=tuple)

    # Multi-source training zarr paths (overrides single .env source for training).
    # When set, uses per-source start/end indices instead of train_start_index/train_end_index.
    # The validation dataloader always reads from the single .env source regardless.
    train_main_zarr_paths: tuple[str, ...] | None = None
    train_aux_zarr_paths: tuple[str, ...] | None = None
    train_zarr_weights: tuple[float, ...] | None = None
    train_start_indices: tuple[int | None, ...] | None = None  # per-source
    train_end_indices: tuple[int | None, ...] | None = None  # per-source

    # Train/val split on the single .env source.
    # In multi-source mode, test_start_index/test_end_index still define the
    # split boundary used by the validation dataloader (which always reads .env).
    split_by_time: bool = False
    train_start_index: int = 0  # ignored for training when multi-source is active
    train_end_index: int | None = (
        None  # ignored for training when multi-source is active
    )
    test_start_index: int | None = None
    test_end_index: int | None = None
    test_stride: int = 1
    inference_start_index: int = 200  # starting timestep for EvalIndiaOceantile

    # Backtest (separate held-out zarr for evaluation; independent of train/test split)
    backtest_main_zarr_path: str | None = None
    backtest_aux_zarr_path: str | None = None
    backtest_start_index: int | None = None
    backtest_end_index: int | None = None
    backtest_stride: int | None = None

    # Static data paths
    latlon_path: str = "data/latlon_ne1024pg2.nc"

    def __post_init__(self) -> None:
        if self.grid_type and self.grid_type.lower() not in {"healpix", "cubesphere"}:
            raise ValueError(
                f"grid_type must be 'healpix' or 'cubesphere', got '{self.grid_type}'"
            )
        n = len(self.train_main_zarr_paths) if self.train_main_zarr_paths else 0
        if n:
            if self.train_aux_zarr_paths and len(self.train_aux_zarr_paths) != n:
                raise ValueError(
                    "train_aux_zarr_paths must match train_main_zarr_paths length"
                )
            if self.train_zarr_weights and len(self.train_zarr_weights) != n:
                raise ValueError(
                    "train_zarr_weights must match train_main_zarr_paths length"
                )
            if self.train_start_indices and len(self.train_start_indices) != n:
                raise ValueError(
                    "train_start_indices must match train_main_zarr_paths length"
                )
            if self.train_end_indices and len(self.train_end_indices) != n:
                raise ValueError(
                    "train_end_indices must match train_main_zarr_paths length"
                )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DataConfig":
        return _from_dict(cls, d)


@dataclass
class ExperimentConfig:
    """Top-level experiment metadata: name, output directory, and resume state."""

    name: str = ""
    rundir: str = "output"
    model_type: str = "dit3d"  # architecture selector; "dit3d" or "pixeldit"
    resume_from: str | None = None
    reset_best_valid_loss: bool = False
    reset_scheduler_state: bool = False
    skip_optimizer_reloading: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperimentConfig":
        return _from_dict(cls, d)


@dataclass
class DebugConfig:
    """Debug and testing flags shared across training and eval scripts."""

    print_steps: int = 100
    mock: bool = False
    skip_india_ocean_eval: bool = False
    use_full_test_set: bool = False
    recreate_dataloader_each_validation: bool = False
    wandb_disabled: bool = (
        False  # set True to skip wandb even when WANDB_API_KEY is set
    )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DebugConfig":
        return _from_dict(cls, d)


@dataclass
class TrainConfig:
    """Top-level config for training. Returned by ``load_train_config_from_yaml``.

    This object is embedded in checkpoints under the ``"train_config"`` key so
    that rollout scripts can reconstruct the model without re-specifying any
    architecture or data arguments.
    """

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    dit: DiTConfig = field(default_factory=DiTConfig)
    pixel_dit: PixelDiTConfig | None = (
        None  # required when experiment.model_type == "pixeldit"
    )
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainConfig":
        """Reconstruct from a nested dict (e.g. loaded from a checkpoint)."""
        pixel_dit_raw = d.get("pixel_dit")
        return cls(
            experiment=ExperimentConfig.from_dict(d.get("experiment", {})),
            dit=DiTConfig.from_dict(d.get("dit", {})),
            pixel_dit=PixelDiTConfig.from_dict(pixel_dit_raw)
            if pixel_dit_raw
            else None,
            pipeline=PipelineConfig.from_dict(d.get("pipeline", {})),
            compute=ComputeConfig.from_dict(d.get("compute", {})),
            training=TrainingConfig.from_dict(d.get("training", {})),
            data=DataConfig.from_dict(d.get("data", {})),
            debug=DebugConfig.from_dict(d.get("debug", {})),
        )


def load_train_config_from_yaml(path: str) -> TrainConfig:
    """Load training config from a YAML file. Missing sections fall back to defaults."""
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    return TrainConfig.from_dict(raw)


def configs_to_flat_dict(*cfgs: Any) -> dict[str, Any]:
    """Merge config dataclass objects into a single flat dict.

    Useful for ``wandb.init(config=configs_to_flat_dict(cfg.dit, cfg.training, ...))``.
    """
    out: dict[str, Any] = {}
    for cfg in cfgs:
        out.update(dataclasses.asdict(cfg))
    return out
