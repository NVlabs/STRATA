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
import dataclasses
import enum
import gc
import logging
import os
import random
import time

import earth2grid
import numpy as np
import nvtx
import pandas as pd
import torch
import torch.distributed as dist
import torchmetrics
import typer
import xarray as xr
from lightning.fabric import Fabric
from lightning.fabric.plugins import environments
from pytorch_optimizer import StableAdamW
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb as wandb_lib
except ImportError:
    wandb_lib = None

from screamcast import inference, model_pipelines
from screamcast.config import DiTConfig, PixelDiTConfig
from screamcast.constructors import (
    get_dampened_cosine_with_hard_restarts_schedule_with_warmup,
    get_warmup_constant_schedule,
)
from screamcast.dali_ext_src import ScreamV2
from screamcast.normalization import RunningNorm2d
from screamcast.pipelines import get_dataloader
from screamcast.strata_wrappers import (
    StrataBackboneModel,
    StrataModel,
)


def print_network_info(model):
    total_params = sum(p.numel() for p in model.parameters())
    print("Number of params", total_params)


# Initialize distributed training
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


class TargetOptions(str, enum.Enum):
    diff = "diff"
    diff_scaled_by_diff = "diff_scaled_by_diff"
    state = "state"
    mixed_state_diff = "mixed_state_diff"


class LossOptions(str, enum.Enum):
    mse = "mse"
    smooth_l1 = "smooth_l1"


def resolve_loss(opt: LossOptions) -> torch.nn.Module:
    if opt == LossOptions.mse:
        return torch.nn.MSELoss()
    elif opt == LossOptions.smooth_l1:
        return torch.nn.SmoothL1Loss()


def _dit_common_kwargs(
    in_channels: int,
    out_channels: int,
    nside: int,
    tile_size: int,
    dit_cfg: DiTConfig,
    do_bf16_mixed: bool,
    depth_levels: int,
    wind_channel_indices: tuple[int, int] | None = None,
    grid_type: str = "healpix",
    cubesphere_latlon_path: str | None = None,
) -> dict:
    """Return kwargs dict shared by build_backbone and build_strata."""
    return dict(
        depth=depth_levels,
        height=tile_size,
        width=tile_size,
        patch_size=dit_cfg.patch_size,
        patch_size_vert=None
        if dit_cfg.patch_size_vert < 0
        else dit_cfg.patch_size_vert,
        patch_size_horiz=None
        if dit_cfg.patch_size_horiz < 0
        else dit_cfg.patch_size_horiz,
        in_chans=in_channels,
        base_out_chans=out_channels,
        nside=nside,
        do_alt_depthwise_attn=dit_cfg.do_alt_depthwise_attn,
        do_interleaved_dilation=dit_cfg.do_interleaved_dilation,
        na_dilations=dit_cfg.na_dilations,
        embed_dim=dit_cfg.embed_dim,
        n_layers=dit_cfg.n_layers,
        num_heads=dit_cfg.num_heads,
        attn_kernel=dit_cfg.attn_kernel,
        do_rope_2d=dit_cfg.do_rope_2d,
        rope_length_scale=dit_cfg.rope_length_scale or None,
        do_concat_latitude=True,
        qk_norm=dit_cfg.qk_norm,
        qk_norm_elementwise_affine=dit_cfg.qk_norm_elementwise_affine,
        do_bf16_mixed=do_bf16_mixed,
        do_activation_checkpointing=dit_cfg.do_activation_checkpointing,
        do_rope_2d_stereographic=dit_cfg.do_rope_2d_stereographic,
        do_rotate_wind=dit_cfg.do_rotate_wind,
        wind_channel_indices=wind_channel_indices,
        gated_attention=dit_cfg.gated_attention,
        grid_type=grid_type,
        cubesphere_latlon_path=cubesphere_latlon_path,
        index_is_latlon=dit_cfg.index_is_latlon,
        na3d_backend=dit_cfg.na3d_backend or None,
        use_dealiased_patch_embed=dit_cfg.use_dealiased_patch_embed,
        dealias_resample_filter=dit_cfg.dealias_resample_filter,
    )


def build_backbone(
    in_channels: int,
    out_channels: int,
    nside: int,
    tile_size: int,
    dit_cfg: DiTConfig,
    do_bf16_mixed: bool,
    depth_levels: int,
    wind_channel_indices: tuple[int, int] | None = None,
    grid_type: str = "healpix",
    cubesphere_latlon_path: str | None = None,
):
    return StrataBackboneModel(
        **_dit_common_kwargs(
            in_channels,
            out_channels,
            nside,
            tile_size,
            dit_cfg,
            do_bf16_mixed,
            depth_levels,
            wind_channel_indices,
            grid_type,
            cubesphere_latlon_path,
        )
    )


def build_strata(
    in_channels: int,
    out_channels: int,
    nside: int,
    tile_size: int,
    dit_cfg: DiTConfig,
    pixel_cfg: PixelDiTConfig,
    do_bf16_mixed: bool,
    depth_levels: int,
    wind_channel_indices: tuple[int, int] | None = None,
    grid_type: str = "healpix",
    cubesphere_latlon_path: str | None = None,
):
    return StrataModel(
        **_dit_common_kwargs(
            in_channels,
            out_channels,
            nside,
            tile_size,
            dit_cfg,
            do_bf16_mixed,
            depth_levels,
            wind_channel_indices,
            grid_type,
            cubesphere_latlon_path,
        ),
        embed_dim_pixel=pixel_cfg.embed_dim,
        n_layers_pixel=pixel_cfg.n_layers,
        num_heads_pixel=pixel_cfg.num_heads,
        mlp_ratio_pixel=pixel_cfg.mlp_ratio,
        attn_kernel_pixel=pixel_cfg.attn_kernel,
        gated_attention_pixel=pixel_cfg.gated_attention,
        qk_norm_pixel=pixel_cfg.qk_norm,
        qk_norm_elementwise_affine_pixel=pixel_cfg.qk_norm_elementwise_affine,
        do_bf16_mixed_pixel=do_bf16_mixed,
        freeze_semantic=pixel_cfg.freeze_semantic,
        freeze_pixel_blocks=pixel_cfg.freeze_pixel_blocks,
        use_bilinear_dw_gelu_project_adaln_pixel=pixel_cfg.use_bilinear_dw_gelu_project_adaln,
        use_chunked_depthwise_conv_pixel=pixel_cfg.use_chunked_depthwise_conv,
        first_block_only_adaln_pixel=pixel_cfg.first_block_only_adaln,
        do_rope_2d_pixel=pixel_cfg.do_rope_2d,
        do_rope_2d_stereographic_pixel=pixel_cfg.do_rope_2d_stereographic,
    )


def _pixel_blocks_adaln_changed(checkpoint_path: str, model: torch.nn.Module) -> bool:
    """Return True if the model's pixel block adaln params are absent from the checkpoint.

    Indicates the adaln path changed (e.g. plain→bilinear_gelu), meaning pixel blocks
    have co-adapted to different conditioning and should be reinitialized.
    Only called on foreign checkpoint loads (not latest.pth resume).
    """
    from screamcast.checkpoint_compat import remap_legacy_state_dict

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        network_sd = ckpt.get("network", {})
        # Compare in migrated key space so legacy checkpoints (semantic./
        # _pixel_blocks./_orig_mod. keys) are judged correctly.
        network_sd, _ = remap_legacy_state_dict(dict(network_sd))
        ckpt_keys = set(network_sd.keys())
    except Exception:
        return False

    if not ckpt_keys:
        return False

    def _strip_wrapper(name: str) -> str:
        # Under torch.compile the model's parameter names carry _orig_mod.
        # while the remapped checkpoint keys never do; compare bare names.
        for prefix in ("_orig_mod.", "module.", "_forward_module."):
            while name.startswith(prefix):
                name = name[len(prefix) :]
        return name

    current_adaln_keys = {
        _strip_wrapper(name)
        for name, _ in model.named_parameters()
        if "pixel_blocks" in name and "adaln" in name
    }

    if not current_adaln_keys:
        return False

    # If none of the current adaln keys exist in the checkpoint → path changed
    return not any(k in ckpt_keys for k in current_adaln_keys)


def _reinit_pixel_blocks(model: torch.nn.Module) -> None:
    """Reinitialize all pixel block weights after loading a pretrained checkpoint.

    Use when switching adaln modes (e.g. plain→bilinear_dw_gelu_project) with a frozen
    semantic backbone: pixel blocks have co-adapted to the old tiled conditioning, so a
    fresh init avoids a stale prior. After reinit, adaln-zero is re-applied so blocks
    start as identity regardless of the new conditioning signal.
    """
    for block in model.strata.pixel_blocks:
        for m in block.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        # Re-apply adaln-zero to the output projection, ensuring
        # shift=0 / scale=1 / gate=0 at the start of fine-tuning.
        if hasattr(block, "adaln_bilinear_dw_proj"):
            torch.nn.init.zeros_(block.adaln_bilinear_dw_proj.weight)
            torch.nn.init.zeros_(block.adaln_bilinear_dw_proj.bias)
        elif hasattr(block, "adaln_pixel_proj"):
            torch.nn.init.zeros_(block.adaln_pixel_proj[-1].weight)
            torch.nn.init.zeros_(block.adaln_pixel_proj[-1].bias)


def train(
    name: str = typer.Argument(
        ..., help="Experiment name from experiments.py CONFIGS."
    ),
):
    from train_configs import CONFIGS

    if name not in CONFIGS:
        raise SystemExit(
            f"Unknown experiment '{name}'. Available: {sorted(CONFIGS.keys())}"
        )
    cfg = CONFIGS[name]

    # Enum conversions (genuine transformation)
    target_type = TargetOptions(cfg.pipeline.target_type)
    loss_type = LossOptions(cfg.training.loss_type)

    # Grid type normalization
    grid_type = (cfg.data.grid_type or "healpix").lower()

    # nside can be overridden for cubesphere (keep as mutable local)
    nside = cfg.data.nside

    train_main_zarr_paths = cfg.data.train_main_zarr_paths
    train_aux_zarr_paths = cfg.data.train_aux_zarr_paths
    train_zarr_weights = cfg.data.train_zarr_weights
    train_start_indices = cfg.data.train_start_indices
    train_end_indices = cfg.data.train_end_indices
    backtest_main_zarr_path = cfg.data.backtest_main_zarr_path
    backtest_aux_zarr_path = cfg.data.backtest_aux_zarr_path

    if train_main_zarr_paths:
        logging.info("Training zarr sources (main): %s", train_main_zarr_paths)
        if train_aux_zarr_paths:
            logging.info("Training zarr sources (aux): %s", train_aux_zarr_paths)
        if train_start_indices or train_end_indices:
            logging.info(
                "Training zarr indices: start=%s end=%s",
                train_start_indices,
                train_end_indices,
            )
    if backtest_main_zarr_path or backtest_aux_zarr_path:
        logging.info(
            "Backtest zarr paths: main=%s aux=%s",
            backtest_main_zarr_path,
            backtest_aux_zarr_path,
        )

    if grid_type not in {"healpix", "cubesphere"}:
        raise ValueError(
            f"Unsupported grid_type='{grid_type}'. Expected 'healpix' or 'cubesphere'."
        )

    # Face size used by models/tilers. For CubeSphere, ScreamV2 overwrites to ne*npg
    # (defaults ne=1024, npg=2 -> 2048). Ensure training code uses the same size.
    if grid_type == "cubesphere":
        nside = 2048

    if cfg.dit.do_rotate_wind:
        dit_wind_channel_indices = (
            cfg.data.variables_prognostic.index("U"),
            cfg.data.variables_prognostic.index("V"),
        )
    else:
        dit_wind_channel_indices = None

    random.seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)
    torch.cuda.manual_seed_all(cfg.training.seed)
    if cfg.training.do_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    logging.basicConfig(level=logging.INFO)
    plugins = None
    if os.getenv("SLURM_JOB_ID"):
        plugins = [environments.SLURMEnvironment()]

    fabric = Fabric(
        accelerator="cuda",
        num_nodes=os.getenv("SLURM_NNODES", 1),
        plugins=plugins,
    )
    fabric.launch()
    in_channels = ScreamV2.num_of_input_channels(
        cfg.data.variables_prognostic,
        cfg.data.variables_forcing,
        cfg.data.plevel,
        cfg.data.level_start,
        cfg.data.level_end,
    )
    out_channels = ScreamV2.num_of_output_channels(
        cfg.data.variables_prognostic,
        cfg.data.variables_diagnostic,
        cfg.data.plevel,
        cfg.data.level_start,
        cfg.data.level_end,
    )
    if cfg.experiment.model_type in ["songunetv2", "BruteForce_fix", "unet3dv2"]:
        raise NotImplementedError(
            f"model_type '{cfg.experiment.model_type}' is not supported with YAML config. Use 'dit3d'."
        )
    elif cfg.experiment.model_type in ["dit3d", "pixeldit"]:
        in_channels_3d = len(cfg.data.variables_prognostic) + len(
            cfg.data.variables_forcing
        )

        out_channels_3d = len(cfg.data.variables_prognostic) + len(
            cfg.data.variables_diagnostic
        )
        # Determine number of vertical levels for the 3D model depth
        num_depth_levels = len(
            np.r_[cfg.data.level_start : cfg.data.level_end : cfg.data.plevel]
        )

        if cfg.experiment.model_type == "pixeldit":
            if cfg.pixel_dit is None:
                raise ValueError("model_type='pixeldit' requires pixel_dit config")
            network = build_strata(
                in_channels_3d,
                out_channels_3d,
                nside,
                cfg.data.tile_size,
                dit_cfg=cfg.dit,
                pixel_cfg=cfg.pixel_dit,
                do_bf16_mixed=cfg.compute.do_bf16_mixed,
                depth_levels=num_depth_levels,
                wind_channel_indices=dit_wind_channel_indices,
                grid_type=grid_type,
                cubesphere_latlon_path=cfg.data.latlon_path
                if grid_type == "cubesphere"
                else None,
            )
        else:
            network = build_backbone(
                in_channels_3d,
                out_channels_3d,
                nside,
                cfg.data.tile_size,
                dit_cfg=cfg.dit,
                do_bf16_mixed=cfg.compute.do_bf16_mixed,
                depth_levels=num_depth_levels,
                wind_channel_indices=dit_wind_channel_indices,
                grid_type=grid_type,
                cubesphere_latlon_path=cfg.data.latlon_path
                if grid_type == "cubesphere"
                else None,
            )
    else:
        raise ValueError(
            f"Unknown model: {cfg.experiment.model_type}. Expected 'dit3d' or 'pixeldit'."
        )
    if cfg.compute.do_torch_compile:
        network = torch.compile(network)

    enable_3d_adapter = cfg.experiment.model_type in ["dit3d", "pixeldit"]

    loss_fn = resolve_loss(loss_type)

    # Helper function to compute U/V channel groups for wind rotation
    def get_uv_channel_groups_from_ranges(ranges_dict):
        """Compute U/V channel groups for tied normalization using channel ranges."""
        u_slice = ranges_dict["U"]
        v_slice = ranges_dict["V"]

        # Create channel groups: pair U and V at each vertical level
        channel_groups = []
        for u_channel, v_channel in zip(
            range(u_slice.start, u_slice.stop), range(v_slice.start, v_slice.stop)
        ):
            channel_groups.append([u_channel, v_channel])

        return channel_groups

    # Compute U/V channel groups if wind rotation is enabled (shared for all target types)
    input_channel_groups = None
    target_channel_groups = None
    if cfg.dit.do_rotate_wind:
        # check if U and V are in the variables_prognostic
        if (
            "U" not in cfg.data.variables_prognostic
            or "V" not in cfg.data.variables_prognostic
        ):
            raise ValueError("U and V must be in the variables_prognostic")

        # Get channel ranges for input and output
        ranges_input = ScreamV2.ranges_input(
            variables_prognostic=cfg.data.variables_prognostic,
            variables_forcing=cfg.data.variables_forcing,
            plevel=cfg.data.plevel,
            level_start=cfg.data.level_start,
            level_end=cfg.data.level_end,
        )
        ranges_output = ScreamV2.ranges_output(
            variables_prognostic=cfg.data.variables_prognostic,
            variables_diagnostic=cfg.data.variables_diagnostic,
            plevel=cfg.data.plevel,
            level_start=cfg.data.level_start,
            level_end=cfg.data.level_end,
        )

        input_channel_groups = get_uv_channel_groups_from_ranges(ranges_input)
        target_channel_groups = get_uv_channel_groups_from_ranges(ranges_output)
        print(
            f"Tied U/V channel normalization: {len(input_channel_groups)} level pairs"
        )

    if target_type == TargetOptions.mixed_state_diff:
        pipeline = model_pipelines.MixedPredictionAsymmetric(
            network=network,
            loss_fn=loss_fn,
            input_norm=RunningNorm2d(
                in_channels,
                fit_batches=cfg.training.fit_batches_for_norm,
                channel_groups=input_channel_groups,
            ),
            target_norm=RunningNorm2d(
                out_channels,
                fit_batches=cfg.training.fit_batches_for_norm,
                channel_groups=target_channel_groups,
            ),
            plevel=cfg.data.plevel,
            level_start=cfg.data.level_start,
            level_end=cfg.data.level_end,
            variables_prognostic=cfg.data.variables_prognostic,
            variables_forcing=cfg.data.variables_forcing,
            variables_diagnostic=cfg.data.variables_diagnostic,
            variables_prognostic_state=cfg.data.variables_prognostic_state,
            enable_3d_adapter=enable_3d_adapter,
            do_qv_softplus=cfg.pipeline.do_qv_softplus,
            do_precip_relu=cfg.pipeline.do_precip_relu,
            variables_input_zeroed=cfg.pipeline.variables_input_zeroed,
        )
    else:
        raise NotImplementedError(
            f"target_type {target_type} currently not implemented for ScreamV2"
        )

    pipeline.plevel = cfg.data.plevel

    print_network_info(pipeline.network)

    if cfg.training.optimizer == "adam":
        # Define Adam optimizer
        optimizer = Adam(
            pipeline.network.parameters(),
            lr=cfg.training.lr,
            betas=(cfg.training.beta1, cfg.training.beta2),
        )
    elif cfg.training.optimizer == "adamw":
        optimizer = AdamW(
            pipeline.network.parameters(),
            lr=cfg.training.lr,
            betas=(cfg.training.beta1, cfg.training.beta2),
            weight_decay=cfg.training.weight_decay,
        )
    elif cfg.training.optimizer == "stableadamw":
        optimizer = StableAdamW(
            pipeline.network.parameters(),
            lr=cfg.training.lr,
            betas=(cfg.training.beta1, cfg.training.beta2),
            weight_decay=cfg.training.weight_decay,
        )
    elif cfg.training.optimizer == "sgd":
        optimizer = SGD(
            pipeline.network.parameters(),
            lr=cfg.training.lr,
            momentum=cfg.training.beta1,
            weight_decay=cfg.training.weight_decay,
            nesterov=True,
        )
    else:
        raise ValueError(
            f"Unknown optimizer: {cfg.training.optimizer}. Expected 'adam', 'adamw', 'stableadamw' or 'sgd'"
        )

    loss_optimizer = None
    if len(list(pipeline.loss_fn.parameters())) > 0:
        loss_optimizer = Adam(pipeline.loss_fn.parameters(), lr=cfg.training.loss_lr)

    # Determine whether to only load final target (optimization for "final_only" mode)
    # When num_steps == 1 or multistep_training_mode == "final_only", we only need
    # the final target for loss computation, not all intermediate targets.
    load_final_target_only = (
        cfg.training.num_steps == 1
        or cfg.training.multistep_training_mode == "final_only"
    )

    # Helper functions to create test/backtest dataloaders
    # These will be called fresh each validation for deterministic sampling
    def create_test_dataloader():
        return get_dataloader(
            global_rank=fabric.global_rank,
            world_size=fabric.world_size,
            device_id=fabric.device.index,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.training.num_workers_inference,
            split="test",
            mock=cfg.debug.mock,
            grid_type=grid_type,
            plevel=cfg.data.plevel,
            tile_size=cfg.data.tile_size,
            level_start=cfg.data.level_start,
            level_end=cfg.data.level_end,
            variables_prognostic=cfg.data.variables_prognostic,
            variables_forcing=cfg.data.variables_forcing,
            variables_diagnostic=cfg.data.variables_diagnostic,
            use_time_variation_seed=False,
            use_fixed_seed=True,
            split_by_time=cfg.data.split_by_time,
            train_start_index=cfg.data.train_start_index,
            train_end_index=cfg.data.train_end_index,
            test_start_index=cfg.data.test_start_index,
            test_end_index=cfg.data.test_end_index,
            test_stride=cfg.data.test_stride,
            num_steps=cfg.training.num_steps,
            load_final_target_only=load_final_target_only,
            cross_face_tiles=False,
            shuffle_tiles=False,
        )

    def create_backtest_dataloader():
        bt_start_index = cfg.data.backtest_start_index
        bt_end_index = cfg.data.backtest_end_index
        bt_stride = cfg.data.backtest_stride or cfg.data.test_stride
        if bt_start_index is None or bt_end_index is None:
            if (
                cfg.data.train_end_index is None
                or cfg.data.test_start_index is None
                or cfg.data.test_end_index is None
            ):
                raise ValueError(
                    "backtest_start_index/backtest_end_index are required when "
                    "train_end_index/test_start_index/test_end_index are not set."
                )
            bt_start_index = cfg.data.train_end_index - (
                cfg.data.test_end_index - cfg.data.test_start_index
            )
            bt_end_index = cfg.data.train_end_index
        return get_dataloader(
            global_rank=fabric.global_rank,
            world_size=fabric.world_size,
            device_id=fabric.device.index,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.training.num_workers_inference,
            split="test",
            mock=cfg.debug.mock,
            grid_type=grid_type,
            plevel=cfg.data.plevel,
            tile_size=cfg.data.tile_size,
            level_start=cfg.data.level_start,
            level_end=cfg.data.level_end,
            variables_prognostic=cfg.data.variables_prognostic,
            variables_forcing=cfg.data.variables_forcing,
            variables_diagnostic=cfg.data.variables_diagnostic,
            use_time_variation_seed=False,
            use_fixed_seed=True,
            split_by_time=cfg.data.split_by_time,
            train_start_index=cfg.data.train_start_index,
            train_end_index=cfg.data.train_end_index,
            test_start_index=bt_start_index,
            test_end_index=bt_end_index,
            test_stride=bt_stride,
            num_steps=cfg.training.num_steps,
            load_final_target_only=load_final_target_only,
            cross_face_tiles=False,
            shuffle_tiles=False,
            main_zarr_path=backtest_main_zarr_path,
            aux_zarr_path=backtest_aux_zarr_path,
        )

    dataloader = get_dataloader(
        global_rank=fabric.global_rank,
        world_size=fabric.world_size,
        device_id=fabric.device.index,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        split="train",
        mock=cfg.debug.mock,
        grid_type=grid_type,
        plevel=cfg.data.plevel,
        tile_size=cfg.data.tile_size,
        level_start=cfg.data.level_start,
        level_end=cfg.data.level_end,
        variables_prognostic=cfg.data.variables_prognostic,
        variables_forcing=cfg.data.variables_forcing,
        variables_diagnostic=cfg.data.variables_diagnostic,
        use_time_variation_seed=cfg.training.use_time_variation_seed,
        split_by_time=cfg.data.split_by_time,
        train_start_index=cfg.data.train_start_index,
        train_end_index=cfg.data.train_end_index,
        test_start_index=cfg.data.test_start_index,
        test_end_index=cfg.data.test_end_index,
        test_stride=cfg.data.test_stride,
        num_steps=cfg.training.num_steps,
        load_final_target_only=load_final_target_only,
        cross_face_tiles=cfg.data.cross_face_tiles,
        use_duo_padding=cfg.data.use_duo_padding,
        duo_padding_scrip_src_path=cfg.data.duo_padding_scrip_src_path or None,
        duo_padding_scrip_tgt_path=cfg.data.duo_padding_scrip_tgt_path or None,
        skip_corner_tiles=cfg.data.skip_corner_tiles,
        index_is_latlon=cfg.dit.index_is_latlon,
        latlon_path=cfg.data.latlon_path if cfg.dit.index_is_latlon else None,
        balance_cross_face=cfg.data.balance_cross_face,
        train_main_zarr_paths=train_main_zarr_paths,
        train_aux_zarr_paths=train_aux_zarr_paths,
        train_zarr_weights=train_zarr_weights,
        train_start_indices=train_start_indices,
        train_end_indices=train_end_indices,
    )

    # Create test/backtest dataloaders once if not recreating each validation
    if not cfg.debug.recreate_dataloader_each_validation:
        fabric.print("Creating test dataloader once at startup...")
        test_dataloader = create_test_dataloader()
        if cfg.training.do_backtest_inference:
            fabric.print("Creating backtest dataloader once at startup...")
            backtest_dataloader = create_backtest_dataloader()

    # setup fabric stuff
    pipeline, optimizer = fabric.setup(pipeline, optimizer)
    pipeline.mark_forward_method("get_loss")  # get_loss is the method used for ddp
    pipeline.mark_forward_method("get_multistep_loss")
    pipeline.mark_forward_method("step")
    pipeline.mark_forward_method("initialize")

    if loss_optimizer is not None:
        loss_optimizer = fabric.setup_optimizers(loss_optimizer)

    validate_steps = cfg.training.validate_steps or len(dataloader)
    if cfg.training.max_steps_per_epoch is not None:
        validate_steps = min(validate_steps, cfg.training.max_steps_per_epoch)
    fabric.print(f"{validate_steps=}")
    fabric.print(f"{len(dataloader)=}")

    # set LR scheduler
    scheduler = None
    if cfg.training.scheduler_type.lower() == "cosine":
        # Cosine schedule stepped per batch. If cosine_t_max is 0, use total training steps.
        if cfg.training.cosine_t_max == 0:
            steps_per_epoch = (
                cfg.training.max_steps_per_epoch
                if cfg.training.max_steps_per_epoch is not None
                else len(dataloader)
            )
            t_max_steps = steps_per_epoch * cfg.training.epochs
        else:
            t_max_steps = cfg.training.cosine_t_max * validate_steps
        scheduler = CosineAnnealingLR(
            optimizer, T_max=t_max_steps, eta_min=0.1 * cfg.training.lr
        )
    elif cfg.training.scheduler_type.lower() == "dampened_cosine_with_hard_restarts":
        if cfg.training.cosine_t_max == 0:
            steps_per_epoch = (
                cfg.training.max_steps_per_epoch
                if cfg.training.max_steps_per_epoch is not None
                else len(dataloader)
            )
            t_max_steps = steps_per_epoch * cfg.training.epochs
        else:
            t_max_steps = cfg.training.cosine_t_max * validate_steps

        scheduler = get_dampened_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg.training.warmup_steps,
            num_training_steps=t_max_steps,
            num_cycles=cfg.training.cosine_n_cycles,
            decay=cfg.training.cosine_decay,
            last_epoch=-1,
            min_ratio=0.01,
        )
    elif cfg.training.scheduler_type.lower() == "warmup_constant":
        scheduler = get_warmup_constant_schedule(
            optimizer,
            num_warmup_steps=cfg.training.warmup_steps,
        )

    # start training
    state = {
        "network": pipeline.network,
        "optimizer": optimizer,
        "input_norm": pipeline.input_norm,
        "target_norm": getattr(pipeline, "target_norm", None),
        "loss_fn": pipeline.loss_fn,
        "epoch": 0,
        "best_valid_loss": 100.0,
        "nimg": 0,
        "steps": 0,
        "plevel": cfg.data.plevel,
        "level_start": cfg.data.level_start,
        "level_end": cfg.data.level_end,
        "variables_prognostic": cfg.data.variables_prognostic,
        "variables_forcing": cfg.data.variables_forcing,
        "variables_diagnostic": cfg.data.variables_diagnostic,
        "variables_prognostic_state": cfg.data.variables_prognostic_state,
        "train_config": dataclasses.asdict(cfg),
        "wandb_run_id": None,
    }
    if scheduler is not None:
        state["scheduler_state"] = scheduler.state_dict()

    checkpoint_path = os.path.join(cfg.experiment.rundir, "latest.pth")
    checkpoint_path_best = os.path.join(cfg.experiment.rundir, "best.pth")
    latest_pth_exists = os.path.exists(checkpoint_path)
    if not latest_pth_exists and cfg.experiment.resume_from:
        checkpoint_path = cfg.experiment.resume_from

    checkpoint_path = os.path.abspath(checkpoint_path)

    lat_radians = None
    lon_radians = None
    if cfg.dit.index_is_latlon:
        if grid_type == "healpix":
            input_grid = earth2grid.healpix.Grid(
                earth2grid.healpix.nside2level(nside),
                pixel_order=earth2grid.healpix.PixelOrder.NEST,
            )
            lat_radians = input_grid.lat * np.pi / 180.0
            lon_radians = input_grid.lon * np.pi / 180.0
        else:
            ds_ll = xr.open_dataset(cfg.data.latlon_path)
            lat_deg = ds_ll["lat"].values
            lon_deg = ds_ll["lon"].values
            lat_radians = lat_deg * (np.pi / 180.0)
            lon_radians = lon_deg * (np.pi / 180.0)
        lat_radians = torch.from_numpy(lat_radians).float()
        lon_radians = torch.from_numpy(lon_radians).float()

    os.makedirs(cfg.experiment.rundir, exist_ok=True)
    os.chdir(cfg.experiment.rundir)

    try:
        # skip_optimizer_reloading only applies to foreign-checkpoint loads
        # (latest.pth missing). Once latest.pth exists, the job has been launched
        # at least once and we're resuming an ongoing run — always reload optimizer
        # state so Adam momenta survive preemption/resubmission. This mirrors the
        # scheduler restore logic below.
        skip_optimizer_reload = (
            cfg.experiment.skip_optimizer_reloading and not latest_pth_exists
        )
        if not skip_optimizer_reload:
            try:
                fabric.load(checkpoint_path, state, strict=False)
            except FileNotFoundError:
                raise
            except Exception as e:
                # Fallback: architecture changed (e.g., 16 -> 32 layers) causing optimizer param-group mismatch
                fabric.print(
                    f"Warning: failed to load optimizer state ({e}). "
                    "Reloading checkpoint without optimizer state."
                )
                _ = state.pop("optimizer", None)
                fabric.load(checkpoint_path, state, strict=False)
                state["optimizer"] = optimizer
        else:
            _ = state.pop("optimizer", None)
            fabric.load(checkpoint_path, state, strict=False)
            state["optimizer"] = optimizer
        fabric.print(f"Loaded checkpoint from {checkpoint_path}")

        # Restore scheduler if available and used.
        # latest.pth exists → ongoing run being resumed → always load scheduler state.
        # latest.pth missing (e.g. resume_from a pre-trained model) → respect reset_scheduler_state flag.
        if scheduler is not None and latest_pth_exists:
            try:
                scheduler.load_state_dict(state["scheduler_state"])
            except Exception as e:
                fabric.print(f"Warning: failed to load scheduler state: {e}")

        if not latest_pth_exists and cfg.experiment.reset_scheduler_state:
            # Reset optimizer LR unconditionally — covers both scheduler and
            # no-scheduler (e.g. constant) cases when resuming from a foreign
            # checkpoint that may have been saved at a different LR.
            for pg in optimizer.param_groups:
                pg["lr"] = cfg.training.lr
            fabric.print(
                f"reset_scheduler_state=True: reset optimizer LR to {cfg.training.lr}"
            )

        if (
            not os.path.exists(checkpoint_path_best)
            and cfg.experiment.reset_best_valid_loss
        ):
            state["best_valid_loss"] = 100.0
            fabric.print(
                "reset_best_valid_loss=True: starting best_valid_loss from 100.0"
            )

        if not latest_pth_exists:
            # Resuming from a foreign checkpoint — start a fresh wandb run so we
            # don't inherit the base experiment's run ID and log out-of-order steps.
            state["wandb_run_id"] = None

        if not latest_pth_exists and _pixel_blocks_adaln_changed(
            checkpoint_path, pipeline.module.network
        ):
            # Adaln path changed vs checkpoint (e.g. plain→bilinear_gelu): pixel blocks
            # have co-adapted to the old conditioning so reinitialize from scratch.
            # adaln-zero is re-applied inside for training stability.
            _reinit_pixel_blocks(pipeline.module.network)
            fabric.print("Auto-detected adaln path change: reinitialized pixel blocks")
    except FileNotFoundError:
        fabric.print("Training from scratch")

    steps = state["steps"]
    nimg = state["nimg"]

    # fabric.load() overwrites scalar entries in state with values from the
    # resumed checkpoint.  Re-stamp train_config so saved checkpoints always
    # reflect the *current* CONFIGS entry
    state["train_config"] = dataclasses.asdict(cfg)

    if state["plevel"] != cfg.data.plevel:
        state["plevel"] = cfg.data.plevel
    if state["level_start"] != cfg.data.level_start:
        state["level_start"] = cfg.data.level_start
    if state["level_end"] != cfg.data.level_end:
        state["level_end"] = cfg.data.level_end

    start_epoch = state["epoch"]
    pipeline.input_norm.n_batches = state["steps"]
    try:
        pipeline.target_norm.n_batches = state["steps"]
    except AttributeError:
        pass

    # Apply channel tying if resuming from checkpoint with fitted normalization
    if (
        cfg.dit.do_rotate_wind
        and pipeline.input_norm.n_batches >= cfg.training.fit_batches_for_norm
    ):
        fabric.print("Applying U/V channel tying to loaded normalization statistics...")
        pipeline.input_norm.apply_channel_tying_now()
        try:
            pipeline.target_norm.apply_channel_tying_now()
        except AttributeError:
            pass
    if cfg.training.normalization_ref_path_input:
        try:
            checkpoint = torch.load(
                cfg.training.normalization_ref_path_input, weights_only=True
            )
            pipeline.input_norm.load_state_dict(checkpoint["input_norm"])
            pipeline.input_norm.n_batches = cfg.training.fit_batches_for_norm + 1
            # Apply channel tying after loading external normalization
            if cfg.dit.do_rotate_wind:
                fabric.print(
                    "Applying U/V channel tying to loaded input normalization..."
                )
                pipeline.input_norm.apply_channel_tying_now()
        except Exception as e:
            fabric.print(
                f"Warning: failed to load input normalization from {cfg.training.normalization_ref_path_input}: {e}"
            )

    if cfg.training.normalization_ref_path_target:
        try:
            checkpoint = torch.load(
                cfg.training.normalization_ref_path_target, weights_only=True
            )
            pipeline.target_norm.load_state_dict(checkpoint["target_norm"])
            pipeline.target_norm.n_batches = cfg.training.fit_batches_for_norm + 1
            # Apply channel tying after loading external normalization
            if cfg.dit.do_rotate_wind:
                fabric.print(
                    "Applying U/V channel tying to loaded target normalization..."
                )
                pipeline.target_norm.apply_channel_tying_now()
        except Exception as e:
            fabric.print(
                f"Warning: failed to load target normalization from {cfg.training.normalization_ref_path_target}: {e}"
            )

    writer = None
    wandb_run = None
    if fabric.global_rank == 0:
        writer = SummaryWriter()
        if (
            wandb_lib is not None
            and os.environ.get("WANDB_API_KEY")
            and not cfg.debug.wandb_disabled
        ):
            try:
                wandb_run = wandb_lib.init(
                    id=state.get("wandb_run_id"),
                    resume="allow",
                    name=cfg.experiment.name,
                    project=os.environ.get("WANDB_PROJECT", "screamcast"),
                    entity=os.environ.get("WANDB_ENTITY") or None,
                    config=dataclasses.asdict(cfg),
                )
                state["wandb_run_id"] = wandb_run.id
            except Exception as e:
                fabric.print(
                    f"Warning: wandb init failed: {e}. Continuing without wandb."
                )
        else:
            fabric.print(
                "WANDB_API_KEY not set or wandb not installed, skipping wandb."
            )

    # Add validation using inference.validate
    eval_on_india_ocean_tile = inference.EvalIndiaOceantile(
        fabric.device,
        grid_type=grid_type,
        plevel=cfg.data.plevel,
        tile_size=cfg.data.tile_size,
        level_start=cfg.data.level_start,
        level_end=cfg.data.level_end,
        variables_prognostic=cfg.data.variables_prognostic,
        variables_forcing=cfg.data.variables_forcing,
        variables_diagnostic=cfg.data.variables_diagnostic,
        inference_start_index=cfg.data.inference_start_index,
    )
    ds = ScreamV2(
        batch_size=1,
        grid_type=grid_type,
        plevel=cfg.data.plevel,
        level_start=cfg.data.level_start,
        level_end=cfg.data.level_end,
        variables_prognostic=cfg.data.variables_prognostic,
        variables_forcing=cfg.data.variables_forcing,
        variables_diagnostic=cfg.data.variables_diagnostic,
        split_by_time=cfg.data.split_by_time,
        train_start_index=cfg.data.train_start_index,
        train_end_index=cfg.data.train_end_index,
        test_start_index=cfg.data.test_start_index,
        test_end_index=cfg.data.test_end_index,
        test_stride=cfg.data.test_stride,
        num_steps=cfg.training.num_steps,
        mock=cfg.debug.mock,
    )

    # Parse pooled loss weights into a dict[int, float]
    pooled_loss_weights_dict: dict[int, float] | None = None
    if cfg.training.pooled_loss:
        try:
            parsed: dict[int, float] = {}
            for item in cfg.training.pooled_loss.split(","):
                item = item.strip()
                if not item:
                    continue
                size_str, weight_str = item.split(":")
                size = int(size_str.strip())
                weight = float(weight_str.strip())
                parsed[size] = weight
            pooled_loss_weights_dict = parsed if parsed else None
        except Exception as e:
            fabric.print(
                f"Failed to parse --pooled-loss '{cfg.training.pooled_loss}': {e}. Ignoring."
            )
            pooled_loss_weights_dict = None

    # Parse pooled loss channel weights from CSV
    pooled_loss_channel_weights_tensor: torch.Tensor | None = None
    if cfg.training.pooled_loss_channel_weights_csv:
        try:
            df = pd.read_csv(cfg.training.pooled_loss_channel_weights_csv)
            # Calculate the ratio of loss over loss_tilemean
            df["ratio"] = df["loss"] / df["loss_tilemean"]
            weight = df["ratio"].values
            pooled_loss_channel_weights_tensor = torch.tensor(
                weight, dtype=torch.float32, device=fabric.device
            )
            fabric.print(
                f"Loaded pooled_loss_channel_weights from {cfg.training.pooled_loss_channel_weights_csv}: "
                f"shape={pooled_loss_channel_weights_tensor.shape}"
            )
        except Exception as e:
            fabric.print(
                f"Failed to load pooled_loss_channel_weights from '{cfg.training.pooled_loss_channel_weights_csv}': {e}. Ignoring."
            )
            pooled_loss_channel_weights_tensor = None

    n_validations_done = 0
    for epoch in range(start_epoch, cfg.training.epochs):  # Example: 10 epochs
        fabric.print(f"Epoch {epoch}")
        loss_meter = torchmetrics.MeanMetric().to(fabric.device)
        print_loss_meter = torchmetrics.MeanMetric().to(fabric.device)
        # Track averaged component losses per epoch (created lazily)
        term_loss_meters: dict[str, torchmetrics.MeanMetric] = {}

        last_time = time.time()
        epoch_steps = 0  # Counter for steps in current epoch

        for batch in dataloader:
            # Unpack batch for 1-step or multi-step regime
            if cfg.training.num_steps > 1:
                # Multi-step format: targets [B, T, C, H, W], s_forcings [B, T-1, C, H, W]
                inputs, targets_all, index, j, s_forcings = batch
            else:
                # Single-step format: targets [B, C, H, W]
                inputs, targets, index, j = batch

            if cfg.dit.index_is_latlon:
                # Fast path only: dataloader must provide [B, 2, H, W] (lat, lon) in radians.
                if not (
                    isinstance(index, torch.Tensor)
                    and index.dim() == 4
                    and index.shape[1] == 2
                ):
                    raise RuntimeError(
                        "dit_index_is_latlon=True requires dataloader index tensor "
                        "shape [B, 2, H, W]. Ensure get_dataloader(..., "
                        "index_is_latlon=True, latlon_path=...) is enabled."
                    )
                lat = index[:, 0]  # [B, H, W]
                lon = index[:, 1]  # [B, H, W]
                index = {"lat": lat, "lon": lon}

            # Check if we've reached the maximum steps for this epoch
            if (
                cfg.training.max_steps_per_epoch is not None
                and epoch_steps >= cfg.training.max_steps_per_epoch
            ):
                fabric.print(
                    f"Reached max_steps_per_epoch ({cfg.training.max_steps_per_epoch}) for epoch {epoch}"
                )
                break
            pipeline.train()

            optimizer.zero_grad()
            if loss_optimizer:
                loss_optimizer.zero_grad()

            # Compute loss based on training mode
            if cfg.training.num_steps == 1:
                # Single-step: simple loss computation
                loss_result = pipeline.get_loss(
                    inputs,
                    targets,
                    index,
                    return_details=pooled_loss_weights_dict is not None,
                    pooled_loss_weights=pooled_loss_weights_dict,
                    pooled_loss_channel_weights=pooled_loss_channel_weights_tensor,
                )
            else:
                # Multi-step: use get_multistep_loss method
                loss_result = pipeline.get_multistep_loss(
                    inputs=inputs,
                    targets_all=targets_all,
                    index=index,
                    s_forcings=s_forcings,
                    num_steps=cfg.training.num_steps,
                    multistep_training_mode=cfg.training.multistep_training_mode,
                    return_details=pooled_loss_weights_dict is not None,
                    pooled_loss_weights=pooled_loss_weights_dict,
                    pooled_loss_channel_weights=pooled_loss_channel_weights_tensor,
                    return_final_output=False,
                )

            # Unpack potential (total, details) return for logging
            if isinstance(loss_result, tuple) and len(loss_result) == 2:
                loss, loss_terms = loss_result
            else:
                loss = loss_result
                loss_terms = None

            fabric.backward(loss)

            grad_norm = fabric.clip_gradients(
                pipeline,
                optimizer,
                max_norm=cfg.training.grad_clip_max_norm,
                norm_type=2,
            )

            optimizer.step()

            if loss_optimizer is not None:
                loss_optimizer.step()

            if scheduler is not None:
                scheduler.step()
                state["scheduler_state"] = scheduler.state_dict()

            loss_meter.update(loss)
            print_loss_meter.update(loss)
            if loss_terms is not None:
                for name, term_val in loss_terms.items():
                    if name not in term_loss_meters:
                        term_loss_meters[name] = torchmetrics.MeanMetric(
                            sync_on_compute=True
                        ).to(fabric.device)
                    term_loss_meters[name].update(term_val)

            steps += 1
            epoch_steps += 1  # Increment epoch step counter
            nimg += fabric.world_size * inputs.size(0)

            state["steps"] = steps
            state["nimg"] = nimg

            if writer is not None:
                writer.add_scalar("train_loss", loss, global_step=nimg)
                if loss_terms is not None:
                    for name, term_val in loss_terms.items():
                        writer.add_scalar(
                            f"train_loss/{name}", term_val, global_step=nimg
                        )
                writer.add_scalar(
                    "lr_per_step",
                    cfg.training.lr
                    if scheduler is None
                    else optimizer.param_groups[0]["lr"],
                    global_step=nimg,
                )
                writer.add_scalar("grad_norm", grad_norm.item(), global_step=nimg)

            if wandb_run is not None:
                current_lr = (
                    cfg.training.lr
                    if scheduler is None
                    else optimizer.param_groups[0]["lr"]
                )
                wandb_metrics = {
                    "train_loss": loss,
                    "lr_per_step": current_lr,
                    "grad_norm": grad_norm.item(),
                }
                if loss_terms is not None:
                    for name, term_val in loss_terms.items():
                        wandb_metrics[f"train_loss/{name}"] = term_val
                wandb_run.log(wandb_metrics, step=nimg)

            if torch.isnan(loss):
                raise RuntimeError(
                    f"Nan found in loss {fabric.global_rank=} {nimg=} {j=}"
                )

            # Periodic backup (independent of validation)
            if (
                cfg.training.backup_every_steps
                and cfg.training.backup_every_steps > 0
                and steps % cfg.training.backup_every_steps == 0
                and not cfg.training.no_save
            ):
                fabric.save("latest.pth", state)

            if (steps % cfg.debug.print_steps == 0) or (steps % validate_steps == 0):
                torch.cuda.synchronize()
                now = time.time()
                step_per_sec = cfg.debug.print_steps / (now - last_time)
                img_per_step = fabric.world_size * cfg.training.batch_size
                img_per_sec = step_per_sec * img_per_step

                train_loss = print_loss_meter.compute().item()
                print_loss_meter.reset()

                fabric.print(
                    f"{nimg=} {step_per_sec=:.2f} {img_per_sec=:.2f} {train_loss=:.2e} "
                    f"grad_norm_l2={grad_norm.item():.2e}"
                )
                last_time = now

            if steps % validate_steps == 0:
                with nvtx.annotate("Validation"):
                    pipeline.eval()
                    train_loss = loss_meter.compute().item()
                    loss_meter.reset()

                    # Optionally recreate test dataloader for deterministic sampling
                    if cfg.debug.recreate_dataloader_each_validation:
                        # Cleanup old test dataloader if it exists
                        if "test_dataloader" in locals():
                            del test_dataloader
                            gc.collect()
                            torch.cuda.empty_cache()

                        fabric.print("Recreating test dataloader for validation...")
                        test_dataloader = create_test_dataloader()
                        fabric.barrier()
                    if not cfg.debug.use_full_test_set:
                        fabric.print(
                            "after-epoch validation is on one batch from each gpu"
                        )
                        test_dataloader = [next(iter(test_dataloader))]
                    else:
                        fabric.print("after-epoch validation is on the full test set")
                    test_loss, median_score, median_score_tilemean = inference.validate(
                        # need to pass non-wrapped pipeline to easily use its methods
                        # see https://lightning.ai/docs/fabric/2.4.0/api/wrappers.html#unwrapping-the-model
                        pipeline=pipeline.module,
                        loader=test_dataloader,
                        device=fabric.device,
                        channel_index=ds.channel_index_output(),
                        output_dir=f"scores/{nimg}/" if fabric.global_rank == 0 else "",
                        min_samples=cfg.training.validate_min_samples,
                        writer=writer,
                        nimg=nimg,
                        pooled_loss_weights=pooled_loss_weights_dict,
                        num_steps=cfg.training.num_steps,
                        multistep_training_mode=cfg.training.multistep_training_mode,
                        lat_radians=lat_radians,
                        lon_radians=lon_radians,
                    )

                    if cfg.training.do_backtest_inference:
                        # Optionally recreate backtest dataloader for deterministic sampling
                        if cfg.debug.recreate_dataloader_each_validation:
                            # Cleanup old backtest dataloader if it exists
                            if "backtest_dataloader" in locals():
                                del backtest_dataloader
                                gc.collect()
                                torch.cuda.empty_cache()

                            fabric.print(
                                "Recreating backtest dataloader for validation..."
                            )
                            backtest_dataloader = create_backtest_dataloader()
                            fabric.barrier()
                        (
                            backtest_loss,
                            backtest_median_score,
                            backtest_median_score_tilemean,
                        ) = inference.validate(
                            pipeline=pipeline.module,
                            loader=backtest_dataloader,
                            device=fabric.device,
                            channel_index=ds.channel_index_output(),
                            output_dir=f"scores/{nimg}/"
                            if fabric.global_rank == 0
                            else "",
                            min_samples=cfg.training.validate_min_samples,
                            writer=writer,
                            nimg=nimg,
                            postfix="_backtest",
                            pooled_loss_weights=pooled_loss_weights_dict,
                            num_steps=cfg.training.num_steps,
                            multistep_training_mode=cfg.training.multistep_training_mode,
                            lat_radians=lat_radians,
                            lon_radians=lon_radians,
                        )

                    print_str = f"{epoch=} {nimg=} {test_loss=:.4f} {train_loss=:.4f} {median_score=:.4f} {median_score_tilemean=:.4f}"
                    if cfg.training.do_backtest_inference:
                        print_str += f" {backtest_loss=:.4f} {backtest_median_score=:.4f} {backtest_median_score_tilemean=:.4f}"
                    fabric.print(print_str)

                    # Compute averaged component losses on all ranks, then log (to keep collectives in sync)
                    computed_term_values: dict[str, float] = {}
                    if term_loss_meters:
                        for name, meter in term_loss_meters.items():
                            computed_term_values[name] = meter.compute().item()
                            meter.reset()
                    if writer is not None:
                        writer.add_scalar(
                            "train_loss_average", train_loss, global_step=nimg
                        )
                        for name, avg_val in computed_term_values.items():
                            writer.add_scalar(
                                f"train_loss_average/{name}",
                                avg_val,
                                global_step=nimg,
                            )
                        current_lr = optimizer.param_groups[0]["lr"]
                        writer.add_scalar("lr", current_lr, global_step=nimg)

                    if wandb_run is not None:
                        current_lr = optimizer.param_groups[0]["lr"]
                        val_metrics = {
                            "train_loss_average": train_loss,
                            "lr": current_lr,
                            "test_loss": test_loss,
                            "median_score": median_score,
                            "median_score_tilemean": median_score_tilemean,
                        }
                        for name, avg_val in computed_term_values.items():
                            val_metrics[f"train_loss_average/{name}"] = avg_val
                        if cfg.training.do_backtest_inference:
                            val_metrics["backtest_loss"] = backtest_loss
                            val_metrics["backtest_median_score"] = backtest_median_score
                            val_metrics[
                                "backtest_median_score_tilemean"
                            ] = backtest_median_score_tilemean
                        wandb_run.log(val_metrics, step=nimg)

                    if test_loss < state["best_valid_loss"]:
                        state["best_valid_loss"] = test_loss
                        link_to_best = True
                    else:
                        link_to_best = False

                    if not cfg.training.no_save:
                        fabric.save("latest.pth", state)
                        if link_to_best:
                            if fabric.global_rank == 0:
                                try:
                                    os.unlink("best.pth")
                                except FileNotFoundError:
                                    pass

                                os.link("latest.pth", "best.pth")
                        if cfg.training.save_ckpt_every_epoch:
                            fabric.save(f"steps_{steps}.pth", state)

                    if (
                        not cfg.debug.skip_india_ocean_eval
                        and not cfg.training.no_save
                        and fabric.global_rank == 0
                    ):
                        eval_on_india_ocean_tile(
                            pipeline.module.get_output_full,
                            output_dir=f"images/{nimg}/",
                            index_is_latlon=cfg.dit.index_is_latlon,
                            lat_radians=lat_radians,
                            lon_radians=lon_radians,
                        )

                    if cfg.training.exit_after_first_validation:
                        fabric.print("Exiting after first validation as requested.")
                        return

                    if cfg.training.max_validations is not None:
                        n_validations_done += 1
                        if n_validations_done >= cfg.training.max_validations:
                            fabric.print(
                                f"Exiting after {n_validations_done} validations as requested."
                            )
                            return

        state["epoch"] = epoch + 1

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    app = typer.Typer(pretty_exceptions_enable=False)
    app.command()(train)
    app()
