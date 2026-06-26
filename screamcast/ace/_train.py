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
"""Training entrypoints for ACE forecast-residual fine-tuning."""

from __future__ import annotations

import argparse
import datetime

import numpy as np
import torch
import torch.nn as nn
from earth2studio.models.px import ace2
from torch.utils.data import DataLoader, TensorDataset

from screamcast.ace._channels import P0_SCREAM
from screamcast.ace._finetune_utils import (
    TINY_RESIDUAL_SCALE_FLOOR,
    atomic_torch_save,
    build_coords_payload,
    build_embedded_ace_payload,
    load_ace_backbone,
    load_ace_out_names,
)
from screamcast.ace._forcing import build_ace_forcing_tensor, fetch_static_ace_forcing
from screamcast.ace._residual_model import (
    ACE2ForecastResidualSFNO,
    load_training_tensors,
)
from screamcast.history import git_commit


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    (
        x,
        y,
        lat,
        lon,
        hyam_sub,
        hybm_sub,
        _valid_time_index_unused,
        scream_names,
        valid_times,
    ) = load_training_tensors(
        data_path=args.data,
    )

    ace_checkpoint = (
        args.ace_checkpoint
        or ace2.ACE2ERA5.load_default_package().resolve("ace2_era5_ckpt.tar")
    )
    backbone, input_layout, ace_means, ace_stds = load_ace_backbone(ace_checkpoint)
    ace_out_names = load_ace_out_names(ace_checkpoint)

    static_forcing = fetch_static_ace_forcing(
        time=valid_times[0],
        forcing_names=input_layout.forcing_fme_names,
        lat=lat.numpy(),
        lon=lon.numpy(),
    )
    forcing = torch.stack(
        [
            build_ace_forcing_tensor(
                time=vt,
                lat=lat.numpy(),
                lon=lon.numpy(),
                forcing_names=input_layout.forcing_fme_names,
                static_forcing=static_forcing,
                batch_size=1,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )[0]
            for vt in valid_times
        ],
        dim=0,
    )

    n_times = x.shape[0]
    n_train = max(1, int(n_times * 0.8))
    x_train, x_val = x[:n_train], x[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    f_train, f_val = forcing[:n_train], forcing[n_train:]

    residual_scale_raw = (y_train - x_train).std(dim=(0, 2, 3))
    residual_scale = residual_scale_raw.clamp(min=TINY_RESIDUAL_SCALE_FLOOR)
    residual_scale_b = residual_scale.view(1, -1, 1, 1)

    train_loader = DataLoader(
        TensorDataset(x_train, f_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(x_val, f_val, y_val),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = ACE2ForecastResidualSFNO(
        backbone=backbone,
        input_layout=input_layout,
        normalizer_means=ace_means,
        normalizer_stds=ace_stds,
        ace_out_names=ace_out_names,
        scream_variable_names=scream_names,
        scream_hyam_sub=hyam_sub,
        scream_hybm_sub=hybm_sub,
        freeze_backbone=not args.train_backbone,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs)
    )
    residual_scale_d = residual_scale_b.to(device)

    def _loss_on_loader(loader: DataLoader[tuple[torch.Tensor, ...]]) -> float:
        model.eval()
        total = 0.0
        with torch.no_grad():
            for xb, fb, yb in loader:
                xb = xb.to(device)
                fb = fb.to(device)
                yb = yb.to(device)
                correction = model(xb, fb)
                target = (yb - xb) / residual_scale_d
                total += criterion(correction, target).item() * xb.size(0)
        return total / max(1, len(loader.dataset))

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, fb, yb in train_loader:
            xb = xb.to(device)
            fb = fb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            correction = model(xb, fb)
            target = (yb - xb) / residual_scale_d
            loss = criterion(correction, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= max(1, len(train_loader.dataset))
        val_loss = _loss_on_loader(val_loader)
        scheduler.step()
        print(
            f"Epoch {epoch:3d}/{args.epochs} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )
        best_val = min(best_val, val_loss)
        if epoch % args.save_every == 0 or epoch == args.epochs:
            payload: dict[str, object] = {
                "model_state": model.state_dict(),
                "residual_scale": residual_scale,
                "scale_definition": "std(truth_valid - forecast_valid) over train split",
                "scream_input_variable_names": scream_names,
                "scream_variable_names": scream_names,
                "ace_forcing_names": list(input_layout.forcing_fme_names),
                "embedded_ace": build_embedded_ace_payload(
                    backbone=model.backbone,
                    input_layout=input_layout,
                    normalizer_means=ace_means,
                    normalizer_stds=ace_stds,
                    ace_out_names=ace_out_names,
                ),
                "coords": build_coords_payload(
                    lat=lat,
                    lon=lon,
                    hyam_sub=hyam_sub,
                    hybm_sub=hybm_sub,
                ),
                "static_ace_forcing": static_forcing.cpu(),
                "train_args": vars(args),
                "val_loss": best_val,
                "epoch": epoch,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "git_commit": git_commit(),
                "objective": "forecast_residual_on_ace_grid",
                "forecast_valid_time_unix_s": torch.as_tensor(
                    np.asarray(valid_times, dtype="datetime64[s]").astype(np.int64),
                    dtype=torch.int64,
                ),
                "p0_scream": P0_SCREAM,
            }
            atomic_torch_save(payload, args.output)
