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
import torch
import torch.nn as nn


class RunningNorm2d(nn.Module):
    def __init__(
        self,
        channels: int,
        fit_batches: int,
        eps=1e-9,
        channel_groups: list[list[int]] | None = None,
    ):
        super().__init__()
        self.fit_batches = fit_batches
        self.register_buffer("mean", torch.zeros(channels, 1, 1, dtype=torch.float32))
        self.register_buffer("var", torch.ones(channels, 1, 1, dtype=torch.float32))
        self.eps = eps
        self.n_batches = 0
        self.channel_groups = channel_groups

    def update_stats(self, inputs):
        with torch.amp.autocast("cuda", enabled=False):
            inputs_fp32 = inputs.float()
            # Compute per-channel mean/var over finite (non-NaN, non-inf) entries only
            finite_mask = torch.isfinite(inputs_fp32)  # [B, C, H, W]
            safe_inputs = torch.where(
                finite_mask, inputs_fp32, torch.zeros_like(inputs_fp32)
            )
            sum_per_c = safe_inputs.sum(dim=(0, 2, 3), keepdim=True).squeeze(
                0
            )  # [C,1,1]
            cnt_per_c = finite_mask.sum(dim=(0, 2, 3), keepdim=True).squeeze(
                0
            )  # [C,1,1]
            valid = cnt_per_c > 0
            # Prepare batch mean/var, defaulting to previous stats for invalid channels
            mean = self.mean.clone()
            var = self.var.clone()
            if valid.any():
                mean_valid = sum_per_c[valid] / cnt_per_c[valid]
                mean[valid] = mean_valid
                centered = safe_inputs - mean.unsqueeze(0)  # broadcast over B,H,W
                centered = torch.where(
                    finite_mask, centered, torch.zeros_like(centered)
                )
                sumsq_per_c = (
                    (centered * centered).sum(dim=(0, 2, 3), keepdim=True).squeeze(0)
                )
                var_valid = sumsq_per_c[valid] / cnt_per_c[valid]
                var[valid] = var_valid

            if self.n_batches == 0:
                # Initialize only valid channels; leave others as existing buffers
                init_mean = torch.where(valid, mean, self.mean)
                init_var = torch.where(valid, var, self.var)
                self.mean.copy_(init_mean)
                self.var.copy_(init_var)
            else:
                # Welford-like update only for valid channels
                delta = mean - self.mean
                delta = torch.where(valid, delta, torch.zeros_like(delta))
                new_mean = torch.where(
                    valid, self.mean + delta / (self.n_batches + 1), self.mean
                )
                new_var_valid = (
                    self.var * self.n_batches
                    + torch.where(valid, var, torch.zeros_like(var))
                    + delta**2 * self.n_batches / (self.n_batches + 1)
                ) / (self.n_batches + 1)
                new_var = torch.where(valid, new_var_valid, self.var)

                self.mean.copy_(new_mean)
                self.var.copy_(new_var)

            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(
                    self.mean, op=torch.distributed.ReduceOp.AVG
                )
                torch.distributed.all_reduce(
                    self.var, op=torch.distributed.ReduceOp.AVG
                )

            # Apply channel tying if configured
            if self.channel_groups is not None:
                self._apply_channel_tying()

            self.n_batches += 1

    def denormalize(self, x):
        with torch.amp.autocast("cuda", enabled=False):
            input_dtype = x.dtype
            x_fp32 = x.float()
            denormalized = x_fp32 * (self.var.sqrt() + self.eps) + self.mean
            return denormalized.to(input_dtype)

    def forward(self, inputs):
        with torch.amp.autocast("cuda", enabled=False):
            input_dtype = inputs.dtype
            inputs_fp32 = inputs.to(torch.float32)

            if self.n_batches < self.fit_batches and self.training:
                self.update_stats(inputs_fp32)

            normalized = (inputs_fp32 - self.mean) / (self.var.sqrt() + self.eps)
            # Fill non-finite inputs (e.g., below-surface NaNs) with zeros after normalization
            finite_mask = torch.isfinite(inputs_fp32)
            normalized = torch.where(
                finite_mask, normalized, torch.zeros_like(normalized)
            )
            return normalized.to(input_dtype)

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict["n_batches"] = self.n_batches
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        self.n_batches = state_dict.pop("n_batches", 0)
        super().load_state_dict(state_dict, strict=strict)

    def apply_channel_tying_now(self):
        """
        Manually apply channel tying to current statistics.
        Useful when loading a checkpoint that already has fitted statistics.
        """
        if self.channel_groups is not None:
            self._apply_channel_tying()

    def _apply_channel_tying(self):
        """Apply channel tying after stats update."""
        for group in self.channel_groups:
            if len(group) < 2:
                continue

            group_indices = torch.tensor(group, dtype=torch.long)
            mean_avg = self.mean[group_indices].mean(dim=0, keepdim=True)
            var_avg = self.var[group_indices].mean(dim=0, keepdim=True)

            for idx in group:
                self.mean[idx] = mean_avg.squeeze(0)
                self.var[idx] = var_avg.squeeze(0)
