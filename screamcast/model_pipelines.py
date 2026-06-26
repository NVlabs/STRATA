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
import abc

import torch
import torch.nn.functional as F

from screamcast.dali_ext_src import ScreamV2


class Pipeline(abc.ABC):
    plevel: int = 4

    @abc.abstractmethod
    def get_loss(self, x, y, index, **kwargs):
        pass

    @abc.abstractmethod
    def initialize(self, x):
        pass

    @abc.abstractmethod
    def step(self, state, index) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    def get_output(self, x, index):
        # for backward compatibility
        return self.get_output_prognostic(x, index)

    def get_output_full(self, x, index):
        state = self.initialize(x)
        return self.step(state, index)[0]

    def get_output_prognostic(self, x, index):
        state = self.initialize(x)
        return self.step(state, index)[1]


class MixedPredictionAsymmetric(torch.nn.Module, Pipeline):
    """
    For prognostic variables:
    - By default, use difference prediction
    - variables_prognostic_state can be used to specify variables that should use state prediction

    For diagnostic variables:
    - Always use state prediction
    """

    def __init__(
        self,
        network,
        loss_fn,
        input_norm,
        target_norm,
        plevel=4,
        level_start=3,
        level_end=128,
        variables_prognostic=(),
        variables_forcing=(),
        variables_diagnostic=(),
        variables_prognostic_state=(),
        enable_3d_adapter: bool = False,
        # Non-negativity constraints applied after network output (flat [B,C,H,W] space)
        do_qv_softplus: bool = False,
        do_precip_relu: bool = False,
        # Variables whose input channels are zeroed post-normalization (treat as diagnostic)
        variables_input_zeroed: tuple = (),
    ):
        """
        Args:
            variables_prognostic_state: tuple of prognostic variable names that should use state prediction
                                      instead of difference prediction
        """
        super().__init__()
        self.network = network
        self.loss_fn = loss_fn
        self.input_norm = input_norm
        self.target_norm = target_norm
        self.plevel = plevel
        self.level_start = level_start
        self.level_end = level_end
        self.variables_prognostic = variables_prognostic
        self.variables_forcing = variables_forcing
        self.variables_diagnostic = variables_diagnostic
        self.variables_prognostic_state = variables_prognostic_state
        self.enable_3d_adapter = enable_3d_adapter
        self.do_qv_softplus = do_qv_softplus
        self.do_precip_relu = do_precip_relu
        ranges_input = ScreamV2.ranges_input(
            variables_prognostic=self.variables_prognostic,
            variables_forcing=self.variables_forcing,
            plevel=self.plevel,
            level_start=self.level_start,
            level_end=self.level_end,
        )

        ranges_output = ScreamV2.ranges_output(
            variables_prognostic=self.variables_prognostic,
            variables_diagnostic=self.variables_diagnostic,
            plevel=self.plevel,
            level_start=self.level_start,
            level_end=self.level_end,
        )
        self.ranges_input = ranges_input
        self.ranges_output = ranges_output

        # Precompute channel indices to zero for variables_input_zeroed
        self._zeroed_input_channels: list[int] = []
        for var in variables_input_zeroed:
            sl = ranges_input.get(var)
            if sl is not None:
                self._zeroed_input_channels.extend(range(sl.start, sl.stop))

        # Validate that all state prediction variables exist in prognostic variables
        invalid_vars = set(variables_prognostic_state) - set(variables_prognostic)
        if invalid_vars:
            raise ValueError(
                f"Variables {invalid_vars} specified for state prediction but not in prognostic variables"
            )

        # Cache variables ordering for 3D adapter
        self._variables_input_order = tuple(self.variables_prognostic) + tuple(
            self.variables_forcing
        )
        self._variables_output_order = tuple(self.variables_prognostic) + tuple(
            self.variables_diagnostic
        )

        # Utility: dimensionality map ("3D"/"2D"/"1D")
        self._var_dims = ScreamV2.get_default_variables_dimensions()

        # Infer number of vertical levels from any 3D variable present; fallback to math if none
        self._num_levels = self._infer_num_levels()

    def _infer_num_levels(self) -> int:
        return len(range(self.level_start, self.level_end, self.plevel))

    def _to_3d_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert flat input [B, C_in_flat, H, W] to 3D structure [B, V_in, L, H, W].
        - 3D variables take their L channels directly into the level dim
        - 2D/1D variables are broadcast uniformly over the level dimension
        """
        if x.dim() != 4:
            raise ValueError(
                f"Expected 4D input tensor [B,C,H,W], got shape: {tuple(x.shape)}"
            )

        batch, _, height, width = x.shape
        level_count = self._num_levels
        var_tensors = []
        for var in self._variables_input_order:
            sl = self.ranges_input[var]
            var_dim = self._var_dims[var]
            if var_dim == "3D":
                # x[:, sl] -> [B, L, H, W]; insert var axis
                x_var = x[:, sl]
                if x_var.shape[1] != level_count:
                    raise RuntimeError(
                        f"Levels mismatch for var {var}: expected {level_count}, got {x_var.shape[1]}"
                    )
                x_var = x_var.unsqueeze(1)  # [B, 1, L, H, W]
            else:
                # x[:, sl] -> [B, 1, H, W]; broadcast over L
                base = x[:, sl].unsqueeze(2)  # [B, 1, 1, H, W]
                x_var = base.expand(batch, 1, level_count, height, width)
            var_tensors.append(x_var)
        return torch.cat(var_tensors, dim=1)

    def _from_3d_output(self, y3d: torch.Tensor) -> torch.Tensor:
        """
        Convert 3D network output [B, V_out, L, H, W] back to flat channels [B, C_out_flat, H, W].
        - 3D variables: keep all levels as separate channels (flatten L)
        - 2D/1D variables: average across level dim to produce a single 2D channel
        """
        if y3d.dim() != 5:
            raise ValueError(
                f"Expected 5D output tensor [B,V,L,H,W], got shape: {tuple(y3d.shape)}"
            )

        batch, num_vars, level_count, height, width = y3d.shape
        if num_vars != len(self._variables_output_order):
            raise RuntimeError(
                f"Output vars mismatch: expected {len(self._variables_output_order)} got {num_vars}"
            )

        flat_parts = []
        for i, var in enumerate(self._variables_output_order):
            var_block = y3d[:, i]  # [B, L, H, W]
            if self._var_dims[var] == "3D":
                flat_parts.append(var_block)  # contributes L channels when concatenated
            else:
                # Reduce across levels to single 2D channel
                var_2d = var_block.mean(dim=1, keepdim=True)  # [B,1,H,W]
                flat_parts.append(var_2d)
        return torch.cat(flat_parts, dim=1)

    def forward_from_flat(self, x, index):
        x_norm = self.input_norm(x)
        if self._zeroed_input_channels:
            x_norm = x_norm.clone()
            x_norm[:, self._zeroed_input_channels] = 0.0
        if not self.enable_3d_adapter:
            result = self.network(x_norm, index)
        else:
            # Per-level normalization (all variables)
            x_3d = self._to_3d_input(x_norm)  # [B, V_all, L, H, W]
            result = self._from_3d_output(self.network(x_3d, index))

        # Apply qv non-negativity constraint via ratio-based softplus.
        # ratio = dqv / max(qv, eps); constrain ratio >= -1 via softplus.
        # For positive qv: qv_new = qv * (1 + ratio_constrained) >= 0.
        # For negative qv: add -clamp(qv, max=0) to lift the delta so qv_new >= -eps ~ 0.
        # Gradient w.r.t. dqv = sigmoid(ratio + 1, beta): scale-independent, smooth.
        if self.do_qv_softplus and "qv" in self.ranges_output:
            in_sl = self.ranges_input["qv"]
            out_sl = self.ranges_output["qv"]
            with torch.amp.autocast("cuda", enabled=False):
                # mean/var have shape [C,1,1]; slice to [L,1,1], unsqueeze to [1,L,1,1]
                qv_mean = self.input_norm.mean[in_sl].unsqueeze(0)
                qv_std = (
                    self.input_norm.var[in_sl].sqrt() + self.input_norm.eps
                ).unsqueeze(0)
                dqv_mean = self.target_norm.mean[out_sl].unsqueeze(0)
                dqv_std = (
                    self.target_norm.var[out_sl].sqrt() + self.target_norm.eps
                ).unsqueeze(0)
                qv_norm_vals = x_norm[:, in_sl, :, :].float()  # [B,L,H,W]
                dqv_norm_vals = result[:, out_sl, :, :].float()  # [B,L,H,W]
                qv_phys = qv_norm_vals * qv_std + qv_mean
                dqv_phys = dqv_norm_vals * dqv_std + dqv_mean
                eps = 1e-14
                qv_phys_safe = qv_phys.clamp(min=eps)  # always positive denominator
                ratio = dqv_phys / qv_phys_safe
                ratio_constrained = F.softplus(ratio + 1.0, beta=10.0) - 1.0
                # qv_new = (1 + ratio_constrained) * qv_phys_safe >= 0 always,
                # so dqv_phys_c = qv_new - qv_phys
                dqv_phys_c = (1.0 + ratio_constrained) * qv_phys_safe - qv_phys
                dqv_norm_c = (dqv_phys_c - dqv_mean) / dqv_std
            result = result.clone()
            result[:, out_sl, :, :] = dqv_norm_c.to(result.dtype)

        # Apply precip non-negativity constraint via ReLU in physical space.
        # Denormalize -> relu -> renormalize for all present precip channels at once.
        if self.do_precip_relu:
            precip_chs = [
                i
                for var in ("precip_ice_surf_mass_flux", "precip_liq_surf_mass_flux")
                if var in self.ranges_output
                for i in range(
                    self.ranges_output[var].start, self.ranges_output[var].stop
                )
            ]
            if precip_chs:
                with torch.amp.autocast("cuda", enabled=False):
                    p_mean = self.target_norm.mean[precip_chs].unsqueeze(0)  # [1,N,1,1]
                    p_std = (
                        self.target_norm.var[precip_chs].sqrt() + self.target_norm.eps
                    ).unsqueeze(0)
                    p_norm_vals = result[:, precip_chs, :, :].float()  # [B,N,H,W]
                    p_norm_c = (F.relu(p_norm_vals * p_std + p_mean) - p_mean) / p_std
                result = result.clone()
                result[:, precip_chs, :, :] = p_norm_c.to(result.dtype)

        return result

    def get_loss(self, x, y, index, **kwargs):
        """
        Compute loss for single-step prediction.

        x: input state at time t with shape [batch, in_channels, H, W]
        y: target state at time t+1 with shape [batch, out_channels, H, W]
        index: index for the current tile, with shape [batch, H, W]
        loss_kwargs: additional keyword arguments to pass to the loss function, potentially including the following:
        mask: mask for the current tile to indicate valid pixels above surface, with shape [batch, out_channels, H, W]
        pooled_loss_weights: dictionary of pooled loss weights like {4: w4, 16: w16}
        return_details: if True, return a dictionary of named loss terms
        return_next_state: if True, also return (full_output, next_prognostic) like step() does,
                          avoiding a duplicate forward pass for multi-step training
        """
        # Optional pooled multi-scale losses: pass dict like {4: w4, 16: w16}
        pooled_loss_weights = kwargs.pop("pooled_loss_weights", None)
        # Optional: return total and a dict of named loss terms
        return_details = bool(kwargs.pop("return_details", False))
        # Optional: also return next state to avoid duplicate forward pass
        return_next_state = bool(kwargs.pop("return_next_state", False))
        # Optional: return normalized prediction and target for per-channel loss computation
        return_normalized = bool(kwargs.pop("return_normalized", False))
        pooled_loss_channel_weights = kwargs.pop("pooled_loss_channel_weights", None)
        loss_kwargs = dict(kwargs)

        pred = self.forward_from_flat(x, index)

        # Build normalized target depending on prediction mode
        if (
            not self.variables_forcing
            and not self.variables_diagnostic
            and not self.variables_prognostic_state
        ):
            target_normed = self.target_norm(y - x)
        else:
            target_diagnostic = self._extract_diagnostic_from_output(y)
            target_prognostic = self._build_targets_in_order(x, y)
            target_combined = torch.cat([target_prognostic, target_diagnostic], dim=1)
            target_normed = self.target_norm(target_combined)

        # Base loss at native resolution
        base_loss = self.loss_fn(pred, target_normed, **loss_kwargs)
        total = base_loss
        loss_terms: dict[str, torch.Tensor] = {"base": base_loss}

        # Optional pooled losses (mask-aware if mask provided)
        if pooled_loss_weights:
            pooled_base_mask = loss_kwargs.get("mask", None)
            for ksize, weight in pooled_loss_weights.items():
                if weight is None or float(weight) == 0.0:
                    continue
                k = int(ksize)
                if pooled_base_mask is not None:
                    # Mask-aware pooling: average only valid pixels; avoid NaNs
                    w = pooled_base_mask.to(dtype=pred.dtype)
                    denom = F.avg_pool2d(w, kernel_size=k, stride=k)
                    pred_clean = torch.nan_to_num(pred)
                    target_clean = torch.nan_to_num(target_normed)
                    num_p = F.avg_pool2d(pred_clean * w, kernel_size=k, stride=k)
                    num_t = F.avg_pool2d(target_clean * w, kernel_size=k, stride=k)
                    denom_safe = torch.clamp(denom, min=1e-6)
                    p_pool = num_p / denom_safe
                    t_pool = num_t / denom_safe
                    m_pool = (denom > 0).to(dtype=w.dtype)
                    pooled_kwargs = dict(loss_kwargs)
                    pooled_kwargs["mask"] = m_pool
                    pooled_term = self.loss_fn(p_pool, t_pool, **pooled_kwargs)
                else:
                    p_pool = F.avg_pool2d(pred, kernel_size=k, stride=k)
                    t_pool = F.avg_pool2d(target_normed, kernel_size=k, stride=k)

                    # Apply channel weights if provided
                    if pooled_loss_channel_weights is not None:
                        # Compute per-element loss using loss_fn with return_mean=False if supported
                        try:
                            loss_per_elem = self.loss_fn(
                                p_pool, t_pool, return_mean=False
                            )
                        except TypeError:
                            # Fallback for standard PyTorch losses that don't support return_mean
                            if isinstance(self.loss_fn, torch.nn.SmoothL1Loss):
                                loss_per_elem = F.smooth_l1_loss(
                                    p_pool, t_pool, reduction="none"
                                )
                            elif isinstance(self.loss_fn, torch.nn.MSELoss):
                                loss_per_elem = F.mse_loss(
                                    p_pool, t_pool, reduction="none"
                                )
                            else:
                                raise TypeError(
                                    f"Loss function {type(self.loss_fn)} does not support return_mean=False "
                                    "and is not a recognized standard loss for channel-weighted pooled loss"
                                )
                        # Average over batch and spatial dims to get [C]
                        loss_per_channel = loss_per_elem.mean(dim=(0, 2, 3))
                        # Apply channel weights and sum
                        pooled_term = (
                            loss_per_channel * pooled_loss_channel_weights
                        ).mean()
                    else:
                        pooled_term = self.loss_fn(p_pool, t_pool, **loss_kwargs)

                loss_terms[f"pooled_{k}"] = pooled_term
                total = total + float(weight) * pooled_term

        # Build loss result
        if return_details and pooled_loss_weights:
            loss_result = (total, loss_terms)
        else:
            loss_result = total

        # Optionally compute next state (reusing pred to avoid duplicate forward pass)
        if return_next_state:
            if (
                not self.variables_forcing
                and not self.variables_diagnostic
                and not self.variables_prognostic_state
            ):
                # Simple case: input and output are the same prognostic variables
                next_state = self.target_norm.denormalize(pred) + x
                full_output, next_prognostic = next_state, next_state
            else:
                # Mixed case: need to process different prediction types
                pred_denorm = self.target_norm.denormalize(pred)
                output_prognostic = self._process_mixed_statediff_predictions(
                    pred_denorm, x
                )
                pred_diagnostic = self._extract_diagnostic_from_output(pred_denorm)
                full_output = torch.cat([output_prognostic, pred_diagnostic], dim=1)
                next_prognostic = output_prognostic

            if return_normalized:
                return loss_result, full_output, next_prognostic, pred, target_normed
            return loss_result, full_output, next_prognostic
        elif return_normalized:
            return loss_result, pred, target_normed
        else:
            return loss_result

    def get_multistep_loss(
        self,
        inputs,
        targets_all,
        index,
        s_forcings,
        num_steps,
        multistep_training_mode,
        return_details=False,
        pooled_loss_weights=None,
        pooled_loss_channel_weights=None,
        return_final_output=False,
        return_normalized=False,
    ):
        """
        Compute loss for multi-step prediction.

        This method consolidates the multi-step loss computation logic used in both
        training and validation, avoiding code duplication.

        Args:
            inputs: Input at time t [B, C_in, H, W]
            targets_all: Targets at t+1 to t+num_steps [B, T, C_out, H, W]
            index: Tile index [B, H, W]
            s_forcings: Forcings at t+1 to t+num_steps-1 [B, T-1, C_forcing, H, W] or None
            num_steps: Number of prediction steps (must be > 1)
            multistep_training_mode: 'final_only' or 'all_steps'
                - 'final_only': Run intermediate steps with torch.no_grad(), compute loss only on final step
                - 'all_steps': Compute loss at each step with full gradients, average them
            return_details: If True, return loss terms dict (for pooled losses)
            pooled_loss_weights: Dict of pooled loss weights like {4: w4, 16: w16}
            pooled_loss_channel_weights: Tensor of shape [C] for per-channel weighting of pooled loss
            return_final_output: If True, also return final output_full for MAE computation (used in validation)
            return_normalized: If True, also return normalized pred and target (only for final_only mode)

        Returns:
            If return_final_output is False:
                loss_result: Loss value or (loss, loss_terms) tuple
            If return_final_output is True:
                (loss_result, output_full): Loss and final prediction output
            If return_final_output and return_normalized:
                (loss_result, output_full, pred_normed, target_normed)
        """
        if num_steps <= 1:
            raise ValueError(
                f"get_multistep_loss requires num_steps > 1, got {num_steps}"
            )

        if multistep_training_mode == "final_only":
            # Run intermediate steps with no_grad, compute loss only on final.
            # Force eager on this no-grad rollout: combining activation
            # checkpointing + torch.compile + multistep finetuning raises an
            # error (root cause not yet isolated), and dropping compile on
            # this block sidesteps it. Alternatives considered:
            # (a) turn off activation checkpointing — loses memory savings
            #     needed for larger global batch sizes,
            # (b) turn off torch.compile globally — loses training speedup,
            # (c) narrow the force_eager scope further — not tried yet,
            # (d) file an upstream PyTorch bug with a minimal repro — TODO.
            # Trade-off: this block runs uncompiled, so rollouts here are
            # slower than the compiled final step; acceptable because it is
            # only the intermediate no_grad portion.
            # TODO: revisit once the AC + compile + multistep interaction is
            # understood / fixed upstream, and remove this stance override.
            with torch.compiler.set_stance("force_eager"), torch.no_grad():
                current_prognostic = self._extract_prognostic_from_input(inputs)
                for step_idx in range(num_steps - 1):
                    # Build input for this step
                    if self.variables_forcing and s_forcings is not None:
                        step_input = torch.cat(
                            [current_prognostic, s_forcings[:, step_idx]], dim=1
                        )
                    else:
                        step_input = current_prognostic
                    # Run prediction
                    state = self.initialize(step_input)
                    _, current_prognostic = self.step(state, index)

            # Final step WITH gradients
            if self.variables_forcing and s_forcings is not None:
                final_input = torch.cat([current_prognostic, s_forcings[:, -1]], dim=1)
            else:
                final_input = current_prognostic
            final_target = targets_all[:, -1]  # [B, C, H, W]

            if return_final_output:
                result = self.get_loss(
                    final_input,
                    final_target,
                    index,
                    return_details=return_details,
                    pooled_loss_weights=pooled_loss_weights,
                    pooled_loss_channel_weights=pooled_loss_channel_weights,
                    return_next_state=True,
                    return_normalized=return_normalized,
                )
                if return_normalized:
                    loss_result, output_full, _, pred_normed, target_normed = result
                    return loss_result, output_full, pred_normed, target_normed
                else:
                    loss_result, output_full, _ = result
                    return loss_result, output_full
            else:
                loss_result = self.get_loss(
                    final_input,
                    final_target,
                    index,
                    return_details=return_details,
                    pooled_loss_weights=pooled_loss_weights,
                    pooled_loss_channel_weights=pooled_loss_channel_weights,
                )
                return loss_result

        elif multistep_training_mode == "all_steps":
            # Compute loss at each step, average them
            total_loss = 0.0
            loss_terms_accum = {}
            current_prognostic = self._extract_prognostic_from_input(inputs)
            output_full = None  # Will be set on final step if return_final_output
            pred_normed, target_normed = (
                None,
                None,
            )  # Will be set on final step if return_normalized

            for step_idx in range(num_steps):
                # Build input for this step
                if step_idx == 0:
                    step_input = inputs  # First step uses original input with forcing
                else:
                    if self.variables_forcing and s_forcings is not None:
                        step_input = torch.cat(
                            [current_prognostic, s_forcings[:, step_idx - 1]], dim=1
                        )
                    else:
                        step_input = current_prognostic

                # Get target for this step
                step_target = targets_all[:, step_idx]  # [B, C, H, W]

                # Compute loss and get next state in one forward pass
                is_last_step = step_idx == num_steps - 1
                need_next_state = (not is_last_step) or return_final_output
                # Only return normalized on last step
                step_return_normalized = is_last_step and return_normalized

                step_result = self.get_loss(
                    step_input,
                    step_target,
                    index,
                    return_details=return_details,
                    pooled_loss_weights=pooled_loss_weights,
                    pooled_loss_channel_weights=pooled_loss_channel_weights,
                    return_next_state=need_next_state,
                    return_normalized=step_return_normalized,
                )

                # Parse result based on whether we requested next state and/or normalized
                if need_next_state and step_return_normalized:
                    (
                        step_loss_result,
                        step_output_full,
                        current_prognostic,
                        pred_normed,
                        target_normed,
                    ) = step_result
                    if is_last_step and return_final_output:
                        output_full = step_output_full
                elif need_next_state:
                    step_loss_result, step_output_full, current_prognostic = step_result
                    if is_last_step and return_final_output:
                        output_full = step_output_full
                elif step_return_normalized:
                    step_loss_result, pred_normed, target_normed = step_result
                else:
                    step_loss_result = step_result

                # Accumulate loss
                if isinstance(step_loss_result, tuple):
                    step_loss, step_terms = step_loss_result
                    for k, v in step_terms.items():
                        loss_terms_accum[k] = loss_terms_accum.get(k, 0.0) + v
                else:
                    step_loss = step_loss_result
                total_loss = total_loss + step_loss

            # Average the losses across steps
            total_loss = total_loss / num_steps
            if loss_terms_accum:
                for k in loss_terms_accum:
                    loss_terms_accum[k] = loss_terms_accum[k] / num_steps
                loss_result = (total_loss, loss_terms_accum)
            else:
                loss_result = total_loss

            if return_final_output:
                if return_normalized:
                    return loss_result, output_full, pred_normed, target_normed
                return loss_result, output_full
            else:
                return loss_result

        else:
            raise ValueError(
                f"Unknown multistep_training_mode: {multistep_training_mode}. "
                "Expected 'final_only' or 'all_steps'."
            )

    def initialize(self, x):
        """
        x: initial state with shape [batch, in_channels, H, W]
        """
        return x

    def step(self, state, index):
        """
        state: current state with shape [batch, in_channels, H, W]
        Returns: (full_output, next_prognostic). full_output contains both prognostic and diagnostic variables.
                next_prognostic contains only prognostic variables.
        """
        if (
            not self.variables_forcing
            and not self.variables_diagnostic
            and not self.variables_prognostic_state
        ):
            # in this case, the input and output are the same set of prognostic variables and use difference prediction
            pred = self.forward_from_flat(state, index)
            next_state = self.target_norm.denormalize(pred) + state
            return next_state, next_state
        else:
            pred = self.forward_from_flat(state, index)
            pred_denorm = self.target_norm.denormalize(pred)
            # construct the next step's prognostic variables state based on the prediction and previous state
            output_prognostic = self._process_mixed_statediff_predictions(
                pred_denorm, state
            )
            # enforce the floor directly in physical space. This path is not reached during training and is purely for rollout.
            if self.do_qv_softplus and "qv" in self.ranges_output:
                qv_sl = self.ranges_output["qv"]
                output_prognostic = output_prognostic.clone()
                output_prognostic[:, qv_sl] = output_prognostic[:, qv_sl].clamp(
                    min=1e-15
                )
            pred_diagnostic = self._extract_diagnostic_from_output(pred_denorm)
            full_output = torch.cat([output_prognostic, pred_diagnostic], dim=1)
            return full_output, output_prognostic

    def _process_mixed_statediff_predictions(self, pred_denorm, state):
        """Process network predictions and return the next step's prognostic variables state, applying appropriate prediction type"""
        next_prognostic_parts = []
        for var in self.variables_prognostic:
            var_pred = pred_denorm[:, self.ranges_output[var]]
            # check if prediction type is state or difference
            if var in self.variables_prognostic_state:
                next_var_state = var_pred
            else:
                current_var_state = state[:, self.ranges_input[var]]
                next_var_state = var_pred + current_var_state
            next_prognostic_parts.append(next_var_state)
        return torch.cat(next_prognostic_parts, dim=1)

    def _build_targets_in_order(self, x, y):
        """Build target tensor in original variable order with appropriate target types"""
        target_parts = []

        for var in self.variables_prognostic:
            # Get current state for this variable
            current_var_state = x[:, self.ranges_input[var]]

            # Get target state for this variable
            target_var_state = y[:, self.ranges_output[var]]

            # Build appropriate target based on prediction type
            if var in self.variables_prognostic_state:
                # State prediction: target is the absolute state
                var_target = target_var_state
            else:
                # Difference prediction: target is the difference
                var_target = target_var_state - current_var_state

            target_parts.append(var_target)

        return torch.cat(target_parts, dim=1)

    def _extract_prognostic_from_input(self, x):
        """Extract prognostic variables from input tensor"""
        if not self.variables_forcing:
            return x

        ranges_input = ScreamV2.ranges_input(
            variables_prognostic=self.variables_prognostic,
            variables_forcing=self.variables_forcing,
            plevel=self.plevel,
            level_start=self.level_start,
            level_end=self.level_end,
        )
        prognostic_parts = []
        for var in self.variables_prognostic:
            input_slice = ranges_input[var]
            prognostic_parts.append(x[:, input_slice])

        return torch.cat(prognostic_parts, dim=1)

    def _extract_prognostic_from_output(self, x):
        """Extract prognostic variables from output tensor"""
        if not self.variables_diagnostic:
            return x

        ranges_output = ScreamV2.ranges_output(
            variables_prognostic=self.variables_prognostic,
            variables_diagnostic=self.variables_diagnostic,
            plevel=self.plevel,
            level_start=self.level_start,
            level_end=self.level_end,
        )
        prognostic_parts = []
        for var in self.variables_prognostic:
            output_slice = ranges_output[var]
            prognostic_parts.append(x[:, output_slice])
        return torch.cat(prognostic_parts, dim=1)

    def _extract_diagnostic_from_output(self, x):
        """Extract diagnostic variables from output tensor"""
        if not self.variables_diagnostic:
            return torch.empty(
                x.shape[0], 0, x.shape[2], x.shape[3], dtype=x.dtype, device=x.device
            )

        ranges_output = ScreamV2.ranges_output(
            variables_prognostic=self.variables_prognostic,
            variables_diagnostic=self.variables_diagnostic,
            plevel=self.plevel,
            level_start=self.level_start,
            level_end=self.level_end,
        )
        diagnostic_parts = []
        for var in self.variables_diagnostic:
            output_slice = ranges_output[var]
            diagnostic_parts.append(x[:, output_slice])

        # Handle case where there are no valid diagnostic variables
        if len(diagnostic_parts) == 0:
            # Return empty tensor with correct shape [batch, 0, H, W]
            return torch.empty(
                x.shape[0], 0, x.shape[2], x.shape[3], dtype=x.dtype, device=x.device
            )

        return torch.cat(diagnostic_parts, dim=1)

    def load_checkpoint(self, checkpoint_data):
        self.network.load_state_dict(checkpoint_data["network"])
        self.input_norm.load_state_dict(checkpoint_data["input_norm"])
        self.target_norm.load_state_dict(checkpoint_data["target_norm"])
        self.loss_fn.load_state_dict(checkpoint_data["loss_fn"])
