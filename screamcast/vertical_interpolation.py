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
import math

import torch


def hybrid_to_pressure(
    ak: torch.Tensor,
    bk: torch.Tensor,
    surface_pressure: torch.Tensor,
    p0: float,
) -> torch.Tensor:
    return ak * p0 + bk * surface_pressure


def hybrid_midpoint_pressure(
    ak_interface: torch.Tensor,
    bk_interface: torch.Tensor,
    surface_pressure: torch.Tensor,
    p0: float = 1.0,
) -> torch.Tensor:
    """Compute midpoint pressure for a hybrid vertical coordinate.

    Args:
        ak_interface: Interface `a_k` coefficients with shape `[n_level + 1]`.
        bk_interface: Interface `b_k` coefficients with shape `[n_level + 1]`.
        surface_pressure: Surface pressure with shape `[*spatial]`.
        p0: Reference pressure scalar.

    Returns:
        Midpoint pressure with shape `[n_level, *spatial]`.
    """
    surface_pressure_t = surface_pressure
    ak_t = ak_interface
    bk_t = bk_interface
    if ak_t.ndim != 1 or bk_t.ndim != 1:
        raise ValueError("Expected 1D hybrid interface coefficient arrays.")
    if ak_t.shape != bk_t.shape:
        raise ValueError(
            f"Mismatched hybrid coefficient shapes: {ak_t.shape} vs {bk_t.shape}"
        )
    if ak_t.numel() < 2:
        raise ValueError("Need at least two interface levels.")
    ak_mid = 0.5 * (ak_t[:-1] + ak_t[1:])
    bk_mid = 0.5 * (bk_t[:-1] + bk_t[1:])
    reshape = (ak_mid.shape[0],) + (1,) * surface_pressure_t.ndim
    return hybrid_to_pressure(
        ak=ak_mid.reshape(reshape),
        bk=bk_mid.reshape(reshape),
        surface_pressure=surface_pressure_t,
        p0=p0,
    )


def interpolate_1d(
    src_coordinate: torch.Tensor,
    src_values: torch.Tensor,
    target_coordinate: torch.Tensor,
) -> torch.Tensor:
    """Interpolate along a sorted 1D coordinate with endpoint clamping.

    Args:
        src_coordinate: Sorted source coordinates with shape `[n_point, n_col]`.
        src_values: Source values with shape `[n_point, n_col]`.
        target_coordinate: Target coordinates with shape `[n_target, n_col]`.

    Returns:
        Interpolated values with shape `[n_target, n_col]`.
    """
    if src_coordinate.shape != src_values.shape:
        raise ValueError(
            f"src_coordinate and src_values must match shapes, got {tuple(src_coordinate.shape)} vs {tuple(src_values.shape)}"
        )
    if src_coordinate.ndim != 2 or target_coordinate.ndim != 2:
        raise ValueError("interpolate_1d expects 2D [n_point, n_col] tensors.")
    if src_coordinate.shape[1] != target_coordinate.shape[1]:
        raise ValueError(
            f"Column mismatch: {src_coordinate.shape[1]} vs {target_coordinate.shape[1]}"
        )

    src_coord_cols = src_coordinate.T.contiguous()
    src_value_cols = src_values.T
    target_coord_cols = target_coordinate.T.contiguous()

    idx = torch.searchsorted(src_coord_cols, target_coord_cols)
    left_idx = torch.clamp(idx - 1, min=0, max=src_coord_cols.shape[1] - 1)
    right_idx = torch.clamp(idx, min=0, max=src_coord_cols.shape[1] - 1)

    x0 = torch.gather(src_coord_cols, 1, left_idx)
    x1 = torch.gather(src_coord_cols, 1, right_idx)
    y0 = torch.gather(src_value_cols, 1, left_idx)
    y1 = torch.gather(src_value_cols, 1, right_idx)
    denom = torch.where((x1 - x0) == 0, torch.ones_like(x1), x1 - x0)
    w = (target_coord_cols - x0) / denom
    interp_cols = y0 + w * (y1 - y0)

    first = src_value_cols[:, :1].expand_as(interp_cols)
    last = src_value_cols[:, -1:].expand_as(interp_cols)
    interp_cols = torch.where(idx <= 0, first, interp_cols)
    interp_cols = torch.where(idx >= src_coord_cols.shape[1], last, interp_cols)
    return interp_cols.T


def log_pressure_interpolate(
    src_values: torch.Tensor,
    src_pressure: torch.Tensor,
    target_pressure: torch.Tensor,
    axis: int = 0,
) -> torch.Tensor:
    """Interpolate values onto a target pressure grid in log-pressure space.

    Expected layout is level-first along `axis`. After moving `axis` to the
    front, `src_values` and `src_pressure` must both have shape
    `[n_src_level, *spatial]`, and `target_pressure` must have shape
    `[n_target_level, *spatial]`.

    Returns:
        Interpolated values with shape `[n_target_level, *spatial]` on the
        moved-axis view, then moved back to the original `axis`.
    """
    if src_values.shape != src_pressure.shape:
        raise ValueError(
            f"src_values and src_pressure must match shapes, got {tuple(src_values.shape)} vs {tuple(src_pressure.shape)}"
        )
    src_values_t = torch.movedim(src_values, axis, 0)
    src_pressure_t = torch.movedim(src_pressure, axis, 0)
    target_pressure_t = torch.movedim(target_pressure, axis, 0)
    if src_values_t.shape[1:] != target_pressure_t.shape[1:]:
        raise ValueError(
            f"Target grid mismatch after moving axis: {tuple(src_values_t.shape[1:])} vs {tuple(target_pressure_t.shape[1:])}"
        )

    n_target = target_pressure_t.shape[0]
    n_cols = math.prod(target_pressure_t.shape[1:])
    src_values_flat = src_values_t.reshape(src_values_t.shape[0], n_cols)
    src_pressure_flat = src_pressure_t.reshape(src_pressure_t.shape[0], n_cols)
    target_pressure_flat = target_pressure_t.reshape(n_target, n_cols)
    tiny = torch.finfo(src_values_t.dtype).tiny

    finite_mask = (
        torch.isfinite(src_pressure_flat)
        & torch.isfinite(src_values_flat)
        & (src_pressure_flat > 0.0)
    )
    finite_count = finite_mask.sum(dim=0)
    all_valid = bool(torch.all(finite_mask))
    enough_levels = bool(torch.all(finite_count >= 2))
    if not all_valid:
        raise ValueError(
            "log_pressure_interpolate expects finite src_values and strictly positive finite src_pressure."
        )
    if not enough_levels:
        raise ValueError(
            "log_pressure_interpolate expects at least two valid source levels in every column."
        )

    lp_flat = torch.log(torch.clamp(src_pressure_flat, min=tiny))
    order = torch.argsort(lp_flat, dim=0)
    lp_sorted = torch.gather(lp_flat, 0, order)
    v_sorted = torch.gather(src_values_flat, 0, order)
    lt_flat = torch.log(torch.clamp(target_pressure_flat, min=tiny))

    out = interpolate_1d(
        src_coordinate=lp_sorted,
        src_values=v_sorted,
        target_coordinate=lt_flat,
    )

    out = out.reshape(target_pressure_t.shape)
    return torch.movedim(out, 0, axis)


def regrid_hybrid_vertical(
    src_values: torch.Tensor,
    *,
    src_ak_interface: torch.Tensor,
    src_bk_interface: torch.Tensor,
    surface_pressure: torch.Tensor,
    src_p0: float = 1.0,
    target_ak_interface: torch.Tensor,
    target_bk_interface: torch.Tensor,
    target_p0: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Regrid a level-first field between hybrid vertical coordinates.

    Args:
        src_values: Source field with shape `[n_src_level, *spatial]`.
        src_ak_interface: Source interface `a_k` coefficients with shape
            `[n_src_level + 1]` or midpoint coefficients with shape
            `[n_src_level]`.
        src_bk_interface: Source interface `b_k` coefficients with shape
            `[n_src_level + 1]` or midpoint coefficients with shape
            `[n_src_level]`.
        surface_pressure: Surface pressure with shape `[*spatial]`.
        src_p0: Source reference pressure scalar.
        target_ak_interface: Target interface `a_k` coefficients with shape
            `[n_target_level + 1]`.
        target_bk_interface: Target interface `b_k` coefficients with shape
            `[n_target_level + 1]`.
        target_p0: Target reference pressure scalar.

    Returns:
        A tuple `(remapped, target_pressure)` where both tensors have shape
        `[n_target_level, *spatial]`.
    """
    src_vals = src_values
    src_levels = int(src_vals.shape[0])
    src_ak = src_ak_interface
    src_bk = src_bk_interface
    if src_ak.shape != src_bk.shape or src_ak.ndim != 1:
        raise ValueError(
            f"Expected 1D source coefficients with matching shapes, got {src_ak.shape} and {src_bk.shape}."
        )
    if src_ak.numel() == src_levels + 1:
        src_ak_mid = 0.5 * (src_ak[:-1] + src_ak[1:])
        src_bk_mid = 0.5 * (src_bk[:-1] + src_bk[1:])
    elif src_ak.numel() == src_levels:
        src_ak_mid = src_ak
        src_bk_mid = src_bk
    else:
        raise ValueError(
            f"Source coefficient length mismatch: got {src_ak.numel()}, expected {src_levels} (midpoint) or {src_levels + 1} (interface)."
        )

    surface_pressure_t = surface_pressure
    reshape = (src_levels,) + (1,) * surface_pressure_t.ndim
    src_pressure = hybrid_to_pressure(
        ak=src_ak_mid.reshape(reshape),
        bk=src_bk_mid.reshape(reshape),
        surface_pressure=surface_pressure_t,
        p0=src_p0,
    )

    target_pressure = hybrid_midpoint_pressure(
        ak_interface=target_ak_interface,
        bk_interface=target_bk_interface,
        surface_pressure=surface_pressure,
        p0=target_p0,
    )
    if src_vals.shape[1:] != src_pressure.shape[1:]:
        raise ValueError(
            f"src_values and src_pressure spatial shape mismatch: {src_vals.shape[1:]} vs {src_pressure.shape[1:]}"
        )
    if src_vals.shape[0] != src_pressure.shape[0]:
        raise ValueError(
            f"Vertical level mismatch: values have {src_vals.shape[0]} levels, pressure has {src_pressure.shape[0]}"
        )
    remapped = log_pressure_interpolate(
        src_values=src_vals,
        src_pressure=src_pressure,
        target_pressure=target_pressure,
        axis=0,
    )
    return remapped, target_pressure
