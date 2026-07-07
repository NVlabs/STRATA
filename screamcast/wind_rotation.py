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
"""Tile-local wind rotation.

Rotates horizontal wind components (U, V) between the geographic east/north
frame at each pixel and the tangent-plane frame at the tile center, so the
model sees winds in a locally consistent basis. Moved verbatim from
``dit_3d.py``; the math is unchanged by the physicsnemo Strata migration.
"""

import torch


def local_basis_ENR(lat, lon):
    """
    Build local east, north, radial unit vectors at (lat, lon)
    All in radians. Works with broadcasting tensors.
    """
    # radial
    R_x = torch.cos(lat) * torch.cos(lon)
    R_y = torch.cos(lat) * torch.sin(lon)
    R_z = torch.sin(lat)

    # east
    E_x = -torch.sin(lon)
    E_y = torch.cos(lon)
    E_z = torch.zeros_like(lat)

    # north
    N_x = -torch.sin(lat) * torch.cos(lon)
    N_y = -torch.sin(lat) * torch.sin(lon)
    N_z = torch.cos(lat)

    return (E_x, E_y, E_z), (N_x, N_y, N_z), (R_x, R_y, R_z)


def forward_uv_to_tile(U, V, lat, lon, lat0, lon0):
    """
    Rotate horizontal wind (U,V) at (lat,lon) into the tangent-plane
    basis defined at tile center (lat0, lon0).

    Shapes:
        U, V:          [B, D, H, W]
        lat, lon:      [B, H, W]  (will be unsqueezed to [B, 1, H, W])
        lat0, lon0:    [B, 1, 1]  (will be unsqueezed to [B, 1, 1, 1])
    All angles in radians.
    Returns:
        U_loc, V_loc: [B, D, H, W] components along (E0, N0) at tile center.
    """

    # unsqueeze D dimension for broadcasting
    lat = lat.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
    lon = lon.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
    lat0 = lat0.unsqueeze(1)  # (B, 1, 1) -> (B, 1, 1, 1)
    lon0 = lon0.unsqueeze(1)  # (B, 1, 1) -> (B, 1, 1, 1)

    # pixel basis
    (E_x, E_y, E_z), (N_x, N_y, N_z), _ = local_basis_ENR(lat, lon)

    # 3D horizontal wind at pixel
    v_x = U * E_x + V * N_x
    v_y = U * E_y + V * N_y
    v_z = U * E_z + V * N_z

    # tile-center basis
    (E0_x, E0_y, E0_z), (N0_x, N0_y, N0_z), (R0_x, R0_y, R0_z) = local_basis_ENR(
        lat0, lon0
    )

    # project v into tile-center tangent plane (orthogonal to R0)
    v_dot_R0 = v_x * R0_x + v_y * R0_y + v_z * R0_z
    vtx = v_x - v_dot_R0 * R0_x
    vty = v_y - v_dot_R0 * R0_y
    vtz = v_z - v_dot_R0 * R0_z

    # components in center's east/north
    U_loc = vtx * E0_x + vty * E0_y + vtz * E0_z
    V_loc = vtx * N0_x + vty * N0_y + vtz * N0_z

    return U_loc, V_loc


def inverse_tile_to_uv(U_loc, V_loc, lat, lon, lat0, lon0):
    """
    Inverse of forward_uv_to_tile. The forward and inverse functions are invertible.
    Given (U_loc, V_loc) in tile-center tangent frame, recover original
    (U,V) in pixel's local (east, north) frame.

    Shapes:
        U_loc, V_loc:  [B, D, H, W]
        lat, lon:      [B, H, W]  (will be unsqueezed to [B, 1, H, W])
        lat0, lon0:    [B, 1, 1]  (will be unsqueezed to [B, 1, 1, 1])
    Returns:
        U_rec, V_rec: [B, D, H, W]
    """

    # unsqueeze D dimension for broadcasting
    lat = lat.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
    lon = lon.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
    lat0 = lat0.unsqueeze(1)  # (B, 1, 1) -> (B, 1, 1, 1)
    lon0 = lon0.unsqueeze(1)  # (B, 1, 1) -> (B, 1, 1, 1)

    # tile-center basis
    (E0_x, E0_y, E0_z), (N0_x, N0_y, N0_z), (R0_x, R0_y, R0_z) = local_basis_ENR(
        lat0, lon0
    )

    # reconstruct v_tan in tile-center tangent plane
    vtx = U_loc * E0_x + V_loc * N0_x
    vty = U_loc * E0_y + V_loc * N0_y
    vtz = U_loc * E0_z + V_loc * N0_z

    # pixel radial
    _, _, (R_x, R_y, R_z) = local_basis_ENR(lat, lon)

    # solve for full v: v = v_tan + beta * R0, with constraint v·R = 0 (tangent at pixel)
    v_tan_dot_R = vtx * R_x + vty * R_y + vtz * R_z
    R0_dot_R = R0_x * R_x + R0_y * R_y + R0_z * R_z

    beta = -v_tan_dot_R / R0_dot_R

    v_x = vtx + beta * R0_x
    v_y = vty + beta * R0_y
    v_z = vtz + beta * R0_z

    # pixel basis for decomposition back to (U,V)
    (E_x, E_y, E_z), (N_x, N_y, N_z), _ = local_basis_ENR(lat, lon)

    U_rec = v_x * E_x + v_y * E_y + v_z * E_z
    V_rec = v_x * N_x + v_y * N_y + v_z * N_z

    return U_rec, V_rec
