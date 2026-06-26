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
# Architecture inspired by DiT (Peebles & Xie, 2023) and NVIDIA Makani;
# implemented independently here for 3D atmospheric fields (neighborhood
# attention NA3D + stereographic RoPE).
import math
from functools import partial
from typing import Mapping, Optional, Tuple

import earth2grid
import earth2grid.spatial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
from einops import rearrange
from natten.functional import na3d
from torch.utils.checkpoint import checkpoint as activation_checkpoint


def stereographic_projection(
    lat: torch.Tensor, lon: torch.Tensor, lat0: torch.Tensor, lon0: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Stereographic projection from lat/lon to a local tangent plane.

    Args:
        lat: latitude in radians [..., H, W]
        lon: longitude in radians [..., H, W]
        lat0: center latitude in radians [..., 1, 1]
        lon0: center longitude in radians [..., 1, 1]

    Returns:
        x, y: projected coordinates on tangent plane, with y pointing toward North
              and x pointing toward East
    """
    # Compute differences
    dlon = lon - lon0

    # Stereographic projection formulas
    # k is the scale factor
    cos_c = torch.sin(lat0) * torch.sin(lat) + torch.cos(lat0) * torch.cos(
        lat
    ) * torch.cos(dlon)
    k = 2.0 / (1.0 + cos_c)

    # x points East, y points North in the local tangent plane
    x = k * torch.cos(lat) * torch.sin(dlon)
    y = k * (
        torch.cos(lat0) * torch.sin(lat)
        - torch.sin(lat0) * torch.cos(lat) * torch.cos(dlon)
    )

    return x, y


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


def mean_longitude(
    lon: torch.Tensor,
    reduce_dims=(-2, -1),
    return_0_2pi: bool = True,
) -> torch.Tensor:
    """
    Compute circular mean of longitude, batched, handling 0/2π or -π/π wrapping.

    Args:
        lon: longitude in radians ([..., H, W])
        reduce_dims: which dimensions to average over (default: last two -> H, W)
        return_0_2pi:
            - True  -> output in [0, 2π)
            - False -> output in [-π, π)

    Returns:
        lon_mean: mean longitude ([..., 1, 1])
    """
    # Circular mean: average sin and cos
    sin_mean = lon.sin().mean(dim=reduce_dims, keepdim=True)
    cos_mean = lon.cos().mean(dim=reduce_dims, keepdim=True)

    # atan2 gives angle in [-π, π)
    lon_mean = torch.atan2(sin_mean, cos_mean)

    if return_0_2pi:
        lon_mean = lon_mean % (2 * torch.pi)

    return lon_mean


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        output_bias=True,
        input_format="nchw",
        drop_rate=0.0,
        drop_type="iid",
        checkpointing=0,
        gain=1.0,
        **kwargs,
    ):
        super(MLP, self).__init__()
        self.checkpointing = checkpointing
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # First fully connected layer
        if input_format == "nchw":
            fc1 = nn.Conv2d(in_features, hidden_features, 1, bias=True)
            fc1.weight.is_shared_mp = ["spatial"]
            fc1.bias.is_shared_mp = ["spatial"]
        elif input_format == "traditional":
            fc1 = nn.Linear(in_features, hidden_features, bias=True)
        else:
            raise NotImplementedError(
                f"Error, input format {input_format} not supported."
            )

        # initialize the weights correctly
        scale = math.sqrt(2.0 / in_features)
        nn.init.normal_(fc1.weight, mean=0.0, std=scale)
        nn.init.constant_(fc1.bias, 0.0)

        # activation
        act = act_layer()

        # sanity checks
        if (input_format == "traditional") and (drop_type == "features"):
            raise NotImplementedError(
                "Error, traditional input format and feature dropout cannot be selected simultaneously"
            )

        # output layer
        if input_format == "nchw":
            fc2 = nn.Conv2d(hidden_features, out_features, 1, bias=output_bias)
            fc2.weight.is_shared_mp = ["spatial"]
            if output_bias:
                fc2.bias.is_shared_mp = ["spatial"]
        elif input_format == "traditional":
            fc2 = nn.Linear(hidden_features, out_features, bias=output_bias)
        else:
            raise NotImplementedError(
                f"Error, input format {input_format} not supported."
            )

        # gain factor for the output determines the scaling of the output init
        scale = math.sqrt(gain / hidden_features)
        nn.init.normal_(fc2.weight, mean=0.0, std=scale)
        if fc2.bias is not None:
            nn.init.constant_(fc2.bias, 0.0)

        if drop_rate > 0.0:
            if drop_type == "iid":
                drop = nn.Dropout(drop_rate)
            elif drop_type == "features":
                drop = nn.Dropout2d(drop_rate)
            else:
                raise NotImplementedError(f"Error, drop_type {drop_type} not supported")
        else:
            drop = nn.Identity()

        # create forward pass
        self.fwd = nn.Sequential(fc1, act, drop, fc2, drop)

    @torch.jit.ignore
    def checkpoint_forward(self, x):
        raise NotImplementedError("Activation checkpointing not implemented")
        # return checkpoint(self.fwd, x, use_reentrant=False)

    def forward(self, x):
        if self.checkpointing >= 2:
            return self.checkpoint_forward(x)
        else:
            return self.fwd(x)


def _precompute_rope_2d_cos_sin(
    ij_coords: torch.Tensor, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute cos/sin lookup tables for 2D RoPE.

    Args:
        ij_coords: [N, 2] or [B, N, 2] integer/float grid coordinates
        head_dim: dimension per attention head (must be divisible by 4)

    Returns:
        cos_all, sin_all: each [1, 1, N, head_dim//2] or [B, 1, N, head_dim//2]
    """
    D = head_dim
    pairs_total = D // 2
    if pairs_total % 2 != 0:
        raise ValueError(f"2D RoPE requires head_dim divisible by 4; got head_dim={D}")
    p = pairs_total // 2

    ij = ij_coords.float()

    if ij.ndim == 2:
        # Unbatched: [N, 2]
        i_vals = ij[:, 0].unsqueeze(0)  # [1, N]
        j_vals = ij[:, 1].unsqueeze(0)  # [1, N]
    else:
        # Batched: [B, N, 2]
        i_vals = ij[:, :, 0]  # [B, N]
        j_vals = ij[:, :, 1]  # [B, N]

    exponents = torch.arange(p, device=ij_coords.device, dtype=torch.float32)
    freq = 100.0 ** (-exponents / float(p))  # [p]

    theta_i = i_vals.unsqueeze(1).unsqueeze(-1) * freq  # [*, 1, N, p]
    theta_j = j_vals.unsqueeze(1).unsqueeze(-1) * freq  # [*, 1, N, p]

    cos_all = torch.cat([torch.cos(theta_i), torch.cos(theta_j)], dim=-1)
    sin_all = torch.cat([torch.sin(theta_i), torch.sin(theta_j)], dim=-1)

    return cos_all, sin_all


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        input_format="traditional",
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop_rate=0.0,
        proj_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        qk_norm_layer=nn.RMSNorm,
        attn_mask=None,
        activation_checkpointing: bool = False,
        attn_kernel: int = -1,
        do_depthwise_attention: bool = False,
        na_dilations: int = 1,
        gated_attention: bool = False,
        # NA3D backend; None uses natten default. Options: "cutlass-fna", "hopper-fna",
        # "blackwell-fna", "flex-fna"
        na3d_backend: Optional[str] = None,
    ):
        super().__init__()
        self.activation_checkpointing = activation_checkpointing
        if dim % num_heads != 0:
            raise ValueError("dim should be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = qk_norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = qk_norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop_rate = attn_drop_rate

        self.proj = nn.Linear(dim, dim)
        self.attn_mask = attn_mask
        self.attn_kernel = attn_kernel
        self.do_depthwise_attention = do_depthwise_attention

        if proj_drop_rate > 0:
            self.proj_drop = nn.Dropout(proj_drop_rate)
        else:
            self.proj_drop = nn.Identity()

        self.na_dilations = na_dilations
        self.na3d_backend = na3d_backend
        self.gated_attention = gated_attention
        self.gated_attention_map = (
            nn.Linear(dim, dim) if gated_attention else nn.Identity()
        )

    def _apply_rope_2d(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        rope_cos_sin: tuple[torch.Tensor, torch.Tensor],
    ):
        """
        Apply 2D RoPE using precomputed cos/sin lookup tables.
        q, k: [B, heads, N, head_dim]
        rope_cos_sin: tuple of (cos, sin) each broadcastable to [B, 1, N, head_dim//2]
        """
        cos, sin = rope_cos_sin
        # Cast LUT to match q's dtype (e.g. bfloat16 under autocast)
        cos = cos.to(q.dtype)
        sin = sin.to(q.dtype)

        # Slice even/odd indices as zero-copy views (avoids 5D rearrange)
        q_a, q_b = q[..., 0::2], q[..., 1::2]
        k_a, k_b = k[..., 0::2], k[..., 1::2]

        # Apply rotation and interleave back
        q_out = torch.stack(
            [q_a * cos - q_b * sin, q_a * sin + q_b * cos], dim=-1
        ).view_as(q)
        k_out = torch.stack(
            [k_a * cos - k_b * sin, k_a * sin + k_b * cos], dim=-1
        ).view_as(k)

        return q_out, k_out

    def forward(
        self,
        x,
        per_batch_attn_mask=None,
        latent_dhw=None,
        rope_cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        """
        Args:
            x: (n, tokens, c)
            per_batch_attn_mask: Optional attention mask
            latent_dhw: Tuple of (depth, height, width) dimensions for reshaping
            rope_cos_sin: Precomputed (cos, sin) tables for 2D RoPE, or None
        """

        if per_batch_attn_mask is not None:
            if self.attn_mask is not None:
                raise ValueError(
                    "Cannot pass per-batch attn mask with static attn mask"
                )
            attn_mask = per_batch_attn_mask
        else:
            attn_mask = self.attn_mask

        B, N, C = x.shape
        if self.gated_attention:
            gate = self.gated_attention_map(x)  # [B, N, C]
            gate = gate.sigmoid()

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if rope_cos_sin is not None:
            q, k = self._apply_rope_2d(q, k, rope_cos_sin)

        # Convert q,k to v's dtype if needed (in bfloat16)
        if q.dtype != v.dtype:
            q = q.to(v.dtype)
        if k.dtype != v.dtype:
            k = k.to(v.dtype)

        if not self.do_depthwise_attention:
            if self.attn_kernel == -1:
                # Self-attn over whole sequence
                x = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=self.attn_drop_rate,
                    attn_mask=attn_mask,
                    scale=self.scale,
                )
                x = x.transpose(1, 2).reshape(B, N, C)
            else:
                # Windowed neighborhood self-attention (3D)
                if latent_dhw is None:
                    raise ValueError(
                        "NAT3D requires passing latent d, h, w dimensions to DiTBlock forward pass for attn reshape op"
                    )
                d, h, w = latent_dhw
                q, k, v = map(
                    lambda x: rearrange(
                        x, "b head (d h w) c -> b d h w head c", d=d, h=h, w=w
                    ),
                    [q, k, v],
                )
                x = na3d(
                    q,
                    k,
                    v,
                    kernel_size=(
                        self.attn_kernel
                        if isinstance(self.attn_kernel, tuple)
                        else (self.attn_kernel,) * 3
                    ),
                    dilation=self.na_dilations,
                    is_causal=False,
                    backend=self.na3d_backend,
                )
                x = rearrange(x, "b d h w head c -> b (d h w) (head c)")
        else:
            # Depthwise attention for each (h, w) independently
            if latent_dhw is None:
                raise ValueError(
                    "Depthwise attention requires latent d, h, w dimensions for reshape"
                )
            d, h, w = latent_dhw
            if N != d * h * w:
                raise ValueError("Token count must equal d*h*w for depthwise attention")

            q = rearrange(
                q, "b head (d hh ww) c -> (b hh ww) head d c", d=d, hh=h, ww=w
            )
            k = rearrange(
                k, "b head (d hh ww) c -> (b hh ww) head d c", d=d, hh=h, ww=w
            )
            v = rearrange(
                v, "b head (d hh ww) c -> (b hh ww) head d c", d=d, hh=h, ww=w
            )

            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop_rate,
                attn_mask=None,
                scale=self.scale,
            )
            x = rearrange(
                x, "(b hh ww) head d c -> b (d hh ww) (head c)", b=B, hh=h, ww=w
            )

        if self.gated_attention:
            x = x * gate

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DiTBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        # This following are all standard
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_norm=False,
        # for traditional DiT Block drop params are 0.0
        mlp_drop_rate=0.0,
        attn_drop_rate=0.0,
        path_drop_rate=0.0,
        act_layer=partial(nn.GELU, approximate="tanh"),
        norm_layer=partial(nn.LayerNorm, elementwise_affine=False, eps=1e-6),
        qk_norm_layer=partial(nn.RMSNorm, elementwise_affine=False, eps=1e-6),
        attn_mask=None,
        attn_kernel=-1,
        do_depthwise_attention: bool = False,
        na_dilations: int = 1,
        gated_attention: bool = False,
        na3d_backend: Optional[str] = None,
    ):
        super().__init__()
        self.attn_mask = attn_mask
        self._do_depthwise_attention = do_depthwise_attention
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=mlp_drop_rate,
            norm_layer=norm_layer,
            qk_norm_layer=qk_norm_layer,
            input_format="traditional",
            attn_mask=attn_mask,
            activation_checkpointing=False,
            attn_kernel=attn_kernel,
            do_depthwise_attention=do_depthwise_attention,
            na_dilations=na_dilations,
            gated_attention=gated_attention,
            na3d_backend=na3d_backend,
        )

        if path_drop_rate > 0.0:
            # self.drop_path = DropPath(path_drop_rate)
            raise NotImplementedError("DropPath not currently implemented")
        else:
            self.drop_path = nn.Identity()

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim,
            act_layer=act_layer,
            drop_rate=mlp_drop_rate,
            input_format="traditional",
        )
        #    checkpointing=2)

    def forward(
        self,
        x: torch.Tensor,
        latent_dhw: tuple[int, int, int] | None = None,
        rope_cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        """
        Args:
            x: (n, tokens, c)
            latent_dhw: (depth, height, width) dimensions for reshaping
            rope_cos_sin: Precomputed (cos, sin) tables for 2D RoPE, or None
        """
        # self attention
        y = self.norm1(x)
        z = self.attn(y, latent_dhw=latent_dhw, rope_cos_sin=rope_cos_sin)
        x = x + self.drop_path(z)

        # mlp
        y = self.norm2(x)
        z = self.mlp(y)
        x = x + self.drop_path(z)

        return x


class DiTLastLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        patch_size: Tuple[int, int, int],
        out_chans: int,
        norm_layer=partial(nn.LayerNorm, elementwise_affine=False, eps=1e-6),
        use_buggy_layernorm: bool = False,
    ):
        super().__init__()
        self.norm = norm_layer(hidden_size)
        self.linear = nn.Linear(
            hidden_size, patch_size[0] * patch_size[1] * patch_size[2] * out_chans
        )

    def forward(self, x):
        with torch.amp.autocast("cuda", enabled=False):
            x_fp32 = x.float()
            x = self.norm(x_fp32)
            x = self.linear(x)

        return x


class PatchEmbed3D(nn.Module):
    def __init__(self, depth, height, width, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        # Accept int or (d, h, w) tuple
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        pd, ph, pw = patch_size
        if depth % pd != 0:
            raise ValueError(
                f"Depth ({depth}) must be divisible by vertical patch size ({pd})"
            )
        if height % ph != 0:
            raise ValueError(
                f"Height ({height}) must be divisible by horizontal patch size ({ph})"
            )
        if width % pw != 0:
            raise ValueError(
                f"Width ({width}) must be divisible by horizontal patch size ({pw})"
            )

        self.depth = depth
        self.height = height
        self.width = width
        self.patch_size = patch_size
        self.num_patches = (depth // pd) * (height // ph) * (width // pw)

        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True
        )

    def forward(self, x):

        x = self.proj(x)

        return x


class DealiasedPatchEmbed3D(nn.Module):
    """PatchEmbed3D variant that adds shift-invariance via a fixed dealiasing filter.

    Stage 1: stride-1 *valid* conv (padding=0).
      stage1[j] = conv(x[j : j+patch_size]) — purely from real input, no zeros.
      In particular stage1[0] == PatchEmbed3D token 0 exactly.

    Stage 2: per-axis low-pass filter + strided decimation.
      The low-pass is applied only along axes that are actually being decimated
      (axes with ``patch_size > 1``). Axes with ``patch_size == 1`` use a
      length-1 identity filter, so no information is mixed across them — this
      matters for vertical patching of size 1 where adjacent atmospheric levels
      should not bleed into each other.

      Per-axis pad totals ``(k_axis - 1)``, split symmetrically: ``floor((k-1)/2)``
      on the near side and the remainder on the far side. Replicate-padded so no
      zeros are introduced and the filter sits on-grid (no half-filter phase shift).

    Token-count proof per axis (S divisible by p, axis filter length k_axis):
      stage1 size  = S - p + 1
      after pad    = S - p + k_axis      (total pad = k_axis - 1)
      stage2 size  = floor((S - p + k_axis - k_axis) / p) + 1 = S // p  ✓

    proj weights are Conv3d-shape-compatible with PatchEmbed3D (same kernel,
    same in/out channels), but the effective output differs on any axis with
    ``patch_size > 1`` because stage 2 adds the low-pass on those axes.
    """

    def __init__(
        self,
        depth,
        height,
        width,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        resample_filter=(1, 1),
    ):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        pd, ph, pw = patch_size
        if depth % pd != 0:
            raise ValueError(
                f"Depth ({depth}) must be divisible by vertical patch size ({pd})"
            )
        if height % ph != 0:
            raise ValueError(
                f"Height ({height}) must be divisible by horizontal patch size ({ph})"
            )
        if width % pw != 0:
            raise ValueError(
                f"Width ({width}) must be divisible by horizontal patch size ({pw})"
            )
        self.depth = depth
        self.height = height
        self.width = width
        self.patch_size = patch_size
        self.num_patches = (depth // pd) * (height // ph) * (width // pw)

        # Stage 1: valid conv (no padding) so every feature is computed from real data
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=1,
            padding=0,
            bias=True,
        )

        # Stage 2: per-axis low-pass filter. Use `resample_filter` only on axes
        # that are actually decimated (patch_size > 1); on stride-1 axes there
        # is no aliasing to prevent, so a length-1 identity filter is used to
        # avoid mixing data across that axis.
        f_active = torch.as_tensor(resample_filter, dtype=torch.float32)
        identity = torch.tensor([1.0])
        fd = identity if pd == 1 else f_active
        fh = identity if ph == 1 else f_active
        fw = identity if pw == 1 else f_active
        f3d = fd[:, None, None] * fh[None, :, None] * fw[None, None, :]
        f3d = (f3d / (fd.sum() * fh.sum() * fw.sum())).unsqueeze(0).unsqueeze(1)
        # persistent=False: exclude from state_dict so checkpoints saved with
        # PatchEmbed3D (which has no dealias_filter) load cleanly under strict=True.
        # The filter is a fixed constant recomputed from resample_filter at init.
        self.register_buffer("dealias_filter", f3d, persistent=False)
        self._embed_dim = embed_dim

    def forward(self, x):
        # Stage 1: valid conv — stage1[0] == PatchEmbed3D token 0, no zero contamination
        x = self.proj(x)  # (B, embed_dim, D-pd+1, H-ph+1, W-pw+1)

        f = self.dealias_filter.to(x.dtype)
        _, _, kd, kh, kw = f.shape

        # Per-axis symmetric pad totaling (k_axis - 1). For axes with k_axis=1
        # this is zero pad. For odd (k_axis - 1) the extra slot goes on the far
        # side (PyTorch 'same'-style convention).
        pad_d_lo, pad_d_hi = (kd - 1) // 2, (kd - 1) - (kd - 1) // 2
        pad_h_lo, pad_h_hi = (kh - 1) // 2, (kh - 1) - (kh - 1) // 2
        pad_w_lo, pad_w_hi = (kw - 1) // 2, (kw - 1) - (kw - 1) // 2
        if kd > 1 or kh > 1 or kw > 1:
            # F.pad order: (W_left, W_right, H_left, H_right, D_left, D_right)
            x = F.pad(
                x,
                (pad_w_lo, pad_w_hi, pad_h_lo, pad_h_hi, pad_d_lo, pad_d_hi),
                mode="replicate",
            )

        x = F.conv3d(
            x,
            f.tile([self._embed_dim, 1, 1, 1, 1]),
            groups=self._embed_dim,
            stride=self.patch_size,
            padding=0,
        )
        return x


class DiT(nn.Module):
    def __init__(
        self,
        depth=256,
        height=256,
        width=256,
        n_layers=12,
        nside=1024,
        patch_size=1,
        patch_size_vert: int | None = None,
        patch_size_horiz: int | None = None,
        in_chans=3,
        base_out_chans=3,
        embed_dim=768,
        num_heads=16,
        mlp_ratio=4.0,
        frequency_embed_dim=256,
        qkv_bias=True,
        qk_norm=False,
        qk_norm_elementwise_affine: bool = False,
        learn_sigma=False,
        num_classes=0,
        class_dropout_prob=0.1,
        use_skip_connection: bool = False,
        use_concat_skip_connection: bool = False,
        num_input_time_steps: int = 1,
        num_output_time_steps: int = 1,
        attn_mask_type: Optional[str] = None,
        attn_kernel=3,
        do_alt_depthwise_attn: bool = False,
        do_interleaved_dilation: bool = False,
        na_dilations: int = 3,
        do_rope_2d: bool = False,
        do_rope_2d_stereographic: bool = False,
        do_concat_latitude: bool = True,
        do_rotate_wind: bool = False,
        wind_channel_indices: Optional[Tuple[int, int]] = None,
        wind_channel_indices_dual: Optional[
            Tuple[int, int]
        ] = None,  # For dual normalization
        do_bf16_mixed: bool = False,
        do_activation_checkpointing: bool | float = False,
        gated_attention: bool = False,
        # NA3D backend; None uses natten default. Options: "cutlass-fna", "hopper-fna",
        # "blackwell-fna", "flex-fna"
        na3d_backend: Optional[str] = None,
        grid_type: str = "healpix",
        cubesphere_latlon_path: str = "data/latlon_ne1024pg2.nc",
        use_hpx_pe_scaling: bool = True,
        index_is_latlon: bool = False,
        use_dealiased_patch_embed: bool = False,
        dealias_resample_filter: tuple = (1, 4, 6, 4, 1),
    ):
        super().__init__()
        self._learn_sigma = learn_sigma
        self._out_chans = 2 * base_out_chans if learn_sigma else base_out_chans
        self._in_chans = in_chans
        self._do_concat_latitude = do_concat_latitude
        if self._do_concat_latitude:
            self._in_chans += 2

        # Resolve per-axis patch sizes: explicit vert/horiz override the scalar
        self._patch_size_vert = (
            patch_size_vert if patch_size_vert is not None else patch_size
        )
        self._patch_size_horiz = (
            patch_size_horiz if patch_size_horiz is not None else patch_size
        )
        self._patch_size_3d = (
            self._patch_size_vert,
            self._patch_size_horiz,
            self._patch_size_horiz,
        )
        # Keep scalar for backward-compat paths that only need the horizontal size
        self._patch_size = self._patch_size_horiz
        self._height = height
        self._width = width
        self._embed_dim = embed_dim
        self._depth = depth
        self._num_heads = num_heads
        self._mlp_ratio = mlp_ratio
        self._qkv_bias = qkv_bias
        self._qk_norm = qk_norm
        self._qk_norm_elementwise_affine = qk_norm_elementwise_affine
        self._num_classes = num_classes
        self._class_dropout_prob = class_dropout_prob
        self._frequency_embed_dim = frequency_embed_dim
        self._nside = nside
        self._grid_type = (grid_type or "healpix").lower()
        if self._grid_type not in {"healpix", "cubesphere"}:
            raise ValueError(
                f"Unsupported grid_type='{grid_type}'. Expected 'healpix' or 'cubesphere'."
            )
        self._index_is_latlon = index_is_latlon
        self._do_rope_2d = do_rope_2d
        self._do_rope_2d_stereographic = do_rope_2d_stereographic
        self._use_hpx_pe_scaling = use_hpx_pe_scaling
        self._do_alt_depthwise_attn = do_alt_depthwise_attn
        self._do_rotate_wind = do_rotate_wind
        self._wind_channel_indices = wind_channel_indices
        self._wind_channel_indices_dual = (
            wind_channel_indices_dual  # For dual normalization
        )

        if do_rope_2d and do_rope_2d_stereographic:
            raise ValueError(
                "Cannot use both row/column RoPE and stereographic RoPE simultaneously"
            )

        if do_rotate_wind and wind_channel_indices is None:
            raise ValueError(
                "Must specify wind_channel_indices when do_rotate_wind is True"
            )

        if use_concat_skip_connection and use_skip_connection:
            raise ValueError("Cannot use both concat and regular skip connections")
        self._use_skip_connection = use_skip_connection
        self._use_concat_skip_connection = use_concat_skip_connection
        self._num_input_time_steps = num_input_time_steps
        self._num_output_time_steps = num_output_time_steps
        self._do_bf16_mixed = do_bf16_mixed
        # Activation checkpointing: bool or float (0.0-1.0 for fraction of blocks)
        # True/1.0 = all blocks, False/0.0 = none, 0.5 = half the blocks, etc.
        self._activation_checkpointing_ratio = self._parse_checkpointing_param(
            do_activation_checkpointing
        )

        if use_dealiased_patch_embed:
            self._patch_emb = DealiasedPatchEmbed3D(
                depth=depth,
                height=height,
                width=width,
                patch_size=self._patch_size_3d,
                in_chans=self._in_chans,
                embed_dim=embed_dim,
                resample_filter=dealias_resample_filter,
            )
        else:
            self._patch_emb = PatchEmbed3D(
                depth=depth,
                height=height,
                width=width,
                patch_size=self._patch_size_3d,
                in_chans=self._in_chans,
                embed_dim=embed_dim,
            )

        if num_classes > 0:
            raise NotImplementedError(
                "Since label embedding not implemented, num_classes must be 0"
            )

        if attn_mask_type is None:
            self.attn_mask = None
        else:
            raise ValueError(f"attn_mask_type f{attn_mask_type} not supported")

        _qk_norm_layer = partial(
            nn.RMSNorm,
            elementwise_affine=qk_norm_elementwise_affine,
            eps=1e-6,
        )
        self._blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    qk_norm_layer=_qk_norm_layer,
                    attn_mask=self.attn_mask,
                    attn_kernel=attn_kernel,
                    do_depthwise_attention=(i % 2 == 1) and do_alt_depthwise_attn,
                    na_dilations=na_dilations
                    if (do_interleaved_dilation and i % 4 == 2)
                    else 1,
                    gated_attention=gated_attention,
                    na3d_backend=na3d_backend,
                )
                for i in range(n_layers)
            ]
        )
        head_dim = embed_dim // num_heads
        if self._do_rope_2d:
            # Build per-token 2D coordinates (i,j) for RoPE using row/column indices
            d, h, w = (
                depth // self._patch_size_vert,
                height // self._patch_size_horiz,
                width // self._patch_size_horiz,
            )
            ii = torch.arange(h, dtype=torch.float32)
            jj = torch.arange(w, dtype=torch.float32)
            imesh, jmesh = torch.meshgrid(
                ii, jj, indexing="ij"
            )  # imesh, jmesh are [h, w]
            ij_hw = torch.stack([imesh, jmesh], dim=-1)
            ij_hw = rearrange(ij_hw, "h w coord -> (h w) coord", h=h, w=w, coord=2)
            ij_tokens = ij_hw.repeat(d, 1)  # [d*h*w, 2]
            # Precompute cos/sin LUT (avoids recomputing trig every forward pass)
            rope_cos, rope_sin = _precompute_rope_2d_cos_sin(ij_tokens, head_dim)
            self.register_buffer("_rope_cos", rope_cos, persistent=False)
            self.register_buffer("_rope_sin", rope_sin, persistent=False)
        elif self._do_rope_2d_stereographic:
            # RoPE cos/sin will be computed dynamically in forward pass
            # from stereographic projection coordinates
            self._rope_cos = None
            self._rope_sin = None
        else:
            self._rope_cos = None
            self._rope_sin = None
        self._rope_head_dim = head_dim

        # Precompute lat/lon (radians) over global column ids so forward can do:
        #   lat = self._lat_radians[index.long()]
        # for both HEALPix and CubeSphere.
        if self._grid_type == "healpix":
            input_grid = earth2grid.healpix.Grid(
                earth2grid.healpix.nside2level(self._nside),
                pixel_order=earth2grid.healpix.PixelOrder.NEST,
            )
            lat_radians = input_grid.lat * np.pi / 180.0
            lon_radians = input_grid.lon * np.pi / 180.0
            self._ncol_total = int(12 * self._nside * self._nside)
        else:
            ds_ll = xr.open_dataset(cubesphere_latlon_path)
            lat_deg = ds_ll["lat"].values
            lon_deg = ds_ll["lon"].values
            self._ncol_total = int(lat_deg.shape[0])
            lat_radians = lat_deg * (np.pi / 180.0)
            lon_radians = lon_deg * (np.pi / 180.0)
        self.register_buffer(
            "_lat_radians", torch.from_numpy(lat_radians).float(), persistent=False
        )
        self.register_buffer(
            "_lon_radians", torch.from_numpy(lon_radians).float(), persistent=False
        )

        self._final_layer = DiTLastLayer(
            embed_dim, self._patch_size_3d, self._out_chans
        )

        if use_concat_skip_connection:
            if self._in_chans % num_input_time_steps != 0:
                raise ValueError(
                    "_in_chans must be divisible by num_input_time_steps for concat skip"
                )
            last_time_input_chans = self._in_chans // num_input_time_steps
            self._concat_skip_linear = nn.Linear(
                in_features=last_time_input_chans + self._out_chans,
                out_features=self._out_chans,
            )
        self.initialize_weights()

    @staticmethod
    def _parse_checkpointing_param(do_activation_checkpointing: bool | float) -> float:
        """
        Parse activation checkpointing parameter into a ratio.

        Args:
            do_activation_checkpointing: bool or float (0.0-1.0)
                - False or 0.0: no checkpointing
                - True or 1.0: checkpoint all blocks
                - 0.0 < x < 1.0: checkpoint that fraction of blocks

        Returns:
            float: ratio of blocks to checkpoint (0.0 to 1.0)

        Raises:
            TypeError: if input is not bool / int / float.
            ValueError: if the resulting float is outside [0.0, 1.0].
        """
        # bool is a subclass of int — must be checked first so True/False
        # map deterministically to 1.0/0.0 instead of taking the numeric path.
        if isinstance(do_activation_checkpointing, bool):
            return 1.0 if do_activation_checkpointing else 0.0
        if not isinstance(do_activation_checkpointing, (int, float)):
            raise TypeError(
                "do_activation_checkpointing must be bool or numeric (int/float), "
                f"got {type(do_activation_checkpointing).__name__}"
            )
        ratio = float(do_activation_checkpointing)
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(
                f"do_activation_checkpointing must be bool or float in [0, 1], got {ratio}"
            )
        return ratio

    def _should_checkpoint_block(self, block_idx: int) -> bool:
        """
        Determine if a specific block should use activation checkpointing.

        Args:
            block_idx: index of the block (0-indexed)

        Returns:
            bool: True if this block should be checkpointed

        Checkpoints the first N blocks where N = round(ratio * n_layers).
        Examples with 32 layers:
        - ratio=1.0:  all 32 blocks
        - ratio=0.75: first 24 blocks
        - ratio=0.5:  first 16 blocks
        - ratio=0.25: first 8 blocks
        - ratio=0.0:  no blocks
        """
        if not self.training:
            return False
        ratio = self._activation_checkpointing_ratio
        if ratio <= 0.0:
            return False
        if ratio >= 1.0:
            return True
        n_blocks = len(self._blocks)
        n_checkpointed = round(ratio * n_blocks)
        return block_idx < n_checkpointed

    def initialize_weights(self):
        # init transformer layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

        self.apply(_basic_init)

        w = self._patch_emb.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self._patch_emb.proj.bias, 0.0)

        nn.init.constant_(self._final_layer.linear.weight, 0.0)
        nn.init.constant_(self._final_layer.linear.bias, 0.0)

    def prepare_tokens(self, x):
        x = self._patch_emb(x)
        d, h, w = x.shape[-3:]
        x = rearrange(x, "b e d h w -> b (d h w) e")

        return x, (d, h, w)

    def set_tile_size(self, height: int, width: int) -> None:
        """Recompute static RoPE buffers for a new spatial tile size.

        Only has an effect when ``do_rope_2d=True`` (static RoPE).  Calling
        this before inference allows the model to accept tiles larger or
        smaller than the one it was trained on.
        """
        self._height = height
        self._width = width
        if not self._do_rope_2d:
            return
        d = self._depth // self._patch_size_vert
        h = height // self._patch_size_horiz
        w = width // self._patch_size_horiz
        ii = torch.arange(h, dtype=torch.float32)
        jj = torch.arange(w, dtype=torch.float32)
        imesh, jmesh = torch.meshgrid(ii, jj, indexing="ij")
        ij_hw = torch.stack([imesh, jmesh], dim=-1)
        ij_hw = rearrange(ij_hw, "h w coord -> (h w) coord")
        ij_tokens = ij_hw.repeat(d, 1)
        rope_cos, rope_sin = _precompute_rope_2d_cos_sin(ij_tokens, self._rope_head_dim)
        self.register_buffer(
            "_rope_cos", rope_cos.to(self._rope_cos.device), persistent=False
        )
        self.register_buffer(
            "_rope_sin", rope_sin.to(self._rope_sin.device), persistent=False
        )

    def _get_lat_lon_from_index(
        self, index: torch.Tensor | Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._index_is_latlon:
            try:
                lat = index["lat"]
                lon = index["lon"]
            except Exception as exc:  # pragma: no cover - defensive guard
                raise ValueError(
                    "index must provide 'lat' and 'lon' tensors when index_is_latlon is True"
                ) from exc

            if lat.ndim == 2:
                lat = lat.unsqueeze(0)
            if lon.ndim == 2:
                lon = lon.unsqueeze(0)
            if lat.ndim != 3 or lon.ndim != 3:
                raise ValueError(
                    "lat/lon tensors must be 2D or 3D with shape [B, H, W]"
                )
            if lat.shape != lon.shape:
                raise ValueError("lat/lon tensors must have matching shapes")
            return lat, lon

        return self._lat_radians[index.long()], self._lon_radians[index.long()]

    def compute_stereographic_rope_coords(
        self,
        index: torch.Tensor | Mapping[str, torch.Tensor],
        patch_size_horiz: int,
        d_patch: int | None = None,
    ) -> torch.Tensor:
        """
        Compute stereographic projection coordinates for RoPE.

        Args:
            index: HEALPix pixel indices [B, H, W] or lat/lon tensors
            patch_size_horiz: horizontal patch size for spatial downsampling
            d_patch: number of depth tokens to tile the (h, w) coords over.
                Defaults to ``self._depth // self._patch_size_vert`` (semantic
                pathway). Pass ``d_patch=self._depth`` for pixel resolution.

        Returns:
            xy_coords: [d*h*w, 2] stereographic (x,y) coordinates for each token
        """
        # Get lat/lon for all pixels
        lat, lon = self._get_lat_lon_from_index(index)  # [B, H, W]
        B, H, W = lat.shape

        # Find tile center (handling longitude wrapping)
        lat0 = lat.mean(dim=(1, 2), keepdim=True)  # [B, 1, 1]
        lon0 = mean_longitude(lon)  # [B, 1, 1]

        # Average pooling to get patch-level lat/lon (horizontal dims only).
        # Longitude needs a circular mean to handle patches crossing the
        # 0/2π seam; latitude is bounded and uses a plain arithmetic mean.
        lat = rearrange(
            lat,
            "b (h ph) (w pw) -> b h w ph pw",
            ph=patch_size_horiz,
            pw=patch_size_horiz,
        )
        lon = rearrange(
            lon,
            "b (h ph) (w pw) -> b h w ph pw",
            ph=patch_size_horiz,
            pw=patch_size_horiz,
        )
        lat = lat.mean(dim=(3, 4))  # [B, h_patch, w_patch]
        lon = mean_longitude(lon, reduce_dims=(3, 4)).squeeze((-2, -1))

        # Compute stereographic projection
        x, y = stereographic_projection(lat, lon, lat0, lon0)  # [B, h_patch, w_patch]

        # Scale by global pixel size for consistency across tiles across different grids.
        # Each token covers patch_size_horiz^2 fine pixels. Approx area per token:
        #   area ≈ 4π * patch_size_horiz^2 / ncol_total
        # length scale ≈ sqrt(area)
        if self._use_hpx_pe_scaling:
            pixel_scale = torch.sqrt(
                torch.tensor(np.pi * patch_size_horiz**2 / (3.0 * 1024**2))
            )
        else:
            pixel_scale = torch.sqrt(
                torch.tensor(
                    4.0 * np.pi * (patch_size_horiz**2) / float(self._ncol_total),
                    device=lat.device,
                    dtype=lat.dtype,
                )
            )

        x = x / pixel_scale
        y = y / pixel_scale

        # Reshape for all depth levels (use vertical patch size for depth)
        if d_patch is None:
            d_patch = self._depth // self._patch_size_vert
        xy_hw = torch.stack([x, y], dim=-1)  # [B, h_patch, w_patch, 2]
        xy_hw = rearrange(xy_hw, "b h w coord -> b (h w) coord")

        # Repeat for all depth levels (assuming same x,y for all depths)
        xy_tokens = xy_hw.unsqueeze(1).repeat(1, d_patch, 1, 1)  # [B, d_patch, h*w, 2]
        xy_tokens = rearrange(xy_tokens, "b d hw coord -> b (d hw) coord")

        return xy_tokens  # [B, d*h*w, 2]

    def get_tile_center_lat_lon(
        self, index: torch.Tensor | Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get lat/lon for all pixels and compute tile center.

        Args:
            index: HEALPix pixel indices [B, H, W] or lat/lon tensors

        Returns:
            lat, lon: [B, H, W] lat/lon for all pixels
            lat0, lon0: [B, 1, 1] tile center coordinates
        """
        # Get lat/lon for all pixels
        lat, lon = self._get_lat_lon_from_index(index)  # [B, H, W]

        # Find tile center (handling longitude wrapping)
        lat0 = lat.mean(dim=(1, 2), keepdim=True)  # [B, 1, 1]
        lon0 = mean_longitude(lon)  # [B, 1, 1]

        return lat, lon, lat0, lon0

    def _add_skip_connection(self, inputs: torch.Tensor, outputs: torch.Tensor):
        time_inputs = rearrange(
            inputs, "b (t c) d h w s -> b t c d h w s", t=self._num_input_time_steps
        )

        last_inputs = time_inputs[:, -1:, :, :, :, :, :]
        time_outputs = rearrange(
            outputs, "b (t c) d h w s -> b t c d h w s", t=self._num_output_time_steps
        )
        if self._num_output_time_steps > 1:
            last_inputs = torch.concatenate(
                [last_inputs] * self._num_output_time_steps, dim=1
            )

        outputs = rearrange(
            last_inputs + time_outputs, "b t c d h w s -> b (t c) d h w s"
        )
        return outputs

    def _add_concat_skip_connection(self, inputs: torch.Tensor, outputs: torch.Tensor):
        time_inputs = rearrange(
            inputs, "b (t c) d h w s -> b t c d h w s", t=self._num_input_time_steps
        )

        last_inputs = time_inputs[:, -1, :, :, :, :, :]

        outputs = torch.concat([last_inputs, outputs], axis=1)
        outputs = rearrange(outputs, "b c d h w s -> b s d h w c")
        outputs = self._concat_skip_linear(outputs)

        return rearrange(outputs, "b s d h w c -> b c d h w s")

    def forward_tokens(
        self,
        x: torch.Tensor,
        index: torch.Tensor | Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int], tuple | None]:
        """
        Preprocess input and run all transformer blocks, returning raw tokens
        before the final projection and unpatchify.  Used by DiT_Pixel to obtain
        semantic conditioning tokens for the pixel pathway; also called by forward().
        Returns: (tokens, preprocessed_x, latent_dhw, lat_lon_data)
          - tokens:        [B, N, embed_dim] after all blocks, before final layer
          - preprocessed_x: [B, C', D, H, W] input after preprocessing
          - latent_dhw:    (d, h, w) patch-grid shape
          - lat_lon_data:  (lat, lon, lat0, lon0) or None for inverse wind rotation
        """
        _, _, dd, _, _ = x.shape

        lat_lon_data = None

        if self._do_rotate_wind:
            lat, lon, lat0, lon0 = self.get_tile_center_lat_lon(index)
            lat_lon_data = (lat, lon, lat0, lon0)
            u_idx, v_idx = self._wind_channel_indices

            u_in = x[:, u_idx, :, :, :]
            v_in = x[:, v_idx, :, :, :]
            u_rot, v_rot = forward_uv_to_tile(u_in, v_in, lat, lon, lat0, lon0)

            x = x.clone()
            x[:, u_idx, :, :, :] = u_rot
            x[:, v_idx, :, :, :] = v_rot

            if self._wind_channel_indices_dual is not None:
                u_idx_dual, v_idx_dual = self._wind_channel_indices_dual

                u_in_dual = x[:, u_idx_dual, :, :, :]
                v_in_dual = x[:, v_idx_dual, :, :, :]
                u_rot_dual, v_rot_dual = forward_uv_to_tile(
                    u_in_dual, v_in_dual, lat, lon, lat0, lon0
                )

                x[:, u_idx_dual, :, :, :] = u_rot_dual
                x[:, v_idx_dual, :, :, :] = v_rot_dual

        if self._do_concat_latitude:
            lat, _ = self._get_lat_lon_from_index(index)  # [B, hh, ww]
            lat = lat.unsqueeze(1).unsqueeze(2)
            lat = lat.expand(-1, 1, dd, -1, -1)  # [B, 1, dd, hh, ww]
            x = torch.cat([x, torch.cos(lat), torch.sin(lat)], dim=1)

        preprocessed_x = x

        x, latent_dhw = self.prepare_tokens(x)

        if self._do_rope_2d_stereographic:
            ij_coords = self.compute_stereographic_rope_coords(
                index, self._patch_size_horiz
            )
            rope_cos_sin = _precompute_rope_2d_cos_sin(ij_coords, self._rope_head_dim)
        elif self._rope_cos is not None:
            rope_cos_sin = (self._rope_cos, self._rope_sin)
        else:
            rope_cos_sin = None

        with torch.amp.autocast(
            "cuda", dtype=torch.bfloat16, enabled=self._do_bf16_mixed
        ):
            for i, block in enumerate(self._blocks):
                use_checkpoint = self._should_checkpoint_block(i)
                block_rope = None if block._do_depthwise_attention else rope_cos_sin

                if use_checkpoint:

                    def _block_forward(inp, b=block, cs=block_rope):
                        return b(inp, latent_dhw=latent_dhw, rope_cos_sin=cs)

                    x = activation_checkpoint(_block_forward, x, use_reentrant=False)
                else:
                    x = block(x, latent_dhw=latent_dhw, rope_cos_sin=block_rope)

        return x, preprocessed_x, latent_dhw, lat_lon_data

    def forward(
        self,
        x: torch.Tensor,
        index: torch.Tensor | Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        _, _, dd, hh, ww = x.shape
        x_in = x  # Store original input for skip connections

        tokens, _, _, lat_lon_data = self.forward_tokens(x, index)
        x = tokens

        x = self._final_layer(x)

        x = self.unpatchify(x, dd, hh, ww)

        if self._use_skip_connection:
            x = self._add_skip_connection(x_in, x)  # noqa
        elif self._use_concat_skip_connection:
            x = self._add_concat_skip_connection(x_in, x)  # noqa

        # Rotate output wind components back to geographic coordinates
        if self._do_rotate_wind:
            lat, lon, lat0, lon0 = lat_lon_data
            u_idx, v_idx = self._wind_channel_indices

            u_out = x[:, u_idx, :, :, :]  # [B, D, H, W]
            v_out = x[:, v_idx, :, :, :]  # [B, D, H, W]

            u_geo, v_geo = inverse_tile_to_uv(u_out, v_out, lat, lon, lat0, lon0)

            x = x.clone()
            x[:, u_idx, :, :, :] = u_geo
            x[:, v_idx, :, :, :] = v_geo

        return x

    def unpatchify(self, x, dd=None, hh=None, ww=None):
        _, num_tokens, out_dim = x.shape

        depth = dd or self._depth
        height = hh or self._height
        width = ww or self._width

        out_str = "B (d h w) (P_d P_h P_w C) -> B C (d P_d) (h P_h) (w P_w)"

        x = rearrange(
            x,
            out_str,
            P_d=self._patch_size_vert,
            P_h=self._patch_size_horiz,
            P_w=self._patch_size_horiz,
            C=self._out_chans,
            d=depth // self._patch_size_vert,
            h=height // self._patch_size_horiz,
            w=width // self._patch_size_horiz,
        )

        return x


if __name__ == "__main__":
    model = DiT(
        depth=19,
        height=32,
        width=32,
        patch_size_horiz=2,
        patch_size_vert=1,
        in_chans=6,
        base_out_chans=6,
        n_layers=4,
        embed_dim=128,
        num_heads=2,
        attn_kernel=3,
        do_interleaved_dilation=False,
        do_alt_depthwise_attn=True,
        do_bf16_mixed=True,
        do_rope_2d_stereographic=True,
        do_rotate_wind=True,
        gated_attention=True,
        grid_type="cubesphere",
        index_is_latlon=True,
        wind_channel_indices=(0, 1),
    ).to("cuda")
    x = torch.randn(1, 6, 19, 16, 16).cuda()
    lat_vals = torch.linspace(-0.5 * torch.pi, 0.5 * torch.pi, 16, device="cuda")
    lon_vals = torch.linspace(0.0, 2.0 * torch.pi, 16, device="cuda")
    lat_grid, lon_grid = torch.meshgrid(lat_vals, lon_vals, indexing="ij")
    index = {"lat": lat_grid.unsqueeze(0), "lon": lon_grid.unsqueeze(0)}
    y = model(x, index)
    print(y.shape)
