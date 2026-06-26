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
PixelDiT: Two-stage Diffusion Transformer for weather emulation.

Stage 1 (Semantic): standard DiT with coarse patches → conditioning tokens s_cond.
Stage 2 (Pixel):    pixel-resolution tokens conditioned via pixel-wise AdaLN from s_cond.

Adapts the two-stage semantic→pixel concept of PixelDiT (arXiv:2511.20645) for
deterministic weather regression (no diffusion/timestep/label conditioning). The
pixel-wise AdaLN is an independent reimplementation, and this model adds an
original bilinear-upsample + depthwise-conv conditioning path
(use_bilinear_dw_gelu_project_adaln) — the variant the production model uses.

DiT_Pixel composes DiT (from dit_3d.py) for the semantic stage rather than
duplicating it, so all semantic pathway args are forwarded directly to DiT.
"""

from functools import partial
from typing import Mapping, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from screamcast.depthwise_conv import DepthwiseConv
from screamcast.dit_3d import (
    MLP,
    Attention,
    DiT,
    DiTBlock,
    PatchEmbed3D,
    _precompute_rope_2d_cos_sin,
    inverse_tile_to_uv,
)


class PixelDiTBlock(nn.Module):
    """
    DiT block with pixel-wise AdaLN modulation from semantic conditioning.

    Pixel-wise AdaLN inspired by PixelDiT (arXiv:2511.20645): each semantic patch
    token is projected into per-pixel (shift, scale, gate) parameters for both the
    attention and MLP sub-layers, enabling spatially-varying modulation at pixel
    resolution. An alternative, original conditioning path (bilinear upsample +
    depthwise conv; use_bilinear_dw_gelu_project_adaln) is also provided and is
    the one used by the production model.

    Notation
    --------
    - dim              : pixel-pathway embedding dimension  (D_pix)
    - cond_dim         : semantic-pathway embedding dimension (D_sem)
    - pixels_per_patch : pv * ph * pw  (number of pixels per semantic patch)
    """

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        pixels_per_patch: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        mlp_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        act_layer=partial(nn.GELU, approximate="tanh"),
        norm_layer=partial(nn.LayerNorm, elementwise_affine=False, eps=1e-6),
        qk_norm_layer=partial(nn.RMSNorm, elementwise_affine=False, eps=1e-6),
        attn_kernel: int = 3,
        na_dilations: int = 1,
        gated_attention: bool = False,
        use_bilinear_dw_gelu_project_adaln: bool = False,
        chunk_size_grouped_conv: int = 2,
        use_chunked_depthwise_conv: bool = True,
    ):
        """
        num_tokens_hint: used to determine conv implementation
        """
        super().__init__()
        self._use_bilinear_dw_gelu_project_adaln = use_bilinear_dw_gelu_project_adaln

        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=mlp_drop_rate,
            qk_norm_layer=qk_norm_layer,
            input_format="traditional",
            attn_kernel=attn_kernel,
            do_depthwise_attention=False,
            na_dilations=na_dilations,
            gated_attention=gated_attention,
        )

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.drop_path = nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim,
            act_layer=act_layer,
            drop_rate=mlp_drop_rate,
            input_format="traditional",
        )

        # ── pixel-wise AdaLN ──
        # Projects each semantic token → per-pixel (shift, scale, gate) × 2 sub-layers
        self.num_adaln_params = 6  # shift1, scale1, gate1, shift2, scale2, gate2
        # Zero-init so the pixel pathway starts as identity at initialization
        if self._use_bilinear_dw_gelu_project_adaln:
            # Bilinear-DW-RMSNorm-GeLU-Project adaln:
            #   1) bilinear upsample at full cond_dim  — to pixel resolution, max info preserved
            #   2) DWConv2d(cond_dim, 5×5, replicate)  — smooth patch boundaries at pixel res
            #   3) RMSNorm(cond_dim, no affine)        — stabilize scale after conv drift
            #   4) GeLU                                 — nonlinearity at pixel resolution
            #   5) Linear(cond_dim → 6*dim, zero-init) —project at pixel resolution (full GEMM)
            self.chunk_size_grouped_conv = chunk_size_grouped_conv
            if use_chunked_depthwise_conv:
                self.adaln_bilinear_dw_conv = DepthwiseConv(
                    cond_dim,
                    chunk_size=chunk_size_grouped_conv,
                    kernel_size=(5, 5),
                    padding=(2, 2),
                    bias=True,
                    padding_mode="replicate",
                )
            else:
                self.adaln_bilinear_dw_conv = nn.Conv2d(
                    cond_dim,
                    cond_dim,
                    kernel_size=(5, 5),
                    padding=(2, 2),
                    groups=cond_dim,
                    bias=True,
                    padding_mode="replicate",
                )

            # Identity-init: smoothing starts as a no-op; learns to blend boundary seams
            nn.init.zeros_(self.adaln_bilinear_dw_conv.weight)
            self.adaln_bilinear_dw_conv.weight.data[:, 0, 2, 2] = 1.0
            nn.init.zeros_(self.adaln_bilinear_dw_conv.bias)
            self.adaln_bilinear_dw_norm = nn.RMSNorm(cond_dim, elementwise_affine=False)
            self.adaln_bilinear_dw_proj = nn.Linear(
                cond_dim, self.num_adaln_params * dim
            )
            nn.init.zeros_(self.adaln_bilinear_dw_proj.weight)
            nn.init.zeros_(self.adaln_bilinear_dw_proj.bias)
        else:
            self.pixels_per_patch = pixels_per_patch
            self.adaln_pixel_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(cond_dim, pixels_per_patch * self.num_adaln_params * dim),
            )
            nn.init.zeros_(self.adaln_pixel_proj[-1].weight)
            nn.init.zeros_(self.adaln_pixel_proj[-1].bias)

    @staticmethod
    def _expand_cond_to_pixels(
        adaln_raw: torch.Tensor,
        pixel_dhw: tuple[int, int, int],
        semantic_dhw: tuple[int, int, int],
    ) -> torch.Tensor:
        """Reshape (B, N_patches, ppp*M) → (B, D*H*W, M) in correct spatial order."""
        d, h, w = pixel_dhw
        sd, sh, sw = semantic_dhw
        pv, ph, pw = d // sd, h // sh, w // sw
        return rearrange(
            adaln_raw,
            "b (sd sh sw) (pv ph pw m) -> b (sd pv) (sh ph) (sw pw) m",
            sd=sd,
            sh=sh,
            sw=sw,
            pv=pv,
            ph=ph,
            pw=pw,
        ).reshape(adaln_raw.shape[0], d * h * w, -1)

    def _compute_bilinear_dw_gelu_project_adaln_params(
        self,
        s_cond_bilinear: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        bsz, sd, _, _, _ = s_cond_bilinear.shape
        s_pix = rearrange(s_cond_bilinear, "b sd c h w -> (b sd) c h w")

        s_pix = self.adaln_bilinear_dw_conv(s_pix)
        s_pix = self.adaln_bilinear_dw_norm(s_pix.permute(0, 2, 3, 1)).permute(
            0, 3, 1, 2
        )
        s_pix = rearrange(s_pix, "(b sd) c h w -> b sd c h w", b=bsz, sd=sd)
        s_pix_seq = rearrange(s_pix, "b d c h w -> b (d h w) c")
        s_pix_seq = torch.nn.functional.gelu(s_pix_seq)
        adaln_params = self.adaln_bilinear_dw_proj(s_pix_seq)
        adaln_params = rearrange(
            adaln_params, "b n (six c) -> b n six c", six=self.num_adaln_params
        )
        return adaln_params.unbind(dim=-2)

    @staticmethod
    def precompute_bilinear_cond(
        s_cond: torch.Tensor,
        pixel_dhw: tuple[int, int, int],
        semantic_dhw: tuple[int, int, int],
    ) -> torch.Tensor:
        _, ph, pw = pixel_dhw
        sd, sh, sw = semantic_dhw
        s_sp = rearrange(s_cond, "b (sd sh sw) c -> b sd c sh sw", sd=sd, sh=sh, sw=sw)
        s_sp = rearrange(s_sp, "b sd c sh sw -> (b sd) c sh sw")
        s_pix = torch.nn.functional.interpolate(
            s_sp, size=(ph, pw), mode="bilinear", align_corners=False
        )
        return rearrange(s_pix, "(b sd) c h w -> b sd c h w", sd=sd)

    def forward(
        self,
        x: torch.Tensor,
        s_cond: torch.Tensor,
        pixel_dhw: tuple[int, int, int],
        semantic_dhw: tuple[int, int, int],
        s_cond_bilinear: torch.Tensor | None = None,
        rope_cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x           : [B, D*H*W, dim]          pixel tokens
            s_cond      : [B, N_patches, cond_dim]  semantic conditioning
            pixel_dhw      : (D, H, W)                full pixel grid shape
            semantic_dhw   : (sd, sh, sw)             semantic patch-grid shape
            s_cond_bilinear: optional shared bilinear-upsampled semantic tensor
            rope_cos_sin   : optional 2-D RoPE tables
        """
        if self._use_bilinear_dw_gelu_project_adaln:
            if s_cond_bilinear is None:
                raise ValueError(
                    "s_cond_bilinear is required when "
                    "use_bilinear_dw_gelu_project_adaln is enabled."
                )
            (
                shift1,
                scale1,
                gate1,
                shift2,
                scale2,
                gate2,
            ) = self._compute_bilinear_dw_gelu_project_adaln_params(s_cond_bilinear)
        else:
            adaln_raw = self.adaln_pixel_proj(s_cond)  # [B, N_s, ppp*6*C]
            adaln_params = self._expand_cond_to_pixels(
                adaln_raw, pixel_dhw, semantic_dhw
            )
            shift1, scale1, gate1, shift2, scale2, gate2 = adaln_params.chunk(6, dim=-1)

        y = self.norm1(x)
        y = y * (1 + scale1) + shift1

        z = self.attn(y, latent_dhw=pixel_dhw, rope_cos_sin=rope_cos_sin)

        x = x + self.drop_path(gate1 * z)

        y = self.norm2(x)
        y = y * (1 + scale2) + shift2
        z = self.mlp(y)
        x = x + self.drop_path(gate2 * z)

        return x


class PixelDiTLastLayer(nn.Module):
    """
    Final projection for the pixel pathway: norm → linear.
    All semantic conditioning is already injected by PixelDiTBlock layers,
    so no AdaLN is needed here.
    """

    def __init__(
        self,
        hidden_size: int,
        out_chans: int,
        norm_layer=partial(nn.LayerNorm, elementwise_affine=False, eps=1e-6),
    ):
        super().__init__()
        self.norm = norm_layer(hidden_size)
        self.linear = nn.Linear(hidden_size, out_chans)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            x_fp32 = x.float()
            x = self.norm(x_fp32)
            x = self.linear(x)
        return x


class DiT_Pixel(nn.Module):
    """
    Two-stage PixelDiT model.

    Composes a DiT instance (semantic stage) with a pixel-resolution pathway
    conditioned via pixel-wise AdaLN on the semantic tokens.

    All semantic-pathway args are forwarded directly to DiT; only the
    pixel-specific args are consumed here.
    """

    def __init__(
        self,
        # ── Pixel pathway (Stage 2) ──
        embed_dim_pixel: int = 128,
        n_layers_pixel: int = 4,
        num_heads_pixel: int | None = None,  # default: embed_dim_pixel // 64
        mlp_ratio_pixel: float = 4.0,
        attn_kernel_pixel: int = 3,
        gated_attention_pixel: bool = False,
        qk_norm_pixel: bool = False,
        qk_norm_elementwise_affine_pixel: bool = False,
        do_bf16_mixed_pixel: bool = False,
        freeze_semantic: bool = False,
        freeze_pixel_blocks: bool = False,
        use_bilinear_dw_gelu_project_adaln_pixel: bool = False,
        use_chunked_depthwise_conv_pixel: bool = True,
        first_block_only_adaln_pixel: bool = False,
        do_rope_2d_pixel: bool = False,
        do_rope_2d_stereographic_pixel: bool = False,
        # ── Semantic pathway (Stage 1) — all forwarded to DiT ──
        **semantic_kwargs,
    ):
        super().__init__()

        # ── Stage 1: Semantic pathway (standard DiT) ──
        # Skip connections are disabled on the semantic DiT because DiT_Pixel
        # handles them itself on the final pixel output.
        semantic_kwargs.setdefault("use_skip_connection", False)
        semantic_kwargs.setdefault("use_concat_skip_connection", False)
        self.semantic = DiT(**semantic_kwargs)
        # _final_layer is never called in DiT_Pixel (we use forward_tokens instead).
        # Delete it to avoid DDP unused-parameter deadlock when freeze_semantic=False.
        del self.semantic._final_layer

        if freeze_semantic:
            for p in self.semantic.parameters():
                p.requires_grad_(False)

        # Derive pixel pathway layout from semantic DiT attributes
        depth = self.semantic._depth
        height = self.semantic._height
        width = self.semantic._width
        in_chans = self.semantic._in_chans  # already accounts for lat concat channels
        sem_embed_dim = self.semantic._embed_dim
        patch_size_vert = self.semantic._patch_size_vert
        patch_size_horiz = self.semantic._patch_size_horiz
        out_chans = self.semantic._out_chans

        self._depth = depth
        self._height = height
        self._width = width
        self._do_bf16_mixed_pixel = do_bf16_mixed_pixel
        self._out_chans = out_chans
        self._first_block_only_adaln_pixel = first_block_only_adaln_pixel
        self._do_rope_2d_pixel = do_rope_2d_pixel
        self._do_rope_2d_stereographic_pixel = do_rope_2d_stereographic_pixel

        if do_rope_2d_pixel and do_rope_2d_stereographic_pixel:
            raise ValueError(
                "Cannot use both row/column RoPE and stereographic RoPE on the "
                "pixel pathway simultaneously"
            )

        # Carry over skip-connection flags (applied to the pixel output)
        self._use_skip_connection = semantic_kwargs.get("use_skip_connection", False)
        self._use_concat_skip_connection = semantic_kwargs.get(
            "use_concat_skip_connection", False
        )
        self._do_rotate_wind = semantic_kwargs.get("do_rotate_wind", False)
        self._wind_channel_indices: Optional[Tuple[int, int]] = semantic_kwargs.get(
            "wind_channel_indices", None
        )

        if use_bilinear_dw_gelu_project_adaln_pixel and patch_size_vert != 1:
            raise ValueError(
                "use_bilinear_dw_gelu_project_adaln requires patch_size_vert=1 "
                "(bilinear upsamples H×W only, not the depth axis); "
                f"got patch_size_vert={patch_size_vert}"
            )

        # pixels per semantic patch = pv * ph * ph
        pixels_per_patch = patch_size_vert * patch_size_horiz * patch_size_horiz

        num_heads_pixel = (
            num_heads_pixel
            if num_heads_pixel is not None
            else max(1, embed_dim_pixel // 64)
        )
        self._rope_head_dim_pixel = embed_dim_pixel // num_heads_pixel

        if do_rope_2d_pixel:
            # Pixel pathway uses a 1×1×1 patch, so the pixel token grid is
            # (depth, height, width). Build per-token (i, j) indices over the
            # spatial axes and tile across the depth axis.
            ii = torch.arange(height, dtype=torch.float32)
            jj = torch.arange(width, dtype=torch.float32)
            imesh, jmesh = torch.meshgrid(ii, jj, indexing="ij")
            ij_hw = torch.stack([imesh, jmesh], dim=-1)
            ij_hw = rearrange(ij_hw, "h w coord -> (h w) coord")
            ij_tokens = ij_hw.repeat(depth, 1)
            rope_cos, rope_sin = _precompute_rope_2d_cos_sin(
                ij_tokens, self._rope_head_dim_pixel
            )
            self.register_buffer("_rope_cos_pixel", rope_cos, persistent=False)
            self.register_buffer("_rope_sin_pixel", rope_sin, persistent=False)
        else:
            self._rope_cos_pixel = None
            self._rope_sin_pixel = None

        if first_block_only_adaln_pixel and n_layers_pixel < 1:
            raise ValueError(
                "first_block_only_adaln_pixel requires n_layers_pixel >= 1 "
                "(need at least one PixelDiTBlock to inject s_cond); "
                f"got n_layers_pixel={n_layers_pixel}"
            )

        _pixel_qk_norm_layer = partial(
            nn.RMSNorm,
            elementwise_affine=qk_norm_elementwise_affine_pixel,
            eps=1e-6,
        )

        # ── Stage 2: Pixel pathway ──
        self._pixel_patch_emb = PatchEmbed3D(
            depth=depth,
            height=height,
            width=width,
            patch_size=(1, 1, 1),
            in_chans=in_chans,
            embed_dim=embed_dim_pixel,
        )

        def _make_pixel_adaln_block() -> PixelDiTBlock:
            return PixelDiTBlock(
                dim=embed_dim_pixel,
                cond_dim=sem_embed_dim,
                pixels_per_patch=pixels_per_patch,
                num_heads=num_heads_pixel,
                mlp_ratio=mlp_ratio_pixel,
                attn_kernel=attn_kernel_pixel,
                gated_attention=gated_attention_pixel,
                qk_norm=qk_norm_pixel,
                qk_norm_layer=_pixel_qk_norm_layer,
                use_bilinear_dw_gelu_project_adaln=use_bilinear_dw_gelu_project_adaln_pixel,
                use_chunked_depthwise_conv=use_chunked_depthwise_conv_pixel,
            )

        def _make_plain_block() -> DiTBlock:
            return DiTBlock(
                dim=embed_dim_pixel,
                num_heads=num_heads_pixel,
                mlp_ratio=mlp_ratio_pixel,
                attn_kernel=attn_kernel_pixel,
                gated_attention=gated_attention_pixel,
                qk_norm=qk_norm_pixel,
                qk_norm_layer=_pixel_qk_norm_layer,
            )

        if first_block_only_adaln_pixel:
            pixel_blocks = [_make_pixel_adaln_block()] + [
                _make_plain_block() for _ in range(n_layers_pixel - 1)
            ]
        else:
            pixel_blocks = [_make_pixel_adaln_block() for _ in range(n_layers_pixel)]

        self._pixel_blocks = nn.ModuleList(pixel_blocks)

        self._pixel_final_layer = PixelDiTLastLayer(
            hidden_size=embed_dim_pixel,
            out_chans=out_chans,
        )

        if freeze_pixel_blocks:
            for p in self._pixel_blocks.parameters():
                p.requires_grad_(False)
            for p in self._pixel_final_layer.parameters():
                p.requires_grad_(False)

    @property
    def _index_is_latlon(self) -> bool:
        return self.semantic._index_is_latlon

    def set_tile_size(self, height: int, width: int) -> None:
        """Recompute static RoPE buffers for a new spatial tile size.

        Delegates to the semantic stage and, when ``do_rope_2d_pixel=True``,
        also recomputes the pixel-resolution buffers. Stereographic RoPE has
        no static buffers (computed per-forward from ``index``), so nothing
        extra is needed in that case.
        """
        self.semantic.set_tile_size(height, width)
        self._height = height
        self._width = width
        if not self._do_rope_2d_pixel:
            return
        ii = torch.arange(height, dtype=torch.float32)
        jj = torch.arange(width, dtype=torch.float32)
        imesh, jmesh = torch.meshgrid(ii, jj, indexing="ij")
        ij_hw = torch.stack([imesh, jmesh], dim=-1)
        ij_hw = rearrange(ij_hw, "h w coord -> (h w) coord")
        ij_tokens = ij_hw.repeat(self._depth, 1)
        rope_cos, rope_sin = _precompute_rope_2d_cos_sin(
            ij_tokens, self._rope_head_dim_pixel
        )
        self.register_buffer(
            "_rope_cos_pixel",
            rope_cos.to(self._rope_cos_pixel.device),
            persistent=False,
        )
        self.register_buffer(
            "_rope_sin_pixel",
            rope_sin.to(self._rope_sin_pixel.device),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        index: torch.Tensor | Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Two-stage forward pass:
          Stage 1 (Semantic): coarse-patch DiT blocks → s_cond (conditioning tokens)
          Stage 2 (Pixel):    1×1×1 pixel blocks predict the full target directly
        """
        _, _, dd, hh, ww = x.shape
        x_in = x

        # ── Stage 1: semantic tokens + preprocessed input ──
        s_cond, x_proc, semantic_dhw, lat_lon_data = self.semantic.forward_tokens(
            x, index
        )

        # ── Stage 2: pixel pathway → full prediction ──
        pixel_dhw = (dd, hh, ww)
        x_pix = rearrange(self._pixel_patch_emb(x_proc), "b c d h w -> b (d h w) c")
        use_shared_bilinear = any(
            pblock._use_bilinear_dw_gelu_project_adaln
            for pblock in self._pixel_blocks
            if isinstance(pblock, PixelDiTBlock)
        )
        if use_shared_bilinear:
            s_cond_bilinear = PixelDiTBlock.precompute_bilinear_cond(
                s_cond, pixel_dhw, semantic_dhw
            )
        else:
            s_cond_bilinear = None

        # Pixel-resolution RoPE (shared across all pixel blocks).
        # patch_size_horiz=1 keeps (h, w) at pixel resolution; d_patch=self._depth
        # tiles the 2-D coords across the full depth axis.
        if self._do_rope_2d_stereographic_pixel:
            ij_coords_pixel = self.semantic.compute_stereographic_rope_coords(
                index, patch_size_horiz=1, d_patch=self._depth
            )
            pixel_rope_cos_sin = _precompute_rope_2d_cos_sin(
                ij_coords_pixel, self._rope_head_dim_pixel
            )
        elif self._do_rope_2d_pixel:
            pixel_rope_cos_sin = (self._rope_cos_pixel, self._rope_sin_pixel)
        else:
            pixel_rope_cos_sin = None

        with torch.amp.autocast(
            "cuda", dtype=torch.bfloat16, enabled=self._do_bf16_mixed_pixel
        ):
            for i, pblock in enumerate(self._pixel_blocks):
                if isinstance(pblock, PixelDiTBlock):
                    x_pix = pblock(
                        x_pix,
                        s_cond=s_cond,
                        pixel_dhw=pixel_dhw,
                        semantic_dhw=semantic_dhw,
                        s_cond_bilinear=s_cond_bilinear,
                        rope_cos_sin=pixel_rope_cos_sin,
                    )
                else:
                    x_pix = pblock(
                        x_pix,
                        latent_dhw=pixel_dhw,
                        rope_cos_sin=pixel_rope_cos_sin,
                    )

            x_pix = self._pixel_final_layer(x_pix)

        x = rearrange(x_pix, "b (d h w) c -> b c d h w", d=dd, h=hh, w=ww)

        # ── Skip connections ──
        if self._use_skip_connection:
            x = self.semantic._add_skip_connection(x_in, x)
        elif self._use_concat_skip_connection:
            x = self.semantic._add_concat_skip_connection(x_in, x)

        # ── Inverse wind rotation ──
        if self._do_rotate_wind and lat_lon_data is not None:
            lat, lon, lat0, lon0 = lat_lon_data
            u_idx, v_idx = self._wind_channel_indices

            u_out = x[:, u_idx, :, :, :]
            v_out = x[:, v_idx, :, :, :]
            u_geo, v_geo = inverse_tile_to_uv(u_out, v_out, lat, lon, lat0, lon0)

            x = x.clone()
            x[:, u_idx, :, :, :] = u_geo
            x[:, v_idx, :, :, :] = v_geo

        return x


if __name__ == "__main__":
    # Quick smoke test of the two-stage DiT_Pixel model
    D, H, W = 24, 64, 64
    PH, PV = 4, 2
    model = DiT_Pixel(
        depth=D,
        height=H,
        width=W,
        patch_size_horiz=PH,
        patch_size_vert=PV,
        in_chans=6,
        base_out_chans=6,
        n_layers=4,
        embed_dim=128,
        num_heads=2,
        attn_kernel=3,
        do_alt_depthwise_attn=True,
        gated_attention=True,
        embed_dim_pixel=64,
        n_layers_pixel=2,
        num_heads_pixel=1,
        attn_kernel_pixel=3,
        do_bf16_mixed=True,
        do_bf16_mixed_pixel=True,
        do_rope_2d_stereographic=True,
        do_rotate_wind=True,
        grid_type="cubesphere",
        index_is_latlon=True,
        wind_channel_indices=(0, 1),
    ).to("cuda")

    n_sem = sum(p.numel() for p in model.semantic._blocks.parameters())
    n_pix = sum(p.numel() for p in model._pixel_blocks.parameters())
    n_all = sum(p.numel() for p in model.parameters())
    print(
        f"Semantic params: {n_sem:,}  |  Pixel params: {n_pix:,}  |  Total: {n_all:,}"
    )

    x = torch.randn(1, 6, D, H, W).cuda()
    lat_vals = torch.linspace(-0.5 * torch.pi, 0.5 * torch.pi, H, device="cuda")
    lon_vals = torch.linspace(0.0, 2.0 * torch.pi, W, device="cuda")
    lat_grid, lon_grid = torch.meshgrid(lat_vals, lon_vals, indexing="ij")
    index = {"lat": lat_grid.unsqueeze(0), "lon": lon_grid.unsqueeze(0)}

    y = model(x, index)
    if y.shape != (1, 6, D, H, W):
        raise ValueError(f"Shape mismatch! Got {y.shape}")
    print(f"Input: {x.shape}  Output: {y.shape}")
    print("smoke test passed")
