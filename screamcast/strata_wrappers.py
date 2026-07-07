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
"""SCREAMCast wrappers around the physicsnemo Strata architecture.

The transformer architecture (3D neighborhood-attention DiT backbone + pixel
stage) lives in physicsnemo (``physicsnemo.experimental.models.strata``); see
``strata_backend.py`` for the class selection. These wrappers keep the
SCREAMCast-specific concerns local:

- deriving per-pixel lat/lon (``pos``) from the tile ``index`` for both
  HEALPix and CubeSphere grids (``tile_geometry.TileGeometry``),
- tile-local wind rotation before/after the network (``wind_rotation``),
- latitude channel concatenation,
- the optional dealiased patch embedding (``dealias.DealiasedPatchEmbed3D``),
- freeze flags for staged fine-tuning,
- loading legacy (pre-migration) checkpoints (``checkpoint_compat``),
- the tanh-GELU activation the shipped checkpoints were trained with.

The forward contract is unchanged from the legacy models:
``network(x, index)`` with ``x`` of shape ``[B, C, D, H, W]``.

Numerics note: the stereographic RoPE coordinates now come from physicsnemo
(`spherical_centroid` tile centers / patch pooling) instead of the legacy
mean-latitude + circular-mean-longitude math, which introduces a small,
accepted numerical difference for ``do_rope_2d_stereographic`` models. The
wind rotation keeps the legacy mean-based center, so its numerics are
unchanged. Everything else is bit-compatible modulo op ordering.
"""

from typing import Mapping, Optional, Tuple

import torch
import torch.nn as nn
from physicsnemo.experimental.models.strata.coords import build_axial_token_coords
from physicsnemo.experimental.nn import build_axial_rope_cos_sin_2d_continuous
from physicsnemo.nn.module.mlp_layers import Mlp

from screamcast.checkpoint_compat import remap_legacy_state_dict
from screamcast.dealias import DealiasedPatchEmbed3D
from screamcast.strata_backend import (
    BACKBONE_CLS,
    CROSSATTN_CLS,
    STRATA_CLS,
    TILE_CENTER_FN,
)
from screamcast.tile_geometry import TileGeometry
from screamcast.wind_rotation import forward_uv_to_tile, inverse_tile_to_uv


def _replace_channels(x, replacements: dict) -> torch.Tensor:
    """Rebuild the channel dim with slice + cat instead of in-place writes.

    Mathematically identical to ``x.clone(); x[:, c] = v`` on plain tensors.
    Required form under domain parallelism: in-place channel assignment on a
    sharded DTensor is unreliable across PyTorch versions, and cat's backward
    slices the incoming grad instead of stacking plain zeros into it.
    """
    pieces = [
        replacements[c].unsqueeze(1) if c in replacements else x[:, c : c + 1]
        for c in range(x.shape[1])
    ]
    return torch.cat(pieces, dim=1)


def _use_tanh_gelu(model: nn.Module) -> None:
    """Swap every Mlp GELU to the tanh approximation, in place.

    physicsnemo's blocks build ``Mlp`` with exact (erf) GELU; the SCREAMCast
    models have always trained with ``nn.GELU(approximate="tanh")``. The swap
    is parameter-free and does not touch state-dict keys. This is a permanent
    local adjustment (no upstream change planned); it must run before loading
    any pre-migration checkpoint, or every MLP output shifts by up to ~5e-4
    per activation.
    """
    for module in model.modules():
        if isinstance(module, Mlp):
            for i, layer in enumerate(module.layers):
                if isinstance(layer, nn.GELU) and layer.approximate == "none":
                    module.layers[i] = nn.GELU(approximate="tanh")


def _init_patch_embed(patch_embed: nn.Module) -> None:
    """DiT-style init for a (swapped-in) patch embedding conv."""
    w = patch_embed.proj.weight.data
    nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
    nn.init.constant_(patch_embed.proj.bias, 0.0)


def _remap_legacy_pre_hook(module, state_dict, prefix, *args) -> None:
    """load_state_dict pre-hook: translate legacy keys under ``prefix`` in place.

    ``prefix`` is this wrapper's position in the outer module tree (e.g.
    ``"_orig_mod."`` when loading a torch.compile-saved checkpoint into a
    compiled model, ``""`` for a direct load). The remap itself additionally
    strips stale wrapper prefixes baked into old checkpoints and is a no-op on
    already-migrated keys.
    """
    scoped = {k: v for k, v in state_dict.items() if k.startswith(prefix)}
    if not scoped:
        return
    bare = {k[len(prefix) :]: v for k, v in scoped.items()}
    remapped, report = remap_legacy_state_dict(bare)
    if not report.was_legacy and len(remapped) == len(bare):
        # Fast path: nothing changed (modulo stale wrapper-prefix strips).
        if all(k in bare for k in remapped):
            return
    for k in scoped:
        del state_dict[k]
    for k, v in remapped.items():
        state_dict[prefix + k] = v


def _rope_mode(do_rope_2d: bool, do_rope_2d_stereographic: bool) -> str:
    if do_rope_2d and do_rope_2d_stereographic:
        raise ValueError(
            "Cannot use both row/column RoPE and stereographic RoPE simultaneously"
        )
    if do_rope_2d_stereographic:
        return "stereographic"
    if do_rope_2d:
        return "axial"
    return "none"


class _ScreamcastStrataBase(nn.Module):
    """Shared geometry/wind/lat-concat plumbing for both wrappers."""

    def __init__(
        self,
        *,
        do_concat_latitude: bool,
        do_rotate_wind: bool,
        wind_channel_indices: Optional[Tuple[int, int]],
        grid_type: str,
        nside: int,
        cubesphere_latlon_path: str,
        index_is_latlon: bool,
    ):
        super().__init__()
        if do_rotate_wind and wind_channel_indices is None:
            raise ValueError(
                "Must specify wind_channel_indices when do_rotate_wind is True"
            )
        self._do_concat_latitude = do_concat_latitude
        self._do_rotate_wind = do_rotate_wind
        self._wind_channel_indices = wind_channel_indices
        self.geometry = TileGeometry(
            grid_type=grid_type,
            nside=nside,
            cubesphere_latlon_path=cubesphere_latlon_path,
            index_is_latlon=index_is_latlon,
        )
        # Remap legacy checkpoint keys inside the nn.Module load machinery
        # itself. A load_state_dict OVERRIDE would not fire when this wrapper
        # is loaded as a submodule (torch.compile's OptimizedModule, DDP,
        # lightning fabric all call load_state_dict on the OUTER module and
        # walk children via hooks) — with strict=False that silently loads
        # zero weights. The pre-hook fires on every path.
        self.register_load_state_dict_pre_hook(_remap_legacy_pre_hook)

    @property
    def _index_is_latlon(self) -> bool:
        # Read by inference code via getattr(network, "_index_is_latlon", False).
        return self.geometry.index_is_latlon

    def _preprocess(
        self,
        x: torch.Tensor,
        index: torch.Tensor | Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, tuple | None]:
        """Wind rotation + latitude concat + pos construction.

        Returns (x, pos, lat_lon_data). ``pos`` is [B, 2, H, W] lat/lon in
        radians; ``lat_lon_data`` is (lat, lon, lat0, lon0) for the inverse
        wind rotation, or None.
        """
        B, _, dd, _, _ = x.shape
        lat, lon = self.geometry.lat_lon_from_index(index)  # [b, H, W]
        if lat.shape[0] != B:
            lat = lat.expand(B, -1, -1)
            lon = lon.expand(B, -1, -1)

        lat_lon_data = None
        if self._do_rotate_wind:
            lat0, lon0 = TILE_CENTER_FN(lat, lon)
            lat_lon_data = (lat, lon, lat0, lon0)
            u_idx, v_idx = self._wind_channel_indices
            u_rot, v_rot = forward_uv_to_tile(
                x[:, u_idx], x[:, v_idx], lat, lon, lat0, lon0
            )
            x = _replace_channels(x, {u_idx: u_rot, v_idx: v_rot})

        if self._do_concat_latitude:
            lat_e = lat.unsqueeze(1).unsqueeze(2).expand(-1, 1, dd, -1, -1)
            x = torch.cat([x, torch.cos(lat_e), torch.sin(lat_e)], dim=1)

        pos = torch.cat([lat.unsqueeze(1), lon.unsqueeze(1)], dim=1)  # [B, 2, H, W]
        return x, pos, lat_lon_data

    def _postprocess(
        self, x: torch.Tensor, lat_lon_data: tuple | None
    ) -> torch.Tensor:
        """Inverse wind rotation on the network output."""
        if self._do_rotate_wind and lat_lon_data is not None:
            lat, lon, lat0, lon0 = lat_lon_data
            u_idx, v_idx = self._wind_channel_indices
            u_geo, v_geo = inverse_tile_to_uv(
                x[:, u_idx], x[:, v_idx], lat, lon, lat0, lon0
            )
            x = _replace_channels(x, {u_idx: u_geo, v_idx: v_geo})
        return x



class ScreamcastStrataBackbone(_ScreamcastStrataBase):
    """Single-stage model (legacy ``DiT`` / model_type ``dit3d``).

    Composes a physicsnemo ``StrataTransformer3D`` under ``self.strata`` and
    adds the SCREAMCast geometry/wind/lat-concat plumbing around it. All
    constructor argument names match the legacy ``DiT`` so the train factories
    and configs are unchanged.
    """

    def __init__(
        self,
        *,
        depth: int = 256,
        height: int = 256,
        width: int = 256,
        n_layers: int = 12,
        nside: int = 1024,
        patch_size: int = 1,
        patch_size_vert: int | None = None,
        patch_size_horiz: int | None = None,
        in_chans: int = 3,
        base_out_chans: int = 3,
        embed_dim: int = 768,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        qk_norm_elementwise_affine: bool = False,
        attn_kernel: int | tuple[int, int, int] = 3,
        do_alt_depthwise_attn: bool = False,
        do_interleaved_dilation: bool = False,
        na_dilations: int = 3,
        do_rope_2d: bool = False,
        do_rope_2d_stereographic: bool = False,
        do_concat_latitude: bool = True,
        do_rotate_wind: bool = False,
        wind_channel_indices: Optional[Tuple[int, int]] = None,
        do_bf16_mixed: bool = False,
        do_activation_checkpointing: bool | float = False,
        gated_attention: bool = False,
        na3d_backend: Optional[str] = None,
        grid_type: str = "healpix",
        cubesphere_latlon_path: str = "data/latlon_ne1024pg2.nc",
        use_hpx_pe_scaling: bool = True,
        index_is_latlon: bool = False,
        use_dealiased_patch_embed: bool = False,
        dealias_resample_filter: tuple = (1, 4, 6, 4, 1),
    ):
        super().__init__(
            do_concat_latitude=do_concat_latitude,
            do_rotate_wind=do_rotate_wind,
            wind_channel_indices=wind_channel_indices,
            grid_type=grid_type,
            nside=nside,
            cubesphere_latlon_path=cubesphere_latlon_path,
            index_is_latlon=index_is_latlon,
        )
        pv = patch_size_vert if patch_size_vert is not None else patch_size
        ph = patch_size_horiz if patch_size_horiz is not None else patch_size
        rope_mode = _rope_mode(do_rope_2d, do_rope_2d_stereographic)

        self.strata = BACKBONE_CLS(
            in_channels=in_chans + (2 if do_concat_latitude else 0),
            out_channels=base_out_chans,
            input_shape=(depth, height, width),
            patch_size=(pv, ph, ph),
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=n_layers,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            qk_norm_affine=qk_norm_elementwise_affine,
            attn_kernel=attn_kernel,
            na_dilation=na_dilations,
            do_interleaved_dilation=do_interleaved_dilation,
            do_alt_depthwise_attn=do_alt_depthwise_attn,
            gated_attention=gated_attention,
            na3d_backend=na3d_backend,
            rope_mode=rope_mode,
            rope_base=100.0,
            rope_length_scale=(
                self.geometry.rope_length_scale(ph, use_hpx_pe_scaling)
                if rope_mode == "stereographic"
                else 1.0
            ),
            activation_checkpointing=do_activation_checkpointing,
            bf16_mixed=do_bf16_mixed,
        )
        _use_tanh_gelu(self.strata)

        if use_dealiased_patch_embed:
            self.strata.patch_embed = DealiasedPatchEmbed3D(
                depth=depth,
                height=height,
                width=width,
                patch_size=(pv, ph, ph),
                in_chans=self.strata.in_channels,
                embed_dim=embed_dim,
                resample_filter=dealias_resample_filter,
            )
            _init_patch_embed(self.strata.patch_embed)

    @property
    def _out_chans(self) -> int:
        return self.strata.out_channels

    @property
    def _depth(self) -> int:
        return self.strata.depth

    @property
    def _height(self) -> int:
        return self.strata.height

    @property
    def _width(self) -> int:
        return self.strata.width

    @property
    def _activation_checkpointing_ratio(self) -> float:
        return self.strata._activation_checkpointing_ratio

    @_activation_checkpointing_ratio.setter
    def _activation_checkpointing_ratio(self, value: float) -> None:
        self.strata._activation_checkpointing_ratio = value

    def disable_activation_checkpointing(self) -> None:
        self.strata._activation_checkpointing_ratio = 0.0

    def set_tile_size(self, height: int, width: int) -> None:
        """Reconfigure the expected spatial tile size (all RoPE modes)."""
        self.strata.set_tile_size(height, width)

    def forward(
        self,
        x: torch.Tensor,
        index: torch.Tensor | Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        x, pos, lat_lon_data = self._preprocess(x, index)
        x = self.strata(x, pos=pos)
        return self._postprocess(x, lat_lon_data)


class ScreamcastStrata(_ScreamcastStrataBase):
    """Two-stage model (legacy ``DiT_Pixel`` / model_type ``pixeldit``).

    Composes a physicsnemo ``Strata`` (backbone + pixel stage) under
    ``self.strata``. Pixel-stage argument names match the legacy ``DiT_Pixel``
    (``*_pixel``); backbone args are passed through ``**semantic_kwargs``
    exactly as before.
    """

    def __init__(
        self,
        *,
        # ── Pixel pathway (Stage 2) ──
        embed_dim_pixel: int = 128,
        n_layers_pixel: int = 4,
        num_heads_pixel: int | None = None,  # default: embed_dim_pixel // 64
        mlp_ratio_pixel: float = 4.0,
        attn_kernel_pixel: int = 3,
        gated_attention_pixel: bool = False,
        qk_norm_pixel: bool = False,
        qk_norm_elementwise_affine_pixel: bool = False,
        na3d_backend_pixel: Optional[str] = None,
        do_bf16_mixed_pixel: bool = False,
        freeze_semantic: bool = False,
        freeze_pixel_blocks: bool = False,
        pixel_cond_mode: str = "adaln",
        semantic_cross_attn_window_pixel: int = 3,
        use_bilinear_dw_gelu_project_adaln_pixel: bool = False,
        use_chunked_depthwise_conv_pixel: bool = True,
        first_block_only_adaln_pixel: bool = False,
        do_rope_2d_pixel: bool = False,
        do_rope_2d_stereographic_pixel: bool = False,
        do_activation_checkpointing_pixel: bool | float = False,
        # ── Semantic pathway (Stage 1) — all legacy DiT args ──
        **semantic_kwargs,
    ):
        # Wrapper-level (geometry / wind / lat-concat) args come in with the
        # semantic kwargs, exactly as they did for the legacy DiT.
        super().__init__(
            do_concat_latitude=semantic_kwargs.get("do_concat_latitude", True),
            do_rotate_wind=semantic_kwargs.get("do_rotate_wind", False),
            wind_channel_indices=semantic_kwargs.get("wind_channel_indices", None),
            grid_type=semantic_kwargs.get("grid_type", "healpix"),
            nside=semantic_kwargs.get("nside", 1024),
            cubesphere_latlon_path=semantic_kwargs.get(
                "cubesphere_latlon_path", "data/latlon_ne1024pg2.nc"
            ),
            index_is_latlon=semantic_kwargs.get("index_is_latlon", False),
        )

        if pixel_cond_mode not in {"adaln", "cross_attn"}:
            raise ValueError(
                "pixel_cond_mode must be one of {'adaln', 'cross_attn'}, "
                f"got {pixel_cond_mode!r}"
            )
        if pixel_cond_mode == "cross_attn" and CROSSATTN_CLS is None:
            raise NotImplementedError(
                "pixel_cond_mode='cross_attn' requires the internal screamcast "
                "backend (strata_backend.CROSSATTN_CLS); it is not part of the "
                "public STRATA release."
            )
        if pixel_cond_mode == "cross_attn" and use_bilinear_dw_gelu_project_adaln_pixel:
            raise ValueError(
                "use_bilinear_dw_gelu_project_adaln_pixel is only valid with "
                "pixel_cond_mode='adaln'"
            )

        backbone_config = self._backbone_config(semantic_kwargs)
        pv = backbone_config["patch_size"][0]
        if use_bilinear_dw_gelu_project_adaln_pixel and pv != 1:
            raise ValueError(
                "use_bilinear_dw_gelu_project_adaln requires patch_size_vert=1 "
                "(bilinear upsamples H×W only, not the depth axis); "
                f"got patch_size_vert={pv}"
            )

        rope_mode_pixel = _rope_mode(do_rope_2d_pixel, do_rope_2d_stereographic_pixel)
        use_hpx_pe_scaling = semantic_kwargs.get("use_hpx_pe_scaling", True)

        strata_kwargs = dict(
            backbone_config=backbone_config,
            embed_dim_pixel=embed_dim_pixel,
            num_layers_pixel=n_layers_pixel,
            num_heads_pixel=num_heads_pixel,
            mlp_ratio_pixel=mlp_ratio_pixel,
            attn_kernel_pixel=attn_kernel_pixel,
            gated_attention_pixel=gated_attention_pixel,
            qk_norm_pixel=qk_norm_pixel,
            qk_norm_affine_pixel=qk_norm_elementwise_affine_pixel,
            na3d_backend_pixel=na3d_backend_pixel,
            adaln_mode=(
                "bilinear_dw"
                if use_bilinear_dw_gelu_project_adaln_pixel
                else "pixel_proj"
            ),
            first_block_only_adaln=first_block_only_adaln_pixel,
            use_chunked_depthwise_conv=use_chunked_depthwise_conv_pixel,
            rope_mode_pixel=rope_mode_pixel,
            rope_base_pixel=100.0,
            rope_length_scale_pixel=(
                self.geometry.rope_length_scale(1, use_hpx_pe_scaling)
                if rope_mode_pixel == "stereographic"
                else 1.0
            ),
            bf16_mixed_pixel=do_bf16_mixed_pixel,
            activation_checkpointing_pixel=do_activation_checkpointing_pixel,
        )
        if pixel_cond_mode == "cross_attn":
            self.strata = CROSSATTN_CLS(
                semantic_cross_attn_window=semantic_cross_attn_window_pixel,
                **strata_kwargs,
            )
        else:
            self.strata = STRATA_CLS(**strata_kwargs)
        _use_tanh_gelu(self.strata)

        if semantic_kwargs.get("use_dealiased_patch_embed", False):
            depth, height, width = backbone_config["input_shape"]
            self.strata.backbone.patch_embed = DealiasedPatchEmbed3D(
                depth=depth,
                height=height,
                width=width,
                patch_size=backbone_config["patch_size"],
                in_chans=backbone_config["in_channels"],
                embed_dim=backbone_config["embed_dim"],
                resample_filter=semantic_kwargs.get(
                    "dealias_resample_filter", (1, 4, 6, 4, 1)
                ),
            )
            _init_patch_embed(self.strata.backbone.patch_embed)

        if freeze_semantic:
            for p in self.strata.backbone.parameters():
                p.requires_grad_(False)
        if freeze_pixel_blocks:
            for p in self.strata.pixel_blocks.parameters():
                p.requires_grad_(False)
            for p in self.strata.pixel_final_layer.parameters():
                p.requires_grad_(False)

    def _backbone_config(self, semantic_kwargs: dict) -> dict:
        """Translate legacy DiT kwargs into a StrataTransformer3D config."""
        sk = dict(semantic_kwargs)
        # Wrapper-level concerns already consumed by _ScreamcastStrataBase.
        for consumed in (
            "do_concat_latitude",
            "do_rotate_wind",
            "wind_channel_indices",
            "grid_type",
            "nside",
            "cubesphere_latlon_path",
            "index_is_latlon",
            "use_hpx_pe_scaling",
            "use_dealiased_patch_embed",
            "dealias_resample_filter",
        ):
            sk.pop(consumed, None)

        depth = sk.pop("depth")
        height = sk.pop("height")
        width = sk.pop("width")
        patch_size = sk.pop("patch_size", 1)
        pv = sk.pop("patch_size_vert", None)
        ph = sk.pop("patch_size_horiz", None)
        pv = pv if pv is not None else patch_size
        ph = ph if ph is not None else patch_size
        rope_mode = _rope_mode(
            sk.pop("do_rope_2d", False), sk.pop("do_rope_2d_stereographic", False)
        )
        in_chans = sk.pop("in_chans")
        config = dict(
            in_channels=in_chans + (2 if self._do_concat_latitude else 0),
            out_channels=sk.pop("base_out_chans"),
            input_shape=(depth, height, width),
            patch_size=(pv, ph, ph),
            embed_dim=sk.pop("embed_dim", 768),
            num_heads=sk.pop("num_heads", 16),
            num_layers=sk.pop("n_layers", 12),
            mlp_ratio=sk.pop("mlp_ratio", 4.0),
            qkv_bias=sk.pop("qkv_bias", True),
            qk_norm=sk.pop("qk_norm", False),
            qk_norm_affine=sk.pop("qk_norm_elementwise_affine", False),
            attn_kernel=sk.pop("attn_kernel", 3),
            na_dilation=sk.pop("na_dilations", 3),
            do_interleaved_dilation=sk.pop("do_interleaved_dilation", False),
            do_alt_depthwise_attn=sk.pop("do_alt_depthwise_attn", False),
            gated_attention=sk.pop("gated_attention", False),
            na3d_backend=sk.pop("na3d_backend", None),
            rope_mode=rope_mode,
            rope_base=100.0,
            rope_length_scale=(
                self.geometry.rope_length_scale(
                    ph, semantic_kwargs.get("use_hpx_pe_scaling", True)
                )
                if rope_mode == "stereographic"
                else 1.0
            ),
            activation_checkpointing=sk.pop("do_activation_checkpointing", False),
            bf16_mixed=sk.pop("do_bf16_mixed", False),
        )
        if sk:
            raise TypeError(
                f"Unsupported legacy DiT kwargs for the Strata backbone: {sorted(sk)}"
            )
        return config

    @property
    def _out_chans(self) -> int:
        return self.strata.out_channels

    @property
    def _depth(self) -> int:
        return self.strata.depth

    @property
    def _height(self) -> int:
        return self.strata.height

    @property
    def _width(self) -> int:
        return self.strata.width

    def disable_activation_checkpointing(self) -> None:
        self.strata.backbone._activation_checkpointing_ratio = 0.0
        self.strata._activation_checkpointing_ratio_pixel = 0.0

    def set_tile_size(self, height: int, width: int) -> None:
        """Reconfigure the expected tile size for backbone AND pixel stage.

        physicsnemo ``Strata`` has no re-tiling hook of its own (documented
        gap), so this reaches into its pixel RoPE buffers to rebuild them for
        ``rope_mode_pixel="axial"``. Mirrors the construction-time code in
        physicsnemo strata.py (pinned commit 07cdcc8b); revisit on pin bumps.
        """
        self.strata.backbone.set_tile_size(height, width)
        self.strata.height = height
        self.strata.width = width
        if self.strata.rope_mode_pixel != "axial":
            return
        # The cross-attention model owns extra axial buffers (cross Q/K
        # tables) and rebuilds all of them itself; the plain Strata falls
        # through to the generic pixel-table rebuild below.
        retile = getattr(self.strata, "retile_axial_rope_buffers", None)
        if retile is not None:
            retile(height, width)
            return
        coords = build_axial_token_coords(self.strata.depth, height, width)
        cos, sin = build_axial_rope_cos_sin_2d_continuous(
            coords[:, 0],
            coords[:, 1],
            self.strata.head_dim_pixel,
            theta=self.strata.rope_base_pixel,
        )
        device = self.strata._rope_cos_pixel.device
        self.strata.register_buffer(
            "_rope_cos_pixel", cos.to(device), persistent=False
        )
        self.strata.register_buffer(
            "_rope_sin_pixel", sin.to(device), persistent=False
        )

    def forward(
        self,
        x: torch.Tensor,
        index: torch.Tensor | Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        x, pos, lat_lon_data = self._preprocess(x, index)
        x = self.strata(x, pos=pos)
        return self._postprocess(x, lat_lon_data)
