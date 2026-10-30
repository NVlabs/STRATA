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
"""Shared CUDA conditioning preparation by Cirrus-42 (AI agent)."""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch
from physicsnemo.experimental.models.strata.layers import StrataPixel3DBlock
from torch.utils.cpp_extension import load

from screamcast import conditioning_shared_pad

_extension = None


def initialize():
    global _extension
    if _extension is None:
        build = (
            Path(tempfile.gettempdir()) / "cirrus42-decoder" / "cuda-fused-prepare-wide"
        )
        build.mkdir(parents=True, exist_ok=True)
        _extension = load(
            name="cirrus42_fused_prepare_wide",
            sources=[str(Path(__file__).parent / "cuda/fused_conditioning_prepare.cu")],
            build_directory=str(build),
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_103,code=sm_103"],
            verbose=True,
        )


@torch.library.custom_op("cirrus42::fused_prepare_wide", mutates_args=())
def fused_prepare(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    a, b = _extension.padding(x)
    return a, b


@fused_prepare.register_fake
def fused_prepare_fake(x):
    return conditioning_shared_pad.shared_pad_fake(x)


def precompute(conditioning, pixel_dhw, backbone_dhw):
    if pixel_dhw != (24, 128, 128) or backbone_dhw != (24, 32, 32):
        raise ValueError("Fused preparation requires fixed B8/T128 decoder geometry")
    return fused_prepare(conditioning)


def install(strata):
    initialize()
    conditioning_shared_pad.install_compute(strata)
    StrataPixel3DBlock.precompute_bilinear_cond = staticmethod(precompute)


def component_profile(conditioning, latent_dhw, args, measure):
    conditioning_shared_pad.initialize()
    incumbent = torch.compile(conditioning_shared_pad.precompute)
    expected = incumbent(conditioning, (24, 128, 128), latent_dhw)
    actual = fused_prepare(conditioning)
    different = 0
    total = 0
    for a, b in zip(actual, expected):
        for aa, bb in zip(a.split(4), b.split(4)):
            different += aa.ne(bb).sum().item()
            total += aa.numel()
    timings = dict(
        incumbent_upsample_and_pad=measure(
            lambda: incumbent(conditioning, (24, 128, 128), latent_dhw),
            args.warmup,
            args.repeats,
        ),
        fused_prepare=measure(
            lambda: fused_prepare(conditioning), args.warmup, args.repeats
        ),
    )
    row = dict(
        time=datetime.now(timezone.utc).isoformat(),
        agent="Cirrus-42 (AI agent)",
        experiment="cuda-fused-prepare-wide-components",
        commit=args.commit,
        batch_size=8,
        tile_size=128,
        amp="bf16",
        components=timings,
        padded_bf16_different_elements=different,
        padded_bf16_elements=total,
        removed_fp32_intermediate_bytes=12884901888,
        decision="diagnostic",
    )
    with Path(args.results).open("a") as f:
        f.write(json.dumps(row) + "\n")
    print("FUSED_PREPARE_COMPONENTS", json.dumps(row), flush=True)
