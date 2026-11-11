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
from types import MethodType

import torch
import torch.nn.functional as F
from physicsnemo.experimental.models.strata.layers import StrataPixel3DBlock
from torch.utils.cpp_extension import load

_extension = None
_native_precompute = StrataPixel3DBlock.precompute_bilinear_cond


def initialize():
    global _extension
    if _extension is not None:
        return
    build = Path(tempfile.gettempdir()) / "cirrus42-decoder" / "cuda-shared-pad"
    build.mkdir(parents=True, exist_ok=True)
    _extension = load(
        name="cirrus42_shared_pad",
        sources=[str(Path(__file__).parent / "cuda" / "shared_conditioning_pad.cu")],
        build_directory=str(build),
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_103,code=sm_103"],
        verbose=True,
    )


@torch.library.custom_op("cirrus42::shared_pad", mutates_args=())
def shared_pad(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    a, b = _extension.padding(x)
    return a, b


@shared_pad.register_fake
def shared_pad_fake(x):
    return tuple(
        x.new_empty((192, 132, 132, 512), dtype=torch.bfloat16).permute(0, 3, 1, 2)
        for _ in range(2)
    )


def precompute(conditioning, pixel_dhw, backbone_dhw):
    s = _native_precompute(conditioning, pixel_dhw, backbone_dhw)
    return shared_pad(s.reshape(192, 1024, 128, 128))


def restore_precompute():
    StrataPixel3DBlock.precompute_bilinear_cond = staticmethod(_native_precompute)


def compute(self, padded):
    weight = self.adaln_bilinear_dw_conv.weight
    bias = self.adaln_bilinear_dw_conv.bias
    # Reproduce the two native groups-512 external convolution calls exactly:
    # BF16 operands, bias=None, BF16 result. Add live FP32 bias afterwards.
    a = F.conv2d(padded[0], weight[:512], groups=512)
    b = F.conv2d(padded[1], weight[512:], groups=512)
    s = torch.cat(
        (
            a.float() + bias[:512][None, :, None, None],
            b.float() + bias[512:][None, :, None, None],
        ),
        dim=1,
    )
    n = self.adaln_bilinear_dw_norm(s.permute(0, 2, 3, 1))
    z = F.gelu(n.reshape(8, 24 * 128 * 128, 1024))
    return self.adaln_bilinear_dw_proj(z).reshape(8, 24 * 128 * 128, 6, 128).unbind(-2)


def install(strata):
    initialize()
    StrataPixel3DBlock.precompute_bilinear_cond = staticmethod(precompute)
    install_compute(strata)


def install_compute(strata):
    """Bind unchanged learned conditioning without loading a padding extension."""
    for block in strata.pixel_blocks:
        block._compute_bilinear_dw_adaln_params = MethodType(compute, block)


def component_profile(strata, conditioning, latent_dhw, args, measure):
    block = strata.pixel_blocks[0]
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        s = torch.compile(_native_precompute)(conditioning, (24, 128, 128), latent_dhw)
        x = s.reshape(192, 1024, 128, 128)
        padded = shared_pad(x)
        counts = []
        for actual, part in zip(padded, x.split(512, dim=1)):
            expected = (
                F.pad(part.contiguous(), (2, 2, 2, 2), mode="replicate")
                .bfloat16()
                .contiguous(memory_format=torch.channels_last)
            )
            counts.append(
                torch.stack(
                    [a.ne(b).sum() for a, b in zip(actual.split(8), expected.split(8))]
                ).sum()
            )
            del expected
        different = torch.stack(counts).sum().item()
        native = torch.compile(
            lambda value: StrataPixel3DBlock._compute_bilinear_dw_adaln_params(
                block, value
            )
        )
        candidate = torch.compile(block._compute_bilinear_dw_adaln_params)
        expected = native(s)
        actual = candidate(padded)
        modulation_error = [
            (a.float() - b.float()).square().mean().sqrt().item()
            for a, b in zip(actual, expected)
        ]
        timings = dict(
            shared_cuda_pad=measure(lambda: shared_pad(x), args.warmup, args.repeats),
            native_conditioning=measure(lambda: native(s), args.warmup, args.repeats),
            candidate_conditioning_prepared=measure(
                lambda: candidate(padded), args.warmup, args.repeats
            ),
        )
        if args.profile_traffic:
            torch.cuda.synchronize()
            torch.cuda.profiler.start()
            native(s)
            candidate(padded)
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
    row = dict(
        time=datetime.now(timezone.utc).isoformat(),
        agent="Cirrus-42 (AI agent)",
        experiment="cuda-shared-pad-components",
        commit=args.commit,
        batch_size=8,
        tile_size=128,
        amp="bf16",
        components=timings,
        padded_bf16_different_elements=different,
        modulation_absolute_rms=modulation_error,
        decision="diagnostic",
        traffic_model="two BF16 NHWC padding chunks, 6,851,395,584 bytes prepared once rather than four times; stored-byte count from tensor dimensions, not hardware DRAM counters",
    )
    with Path(args.results).open("a") as f:
        f.write(json.dumps(row) + "\n")
    print("CUDA_SHARED_PAD", json.dumps(row), flush=True)
