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
"""Capture real column operands without propagating trial error; Cirrus-42 (AI agent)."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from physicsnemo.experimental.models.strata.layers import _build_chunked_depthwise_conv

from screamcast import column_precision
from screamcast.decoder_benchmark import compile_phase
from screamcast.earth2studio_wrappers import ScreamcastModel
from screamcast.forward_stage_benchmark import BackboneFeatures


def run(args):
    cached = torch.load(args.inputs, map_location="cpu", weights_only=True)
    if (cached["batch_size"], cached["tile_size"], cached["checkpoint"]) != (
        8,
        128,
        args.checkpoint,
    ):
        raise ValueError(
            "Real-operand diagnostic requires the fixed B8/T128 capture and checkpoint"
        )
    if args.operands_dir:
        column_precision.operand_directory = Path(args.operands_dir)
        column_precision.operand_directory.mkdir(parents=True, exist_ok=False)
    model = ScreamcastModel.from_checkpoint(args.checkpoint, bf16=True).cuda().eval()
    model.set_tile_size(128)
    strata = model.pipeline.network.strata
    for block in strata.pixel_blocks:
        conv = block.adaln_bilinear_dw_conv
        conv.chunk_size = 512
        conv._chunked_conv = _build_chunked_depthwise_conv(conv, 512)
    x, pos = cached["x"].cuda(), cached["pos"].cuda()
    del cached
    isolated = BackboneFeatures(strata.backbone).eval()
    original_compute = [
        b._compute_bilinear_dw_adaln_params for b in strata.pixel_blocks
    ]
    original_convs = [b.adaln_bilinear_dw_conv for b in strata.pixel_blocks]
    with torch.inference_mode():
        _, calibration = compile_phase(
            isolated,
            (x, pos),
            strata,
            "semantic-column-diagnostic",
            original_compute,
            original_convs,
        )
    if len(column_precision.records) != 12:
        raise RuntimeError("Expected observations for twelve real column blocks")
    operand_files = []
    for observation in column_precision.records:
        if "operand_path" in observation:
            path = Path(observation["operand_path"])
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            operand_files.append(
                dict(path=str(path), sha256=digest, bytes=path.stat().st_size)
            )
    row = dict(
        agent="Cirrus-42 (AI agent)",
        time=datetime.now(timezone.utc).isoformat(),
        commit=args.commit,
        experiment="column-real-normalization-precision",
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        inputs=args.inputs,
        input_sha256=hashlib.file_digest(open(args.inputs, "rb"), "sha256").hexdigest(),
        checkpoint=args.checkpoint,
        scope="One compiled B8/T128 backbone; each diagnostic returns native cuDNN output, collecting trial error on twelve real column blocks without propagating trial error",
        native_reference="aten._scaled_dot_product_cudnn_attention.default(q,k,v,None,False,scale=0.125)[0]",
        canonical_backbone_error=calibration,
        column_blocks=column_precision.records,
        operand_files=operand_files,
    )
    Path(args.results).write_text(json.dumps(row, indent=2) + "\n")
    print(
        "CALIBRATION",
        {k: v for k, v in calibration.items() if k != "channel_relative_rms"},
        flush=True,
    )
