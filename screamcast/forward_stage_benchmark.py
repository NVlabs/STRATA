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
"""Measure compiled semantic and native pixel-decoder stages independently."""

import json
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn


class CachedConditioning(nn.Module):
    """Supply measured backbone features to the unchanged native Strata decoder."""

    def __init__(self, conditioning, latent_dhw):
        super().__init__()
        self.register_buffer("conditioning", conditioning, persistent=False)
        self.latent_dhw = latent_dhw

    def forward_tokens(self, x, pos):
        return self.conditioning, self.latent_dhw


class BackboneFeatures(nn.Module):
    """Expose semantic channels on axis one for existing output-error gates."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x, pos):
        features, _ = self.backbone.forward_tokens(x, pos)
        return features.transpose(1, 2).unsqueeze(-1).unsqueeze(-1)


def run_stage_benchmark(network, network_args, args, measure, errors):
    """Time stages using preprocessed real input, then reconcile with a full control."""
    output_dir = Path(args.output_dir)
    strata = network.strata
    backbone = strata.backbone
    with torch.inference_mode():
        x, pos, _ = network._preprocess(*network_args)
        if args.save_decoder_inputs:
            conditioning, latent_dhw = torch.compile(backbone.forward_tokens)(x, pos)
            torch.cuda.synchronize()
            torch.save(
                dict(
                    x=x.cpu(),
                    pos=pos.cpu(),
                    conditioning=conditioning.cpu(),
                    latent_dhw=latent_dhw,
                    network_args=(
                        network_args[0].cpu(),
                        {k: v.cpu() for k, v in network_args[1].items()},
                    ),
                    checkpoint=args.checkpoint,
                    source_revision=args.source_revision,
                    batch_size=args.batch_size,
                    tile_size=args.tile_size,
                ),
                args.save_decoder_inputs,
            )
            print("Saved fixed decoder inputs:", args.save_decoder_inputs, flush=True)
            return
        eager_full_strata = strata(x, pos)
        eager_cond, latent_dhw = backbone.forward_tokens(x, pos)
        if not torch.isfinite(eager_cond).all().item():
            raise ValueError("Backbone produced nonfinite conditioning")

        compiled_full = torch.compile(network)
        full_output = compiled_full(*network_args)
        if not torch.isfinite(full_output).all().item():
            raise ValueError("Full compiled network output is nonfinite")
        full_timing = measure(lambda: compiled_full(*network_args))
        print("Full compiled control:", full_timing, flush=True)

        compiled_backbone = torch.compile(backbone.forward_tokens)
        started = time.perf_counter()
        compiled_cond, compiled_dhw = compiled_backbone(x, pos)
        torch.cuda.synchronize()
        backbone_first_seconds = time.perf_counter() - started
        if compiled_dhw != latent_dhw or not torch.isfinite(compiled_cond).all().item():
            raise ValueError("Compiled backbone output is invalid")
        backbone_timing = measure(lambda: compiled_backbone(x, pos))
        print("Compiled semantic backbone:", backbone_timing, flush=True)

        cached = CachedConditioning(eager_cond, latent_dhw)
        try:
            strata.backbone = cached
            cached_eager_output = strata(x, pos)
            cached_eager_exact = torch.equal(cached_eager_output, eager_full_strata)
            if not cached_eager_exact:
                raise ValueError("Caching backbone changed the native decoder output")
            cached.conditioning = compiled_cond
            compiled_decoder = torch.compile(strata)
            started = time.perf_counter()
            decoder_output = compiled_decoder(x, pos)
            torch.cuda.synchronize()
            decoder_first_seconds = time.perf_counter() - started
            if not torch.isfinite(decoder_output).all().item():
                raise ValueError("Compiled decoder output is nonfinite")
            decoder_timing = measure(lambda: compiled_decoder(x, pos))
            print("Compiled native pixel decoder:", decoder_timing, flush=True)
            # Postprocess uses the same geometry as the full wrapper control.
            _, _, lat_lon = network._preprocess(*network_args)
            composed_output = network._postprocess(decoder_output, lat_lon)
            composition_error = errors(composed_output, full_output)
            decoder_error = errors(decoder_output, eager_full_strata)
        finally:
            strata.backbone = backbone

        # Recheck the full control after isolated stages; compilation/startup is excluded.
        full_after_timing = measure(lambda: compiled_full(*network_args))
        print("Full compiled control after stages:", full_after_timing, flush=True)

    flop_path = Path(args.flop_estimate)
    estimate = json.loads(flop_path.read_text())
    if estimate["input_shape"] != list(network_args[0].shape):
        raise ValueError("FLOP estimate shape does not describe benchmark input")
    parts = estimate["parts_flops"]
    semantic_flops = sum(
        parts[k] for k in ("semantic_linear", "semantic_attention", "semantic_embed")
    )
    decoder_flops = sum(parts.values()) - semantic_flops
    peak = estimate["mfu_denominator_tflops_per_second"]

    def stage_result(timing, flops):
        latency_seconds = timing["cuda_ms_median"] / 1000
        return dict(
            timing=timing,
            tflops_per_batch=flops / 1e12,
            model_tflops_per_second=flops / 1e12 / latency_seconds,
            estimated_mfu_percent=100 * flops / 1e12 / latency_seconds / peak,
        )

    stage_sum_ms = backbone_timing["cuda_ms_median"] + decoder_timing["cuda_ms_median"]
    result = dict(
        agent="Cirrus-42 (AI agent)",
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        command=shlex.join(sys.argv),
        source_revision=args.source_revision,
        checkpoint=args.checkpoint,
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        batch_size=args.batch_size,
        tile_size=args.tile_size,
        network_input_shape=list(network_args[0].shape),
        preprocessed_input_shape=list(x.shape),
        conditioning_shape=list(compiled_cond.shape),
        mode="eval, inference_mode, internal BF16, independent default torch.compile stages",
        depthwise_chunk_size=args.depthwise_chunk_size,
        warmup=args.warmup,
        repeats=args.repeats,
        cached_decoder_eager_exact=cached_eager_exact,
        backbone_compiled_vs_eager_error=errors(
            compiled_cond.transpose(1, 2).unsqueeze(-1).unsqueeze(-1),
            eager_cond.transpose(1, 2).unsqueeze(-1).unsqueeze(-1),
        ),
        decoder_composition_vs_eager_strata_error=decoder_error,
        stage_composition_vs_full_compiled_network_error=composition_error,
        backbone_first_call_seconds=backbone_first_seconds,
        decoder_first_call_seconds=decoder_first_seconds,
        backbone=stage_result(backbone_timing, semantic_flops),
        pixel_decoder=stage_result(decoder_timing, decoder_flops),
        full_before=stage_result(full_timing, semantic_flops + decoder_flops),
        full_after=stage_result(full_after_timing, semantic_flops + decoder_flops),
        stage_sum_ms=stage_sum_ms,
        stage_sum_relative_to_full_before=stage_sum_ms / full_timing["cuda_ms_median"],
        stage_sum_relative_to_full_after=stage_sum_ms
        / full_after_timing["cuda_ms_median"],
        mfu_denominator_tflops_per_second=peak,
        scope="Backbone includes patch embedding, positions, and transformer; decoder includes pixel embedding, shared upsampling, all learned conditioning, four pixel blocks, output head. Geometry/wind/latitude wrapper is outside isolated stages. Decoder uses cached compiled-backbone conditioning and original native Strata.forward; the real backbone is restored afterwards. Separate stage compilation may alter fusion and launch boundaries.",
    )
    (output_dir / "stage-results.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Saved stage results:", output_dir, flush=True)
