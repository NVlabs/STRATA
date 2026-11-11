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
"""Compare model stages on identical cached real batch-eight inputs.

Cirrus-42 (AI agent). Timings exclude compilation and loading. Candidate and
control use identical backbone conditioning and BF16 AMP. Raw samples and
per-channel errors are retained in the shared append-only JSONL ledger.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from physicsnemo.experimental.models.strata.layers import _build_chunked_depthwise_conv
from torch._inductor import config as inductor_config

from screamcast import (
    attention_layout,
    column_attention,
    column_gate,
    column_pinned_scale,
    column_precision,
    column_rms_attention,
    conditioning_conv_norm_unroll,
    conditioning_norm,
    conditioning_vector,
    gelu_table,
    gelu_table_layout,
    normalization_loop_schedule,
    normalization_model,
    pixel_axis_attention,
    pixel_norm2_mod,
    pixel_norm_mod,
    pixel_packed_projection_model,
    pixel_persistent,
    pixel_qkv_pack,
    pixel_state_layout_model,
    residual_boundary,
    semantic_attention_layout,
    semantic_gate_fusion,
    semantic_gate_precision,
    semantic_qkv_pack,
    token_layout,
)
from screamcast.benchmark_statistics import latency_gate
from screamcast.conditioning_fused_prepare import (
    component_profile as fused_component_profile,
)
from screamcast.conditioning_fused_prepare import (
    install as install_fused,
)
from screamcast.conditioning_shared_pad import (
    component_profile,
    install,
    restore_precompute,
)
from screamcast.conditioning_spatial_reuse import (
    component_profile as reuse_component_profile,
)
from screamcast.conditioning_spatial_reuse import (
    install as install_spatial_reuse,
)
from screamcast.cuda_timing import measure
from screamcast.decoder_pixel_setup import FULL_BASELINES, apply_cached_baseline
from screamcast.earth2studio_wrappers import ScreamcastModel
from screamcast.forward_stage_benchmark import BackboneFeatures, CachedConditioning
from screamcast.native_reference_schedule import native_reference_schedule


def trace_call(call, path):
    """Record one warmed forward; these timings are excluded from the gate."""
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        call()
        torch.cuda.synchronize()
    profile.export_chrome_trace(path)


def errors(actual, expected):
    diff = actual.float() - expected.float()
    dims = (0, 2, 3, 4)
    per_channel = diff.square().mean(dims).sqrt() / expected.float().square().mean(
        dims
    ).sqrt().clamp_min(1e-12)
    return dict(
        finite=bool(torch.isfinite(actual).all()),
        relative_rms=(
            diff.square().mean().sqrt() / expected.float().square().mean().sqrt()
        ).item(),
        max_channel_relative_rms=per_channel.max().item(),
        channel_relative_rms=per_channel.tolist(),
        max_absolute_difference=diff.abs().max().item(),
    )


def compile_phase(module, inputs, strata, experiment, original_compute, original_convs):
    """Independently seed each phase with the validated native-attention control."""
    torch._dynamo.reset()
    if experiment in (
        "semantic-column-fused-output-gate",
        "semantic-column-rms-attention",
        "semantic-rope-packed",
        "semantic-column-rms-residual-layout",
        "semantic-rope-packed-residual-layout",
        "pixel-qkv-packed",
        "pixel-packed-persistent",
        "pixel-packed-persistent-vector-norm",
        "pixel-fused-conditioning-conv-norm",
        "pixel-norm1-modulation",
        "pixel-norm2-modulation",
        "pixel-attention-axis",
        "pixel-attention-axis-last",
        "pixel-state-row-layout",
        "pixel-packed-output-projection",
        "semantic-column-pinned-scale",
        "semantic-normalization-column-schedules",
        "semantic-gelu-address-layout",
    ):
        compiler_backend = column_gate.backend
    elif experiment == "semantic-native-order-norm":
        compiler_backend = normalization_model.backend
    elif experiment in (
        "semantic-attention-fused-gate",
        "semantic-fused-column-attention",
        "semantic-column-diagnostic",
        "semantic-column-native-sum",
        "semantic-column-token-layout",
        "semantic-persistent-token-layout",
        "semantic-gelu-token-layout",
        "semantic-native-order-norm",
        "semantic-column-fused-output-gate",
        "semantic-column-rms-attention",
        "semantic-rope-packed",
        "semantic-column-rms-residual-layout",
        "semantic-rope-packed-residual-layout",
        "pixel-qkv-packed",
        "pixel-packed-persistent",
        "pixel-packed-persistent-vector-norm",
        "pixel-fused-conditioning-conv-norm",
        "pixel-norm1-modulation",
        "pixel-norm2-modulation",
        "pixel-attention-axis",
        "pixel-attention-axis-last",
        "pixel-state-row-layout",
        "pixel-packed-output-projection",
        "semantic-column-pinned-scale",
        "semantic-normalization-column-schedules",
        "semantic-gelu-address-layout",
        "semantic-column-gelu-table",
        "semantic-column-gelu-all",
        "semantic-persistent-gelu-table",
        "semantic-persistent-gelu-all",
        "semantic-column-persistent",
    ):
        compiler_backend = semantic_gate_fusion.backend
    elif experiment == "semantic-attention-rounded-gate":
        compiler_backend = semantic_gate_precision.backend
    else:
        compiler_backend = "inductor"
    for block, conv, method in zip(
        strata.pixel_blocks, original_convs, original_compute
    ):
        block.adaln_bilinear_dw_conv = conv
        block._compute_bilinear_dw_adaln_params = method
    restore_precompute()
    # This exact control seeds the graph partitions used in the independently
    # confirmed attention experiment. Its conditioning preparation is already
    # validated against the unmodified native learned function.
    apply_experiment(strata, "cuda-fused-prepare-reuse")
    with native_reference_schedule() as schedule:
        seed_output = torch.compile(module, backend=compiler_backend)(*inputs)
    seed_snapshot = seed_output.clone()
    apply_experiment(strata, experiment)
    compiled = torch.compile(module, backend=compiler_backend)
    actual = compiled(*inputs)
    seed_error = errors(actual, seed_snapshot)
    seed_error["reference_schedule"] = dict(schedule)
    seed_error["reference_mutation"] = errors(seed_output, seed_snapshot)
    seed_error["shares_raw_reference_storage"] = (
        actual.data_ptr() == seed_output.data_ptr()
    )
    return compiled, seed_error


def apply_experiment(strata, name):
    pixel_packed_projection_model.restore()
    if name == "pixel-packed-output-projection":
        apply_experiment(strata, "pixel-state-row-layout")
        pixel_packed_projection_model.install(strata)
        return
    pixel_state_layout_model.restore()
    if name == "pixel-state-row-layout":
        apply_experiment(strata, "pixel-attention-axis-last")
        pixel_state_layout_model.install(strata)
        return
    pixel_axis_attention.restore()
    if name in (
        "pixel-attention-axis",
        "pixel-attention-axis-last",
        "pixel-state-row-layout",
        "pixel-packed-output-projection",
    ):
        apply_experiment(strata, "semantic-normalization-column-schedules")
        if name == "pixel-attention-axis-last":
            pixel_axis_attention.install(strata, last_only=True)
        else:
            pixel_axis_attention.install(strata)
        return
    gelu_table_layout.restore()
    if name == "semantic-gelu-address-layout":
        apply_experiment(strata, "semantic-normalization-column-schedules")
        gelu_table_layout.install(strata)
        return
    normalization_loop_schedule.restore()
    if name == "semantic-normalization-column-schedules":
        apply_experiment(strata, "semantic-column-pinned-scale")
        if not isinstance(strata.backbone, CachedConditioning):
            normalization_loop_schedule.install(strata)
        return
    column_pinned_scale.restore()
    if name == "semantic-column-pinned-scale":
        apply_experiment(strata, "pixel-norm2-modulation")
        if not isinstance(strata.backbone, CachedConditioning):
            column_pinned_scale.install(strata)
        return
    pixel_norm2_mod.restore()
    if name == "pixel-norm2-modulation":
        apply_experiment(strata, "pixel-norm1-modulation")
        pixel_norm2_mod.install(strata)
        return
    pixel_norm_mod.restore()
    if name == "pixel-norm1-modulation":
        apply_experiment(strata, "pixel-fused-conditioning-conv-norm")
        pixel_norm_mod.install(strata)
        return
    conditioning_conv_norm_unroll.restore()
    if name == "pixel-fused-conditioning-conv-norm":
        apply_experiment(strata, "pixel-packed-persistent-vector-norm")
        conditioning_conv_norm_unroll.install(strata)
        return
    conditioning_vector.restore()
    if name == "pixel-packed-persistent-vector-norm":
        apply_experiment(strata, "pixel-packed-persistent")
        conditioning_vector.install(strata)
        return
    pixel_persistent.restore()
    if name == "pixel-packed-persistent":
        apply_experiment(strata, "pixel-qkv-packed")
        pixel_persistent.install(strata)
        return
    pixel_qkv_pack.restore()
    if name == "pixel-qkv-packed":
        apply_experiment(strata, "semantic-rope-packed-residual-layout")
        pixel_qkv_pack.install(strata)
        return
    if apply_cached_baseline(strata, name):
        return
    if name in ("semantic-rope-packed", "semantic-rope-packed-residual-layout"):
        base = (
            "semantic-column-rms-residual-layout"
            if name.endswith("residual-layout")
            else "semantic-column-rms-attention"
        )
        apply_experiment(strata, base)
        semantic_qkv_pack.install(strata)
        return
    semantic_qkv_pack.restore()
    if name == "semantic-column-rms-residual-layout":
        apply_experiment(strata, "semantic-column-rms-attention")
        residual_boundary.install(strata)
        return
    residual_boundary.restore()
    if name == "semantic-column-rms-attention":
        apply_experiment(strata, "semantic-column-fused-output-gate")
        column_rms_attention.install(strata)
        return
    column_rms_attention.restore()
    if name == "semantic-column-fused-output-gate":
        apply_experiment(strata, "semantic-native-order-norm")
        return
    normalization_model.restore()
    token_layout.restore()
    gelu_table.restore()
    column_attention.restore()
    attention_layout.restore()
    if name in (
        "semantic-attention-layout",
        "semantic-attention-rounded-gate",
        "semantic-attention-fused-gate",
        "semantic-fused-column-attention",
        "semantic-column-diagnostic",
        "semantic-column-native-sum",
        "semantic-column-token-layout",
        "semantic-persistent-token-layout",
        "semantic-gelu-token-layout",
        "semantic-native-order-norm",
        "semantic-column-fused-output-gate",
        "semantic-column-rms-attention",
        "semantic-rope-packed",
        "semantic-column-rms-residual-layout",
        "semantic-rope-packed-residual-layout",
        "pixel-qkv-packed",
        "pixel-packed-persistent",
        "pixel-packed-persistent-vector-norm",
        "pixel-fused-conditioning-conv-norm",
        "pixel-norm1-modulation",
        "pixel-norm2-modulation",
        "pixel-attention-axis",
        "pixel-attention-axis-last",
        "pixel-state-row-layout",
        "pixel-packed-output-projection",
        "semantic-column-pinned-scale",
        "semantic-normalization-column-schedules",
        "semantic-gelu-address-layout",
        "semantic-column-gelu-table",
        "semantic-column-gelu-all",
        "semantic-persistent-gelu-table",
        "semantic-persistent-gelu-all",
        "semantic-column-persistent",
    ):
        install_spatial_reuse(strata)
        conditioning_norm.install(strata)
        semantic_attention_layout.install(
            strata,
            persistent=name
            in (
                "semantic-column-persistent",
                "semantic-persistent-gelu-table",
                "semantic-persistent-gelu-all",
                "semantic-persistent-token-layout",
                "semantic-gelu-token-layout",
                "semantic-native-order-norm",
                "semantic-column-fused-output-gate",
                "semantic-column-rms-attention",
                "semantic-rope-packed",
                "semantic-column-rms-residual-layout",
                "semantic-rope-packed-residual-layout",
                "pixel-qkv-packed",
                "pixel-packed-persistent",
                "pixel-packed-persistent-vector-norm",
                "pixel-fused-conditioning-conv-norm",
                "pixel-norm1-modulation",
                "pixel-norm2-modulation",
                "pixel-attention-axis",
                "pixel-attention-axis-last",
                "pixel-state-row-layout",
                "pixel-packed-output-projection",
                "semantic-column-pinned-scale",
                "semantic-normalization-column-schedules",
                "semantic-gelu-address-layout",
            ),
        )
        if name == "semantic-fused-column-attention":
            column_attention.install(strata)
        elif name == "semantic-column-diagnostic":
            column_precision.install(strata)
        elif name in (
            "semantic-column-native-sum",
            "semantic-column-token-layout",
            "semantic-persistent-token-layout",
            "semantic-gelu-token-layout",
            "semantic-native-order-norm",
            "semantic-column-fused-output-gate",
            "semantic-column-rms-attention",
            "semantic-rope-packed",
            "semantic-column-rms-residual-layout",
            "semantic-rope-packed-residual-layout",
            "pixel-qkv-packed",
            "pixel-packed-persistent",
            "pixel-packed-persistent-vector-norm",
            "pixel-fused-conditioning-conv-norm",
            "pixel-norm1-modulation",
            "pixel-norm2-modulation",
            "pixel-attention-axis",
            "pixel-attention-axis-last",
            "pixel-state-row-layout",
            "pixel-packed-output-projection",
            "semantic-column-pinned-scale",
            "semantic-normalization-column-schedules",
            "semantic-gelu-address-layout",
            "semantic-column-gelu-table",
            "semantic-column-gelu-all",
            "semantic-persistent-gelu-table",
            "semantic-persistent-gelu-all",
            "semantic-column-persistent",
        ):
            column_precision.install_selected(strata)
        if name in (
            "semantic-column-gelu-table",
            "semantic-column-gelu-all",
            "semantic-persistent-gelu-table",
            "semantic-persistent-gelu-all",
        ):
            gelu_table.install(strata, pixel=name.endswith("gelu-all"))
        if name in ("semantic-gelu-token-layout", "semantic-native-order-norm"):
            gelu_table.install(strata, pixel=True)
        if name in (
            "semantic-column-token-layout",
            "semantic-persistent-token-layout",
            "semantic-gelu-token-layout",
            "semantic-native-order-norm",
            "semantic-column-fused-output-gate",
            "semantic-column-rms-attention",
            "semantic-rope-packed",
            "semantic-column-rms-residual-layout",
            "semantic-rope-packed-residual-layout",
            "pixel-qkv-packed",
            "pixel-packed-persistent",
            "pixel-packed-persistent-vector-norm",
            "pixel-fused-conditioning-conv-norm",
            "pixel-norm1-modulation",
            "pixel-norm2-modulation",
            "pixel-attention-axis",
            "pixel-attention-axis-last",
            "pixel-state-row-layout",
            "pixel-packed-output-projection",
            "semantic-column-pinned-scale",
            "semantic-normalization-column-schedules",
            "semantic-gelu-address-layout",
        ):
            token_layout.install(strata)
        if name == "semantic-native-order-norm":
            normalization_model.install(strata)
    elif name == "conditioning-norm-row1":
        install_spatial_reuse(strata)
        attention_layout.install(strata)
        conditioning_norm.install(strata)
    elif name == "attention-layout":
        install_spatial_reuse(strata)
        attention_layout.install(strata)
    elif name == "cuda-fused-prepare-reuse":
        install_spatial_reuse(strata)
    elif name == "cuda-fused-prepare-wide":
        install_fused(strata)
    elif name == "cuda-shared-pad":
        install(strata)
    elif name != "baseline":
        raise ValueError(f"Unknown experiment: {name}")


def run(args):
    if args.experiment == "semantic-gelu-address-layout" and args.scope != "backbone":
        raise ValueError("GELU address layout requires backbone isolation")
    if (
        args.experiment
        in (
            "semantic-column-pinned-scale",
            "semantic-normalization-column-schedules",
            "semantic-gelu-address-layout",
        )
        and args.scope != "backbone"
    ):
        raise ValueError("Rounded-scale column trial requires backbone isolation")
    pixel_trial = args.experiment in (
        "pixel-qkv-packed",
        "pixel-packed-persistent",
        "pixel-packed-persistent-vector-norm",
        "pixel-fused-conditioning-conv-norm",
        "pixel-norm1-modulation",
        "pixel-norm2-modulation",
        "pixel-attention-axis",
        "pixel-attention-axis-last",
        "pixel-state-row-layout",
        "pixel-packed-output-projection",
    )
    if pixel_trial and (
        args.scope != "decoder"
        or (
            args.baseline_experiment not in FULL_BASELINES
            and args.baseline_experiment
            not in (
                "pixel-qkv-packed",
                "pixel-packed-persistent",
                "pixel-packed-persistent-vector-norm",
                "pixel-fused-conditioning-conv-norm",
                "pixel-norm1-modulation",
                "pixel-norm2-modulation",
                "pixel-attention-axis",
                "pixel-attention-axis-last",
                "pixel-state-row-layout",
                "pixel-packed-output-projection",
                "semantic-column-pinned-scale",
                "semantic-normalization-column-schedules",
                "semantic-gelu-address-layout",
            )
        )
    ):
        raise ValueError(
            "Pixel packing requires decoder isolation and a supported full baseline"
        )
    if (
        not pixel_trial
        and args.scope != "backbone"
        and any(
            name
            in (
                "semantic-attention-layout",
                "semantic-attention-rounded-gate",
                "semantic-attention-fused-gate",
                "semantic-fused-column-attention",
                "semantic-column-diagnostic",
                "semantic-column-native-sum",
                "semantic-column-token-layout",
                "semantic-persistent-token-layout",
                "semantic-gelu-token-layout",
                "semantic-native-order-norm",
                "semantic-column-fused-output-gate",
                "semantic-column-rms-attention",
                "semantic-rope-packed",
                "semantic-column-rms-residual-layout",
                "semantic-rope-packed-residual-layout",
                "pixel-qkv-packed",
                "pixel-packed-persistent",
                "pixel-packed-persistent-vector-norm",
                "pixel-fused-conditioning-conv-norm",
                "pixel-norm1-modulation",
                "pixel-norm2-modulation",
                "pixel-attention-axis",
                "pixel-attention-axis-last",
                "pixel-state-row-layout",
                "pixel-packed-output-projection",
                "semantic-column-pinned-scale",
                "semantic-normalization-column-schedules",
                "semantic-gelu-address-layout",
                "semantic-column-gelu-table",
                "semantic-column-gelu-all",
                "semantic-persistent-gelu-table",
                "semantic-persistent-gelu-all",
                "semantic-column-persistent",
            )
            for name in (args.experiment, args.baseline_experiment)
        )
    ):
        raise ValueError("Semantic attention experiments require --scope backbone")
    cached = torch.load(args.inputs, map_location="cpu", weights_only=True)
    if (cached["batch_size"], cached["tile_size"]) != (8, 128):
        raise ValueError("This experiment requires batch 8 and tile 128")
    if args.checkpoint != cached["checkpoint"]:
        raise ValueError("Checkpoint differs from the fixed input capture")
    with open(args.inputs, "rb") as stream:
        input_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    progress = dict(
        agent="Cirrus-42 (AI agent)",
        commit=args.commit,
        experiment=args.experiment,
        compiler_threads=inductor_config.compile_threads,
        column_qk_rms_fused=args.experiment
        in (
            "semantic-column-rms-attention",
            "semantic-column-rms-residual-layout",
            "semantic-rope-packed",
            "semantic-rope-packed-residual-layout",
            "pixel-qkv-packed",
            "pixel-packed-persistent",
            "pixel-packed-persistent-vector-norm",
            "pixel-fused-conditioning-conv-norm",
            "pixel-norm1-modulation",
            "pixel-norm2-modulation",
            "pixel-attention-axis",
            "pixel-attention-axis-last",
            "pixel-state-row-layout",
            "pixel-packed-output-projection",
            "semantic-column-pinned-scale",
            "semantic-normalization-column-schedules",
            "semantic-gelu-address-layout",
        ),
        pixel_projected_output_packed=args.experiment
        == "pixel-packed-output-projection",
        pixel_packed_update_rows=pixel_packed_projection_model.UPDATE_ROWS
        if args.experiment == "pixel-packed-output-projection"
        else None,
        pixel_state_row_layout=args.experiment
        in ("pixel-state-row-layout", "pixel-packed-output-projection"),
        pixel_state_norm1_rows=32
        if args.experiment
        in ("pixel-state-row-layout", "pixel-packed-output-projection")
        else None,
        pixel_attention_axis_blocks=[3]
        if args.experiment
        in (
            "pixel-attention-axis-last",
            "pixel-state-row-layout",
            "pixel-packed-output-projection",
        )
        else [0, 1, 2, 3]
        if args.experiment == "pixel-attention-axis"
        else [],
        pixel_attention_axis=args.experiment
        in (
            "pixel-attention-axis",
            "pixel-attention-axis-last",
            "pixel-state-row-layout",
            "pixel-packed-output-projection",
        ),
        pixel_attention_physical_shape=[128, 128, 24]
        if args.experiment
        in (
            "pixel-attention-axis",
            "pixel-attention-axis-last",
            "pixel-state-row-layout",
            "pixel-packed-output-projection",
        )
        else [24, 128, 128],
        pixel_qkv_packed=args.experiment
        in (
            "pixel-qkv-packed",
            "pixel-packed-persistent",
            "pixel-packed-persistent-vector-norm",
            "pixel-fused-conditioning-conv-norm",
            "pixel-norm1-modulation",
            "pixel-norm2-modulation",
            "pixel-attention-axis",
            "pixel-attention-axis-last",
            "pixel-state-row-layout",
            "pixel-packed-output-projection",
            "semantic-column-pinned-scale",
            "semantic-normalization-column-schedules",
            "semantic-gelu-address-layout",
        ),
        pixel_attention_persistent=args.experiment
        in (
            "pixel-packed-persistent",
            "pixel-packed-persistent-vector-norm",
            "pixel-fused-conditioning-conv-norm",
            "pixel-norm1-modulation",
            "pixel-norm2-modulation",
            "pixel-attention-axis",
            "pixel-attention-axis-last",
            "pixel-state-row-layout",
            "pixel-packed-output-projection",
            "semantic-column-pinned-scale",
            "semantic-normalization-column-schedules",
            "semantic-gelu-address-layout",
        ),
        conditioning_conv_norm_fused=args.experiment
        in (
            "pixel-fused-conditioning-conv-norm",
            "pixel-norm1-modulation",
            "pixel-norm2-modulation",
            "pixel-attention-axis",
            "pixel-attention-axis-last",
            "pixel-state-row-layout",
            "pixel-packed-output-projection",
            "semantic-column-pinned-scale",
            "semantic-normalization-column-schedules",
            "semantic-gelu-address-layout",
        ),
        pixel_norm1_modulated=args.experiment
        in (
            "pixel-norm1-modulation",
            "pixel-norm2-modulation",
            "pixel-attention-axis",
            "pixel-attention-axis-last",
            "pixel-state-row-layout",
            "pixel-packed-output-projection",
            "semantic-column-pinned-scale",
            "semantic-normalization-column-schedules",
            "semantic-gelu-address-layout",
        ),
        pixel_norm2_modulated=args.experiment
        in (
            "pixel-norm2-modulation",
            "pixel-attention-axis",
            "pixel-attention-axis-last",
            "pixel-state-row-layout",
            "pixel-packed-output-projection",
            "semantic-column-pinned-scale",
            "semantic-normalization-column-schedules",
            "semantic-gelu-address-layout",
        ),
        column_pinned_scale=args.experiment
        in (
            "pixel-attention-axis",
            "pixel-attention-axis-last",
            "pixel-state-row-layout",
            "pixel-packed-output-projection",
        )
        or args.experiment
        in (
            "semantic-column-pinned-scale",
            "semantic-normalization-column-schedules",
            "semantic-gelu-address-layout",
        ),
        gelu_address_layout=args.experiment == "semantic-gelu-address-layout",
        normalization_loops_unrolled=args.experiment
        in (
            "semantic-normalization-column-schedules",
            "semantic-gelu-address-layout",
            "pixel-attention-axis",
            "pixel-attention-axis-last",
            "pixel-state-row-layout",
            "pixel-packed-output-projection",
        ),
        conditioning_norm_vectorized=args.experiment
        == "pixel-packed-persistent-vector-norm",
        semantic_qkv_packed=args.experiment
        in (
            "semantic-rope-packed",
            "semantic-rope-packed-residual-layout",
            "pixel-qkv-packed",
            "pixel-packed-persistent",
            "pixel-packed-persistent-vector-norm",
            "pixel-fused-conditioning-conv-norm",
            "pixel-norm1-modulation",
            "pixel-norm2-modulation",
            "pixel-attention-axis",
            "pixel-attention-axis-last",
            "pixel-state-row-layout",
            "pixel-packed-output-projection",
            "semantic-column-pinned-scale",
            "semantic-normalization-column-schedules",
            "semantic-gelu-address-layout",
        ),
        baseline_experiment=args.baseline_experiment,
        baseline_commit=args.baseline_commit,
        checkpoint=args.checkpoint,
        inputs=args.inputs,
        input_sha256=input_sha256,
    )

    def record(stage, **values):
        progress.update(
            values, stage=stage, time=datetime.now(timezone.utc).isoformat()
        )
        if args.trace_prefix:
            destination = Path(args.trace_prefix + "-progress.json")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(progress) + "\n")
            temporary.replace(destination)

    record(
        "initializing",
        inductor_cudagraphs=bool(inductor_config.triton.cudagraphs),
        canonical_native_residual_schedule={
            "XBLOCK": 8,
            "R0_BLOCK": 256,
            "num_warps": 16,
        },
    )
    model = ScreamcastModel.from_checkpoint(args.checkpoint, bf16=True).cuda().eval()
    model.set_tile_size(128)
    network = model.pipeline.network
    strata = network.strata
    for block in strata.pixel_blocks:
        conv = block.adaln_bilinear_dw_conv
        conv.chunk_size = 512
        conv._chunked_conv = _build_chunked_depthwise_conv(conv, 512)
    x, pos = cached["x"].cuda(), cached["pos"].cuda()
    conditioning = cached["conditioning"].cuda()
    network_args = (
        cached["network_args"][0].cuda(),
        {k: v.cuda() for k, v in cached["network_args"][1].items()},
    )
    progress["input_is_inference"] = {
        "decoder_x": x.is_inference(),
        "decoder_pos": pos.is_inference(),
        "network_x": network_args[0].is_inference(),
        "network_index": {k: v.is_inference() for k, v in network_args[1].items()},
    }
    latent_dhw = cached["latent_dhw"]
    del cached
    backbone = strata.backbone
    if args.scope == "decoder":
        strata.backbone = CachedConditioning(conditioning, latent_dhw)
        isolated = strata
    else:
        isolated = BackboneFeatures(backbone).eval()
    original_compute = [
        b._compute_bilinear_dw_adaln_params for b in strata.pixel_blocks
    ]
    original_convs = [b.adaln_bilinear_dw_conv for b in strata.pixel_blocks]
    apply_experiment(strata, args.baseline_experiment)
    if args.numerical_only:
        with torch.inference_mode():
            record("isolated_numerical_start", diagnostic_only=True)
            control, seed_control = compile_phase(
                isolated,
                (x, pos),
                strata,
                args.baseline_experiment,
                original_compute,
                original_convs,
            )
            expected = control(x, pos).clone()
            changed, seed_changed = compile_phase(
                isolated,
                (x, pos),
                strata,
                args.experiment,
                original_compute,
                original_convs,
            )
            isolated_error = errors(changed(x, pos), expected)
            isolated_ok = all(
                e["finite"]
                and e["relative_rms"] <= 0.005
                and e["max_channel_relative_rms"] <= 0.01
                for e in (seed_control, seed_changed, isolated_error)
            )
            record(
                "isolated_numerical_complete",
                diagnostic_only=True,
                isolated_error=isolated_error,
                canonical_isolated_before=seed_control,
                canonical_isolated_candidate=seed_changed,
                isolated_numerical_passed=isolated_ok,
            )
            print(
                "NUMERICAL_ONLY_ISOLATED",
                isolated_ok,
                {
                    k: v
                    for k, v in isolated_error.items()
                    if k != "channel_relative_rms"
                },
                flush=True,
            )
            if not isolated_ok:
                record("numerical_only_rejected", diagnostic_only=True)
                return
            strata.backbone = backbone
            control, seed_control = compile_phase(
                network,
                network_args,
                strata,
                args.baseline_experiment,
                original_compute,
                original_convs,
            )
            expected = control(*network_args).clone()
            changed, seed_changed = compile_phase(
                network,
                network_args,
                strata,
                args.experiment,
                original_compute,
                original_convs,
            )
            full_error = errors(changed(*network_args), expected)
            full_ok = all(
                e["finite"]
                and e["relative_rms"] <= 0.005
                and e["max_channel_relative_rms"] <= 0.01
                for e in (seed_control, seed_changed, full_error)
            )
            record(
                "numerical_only_complete",
                diagnostic_only=True,
                full_error=full_error,
                canonical_full_before=seed_control,
                canonical_full_candidate=seed_changed,
                full_numerical_passed=full_ok,
            )
            print(
                "NUMERICAL_ONLY_FULL",
                full_ok,
                {k: v for k, v in full_error.items() if k != "channel_relative_rms"},
                flush=True,
            )
        return
    with torch.inference_mode():
        baseline, seed_before_error = compile_phase(
            isolated,
            (x, pos),
            strata,
            args.baseline_experiment,
            original_compute,
            original_convs,
        )
        expected = baseline(x, pos).clone()
        before = measure(lambda: baseline(x, pos), args.warmup, args.repeats)
        record(
            "isolated_baseline_before",
            baseline_before=before,
            canonical_isolated_before=seed_before_error,
        )
        print(f"Baseline {args.scope} before:", before["median_ms"], flush=True)
        if args.trace_prefix:
            trace_call(lambda: baseline(x, pos), args.trace_prefix + "-baseline.json")
        # Restore original callables after each candidate, before the trailing control.
        parameter_ids = {name: id(p) for name, p in strata.named_parameters()}
        apply_experiment(strata, args.experiment)
        if parameter_ids != {name: id(p) for name, p in strata.named_parameters()}:
            raise RuntimeError("Experiment changed parameter names or objects")
        if args.experiment == "cuda-shared-pad":
            component_profile(strata, conditioning, latent_dhw, args, measure)
            if args.profile_traffic:
                return
        if args.experiment == "cuda-fused-prepare-wide":
            fused_component_profile(conditioning, latent_dhw, args, measure)
        if args.experiment == "cuda-fused-prepare-reuse":
            reuse_component_profile(conditioning, args, measure)
        candidate, seed_candidate_error = compile_phase(
            isolated,
            (x, pos),
            strata,
            args.experiment,
            original_compute,
            original_convs,
        )
        actual = candidate(x, pos)
        numerical = errors(actual, expected)
        candidate_timing = measure(lambda: candidate(x, pos), args.warmup, args.repeats)
        record(
            "isolated_candidate",
            candidate=candidate_timing,
            isolated_error=numerical,
            canonical_isolated_candidate=seed_candidate_error,
        )
        print(
            f"Candidate {args.scope}:",
            candidate_timing["median_ms"],
            {k: v for k, v in numerical.items() if k != "channel_relative_rms"},
            flush=True,
        )
        if args.trace_prefix:
            trace_call(lambda: candidate(x, pos), args.trace_prefix + "-candidate.json")
        for block, conv in zip(strata.pixel_blocks, original_convs):
            block.adaln_bilinear_dw_conv = conv
        for block, method in zip(strata.pixel_blocks, original_compute):
            block._compute_bilinear_dw_adaln_params = method
        restore_precompute()
        apply_experiment(strata, args.baseline_experiment)
        baseline, seed_after_error = compile_phase(
            isolated,
            (x, pos),
            strata,
            args.baseline_experiment,
            original_compute,
            original_convs,
        )
        after_error = errors(baseline(x, pos), expected)
        after = measure(lambda: baseline(x, pos), args.warmup, args.repeats)
        record(
            "isolated_baseline_after",
            baseline_after=after,
            isolated_control_error=after_error,
            canonical_isolated_after=seed_after_error,
        )
        strata.backbone = backbone
        parameter_ids = {name: id(p) for name, p in strata.named_parameters()}
        full_baseline, full_seed_before_error = compile_phase(
            network,
            network_args,
            strata,
            args.baseline_experiment,
            original_compute,
            original_convs,
        )
        full_expected = full_baseline(*network_args).clone()
        full_before = measure(
            lambda: full_baseline(*network_args), args.warmup, args.repeats
        )
        record(
            "full_baseline_before",
            full_baseline_before=full_before,
            canonical_full_before=full_seed_before_error,
        )
        print("Full baseline before:", full_before["median_ms"], flush=True)
        apply_experiment(strata, args.experiment)
        if parameter_ids != {name: id(p) for name, p in strata.named_parameters()}:
            raise RuntimeError("Experiment changed parameter names or objects")
        full_candidate, full_seed_candidate_error = compile_phase(
            network,
            network_args,
            strata,
            args.experiment,
            original_compute,
            original_convs,
        )
        full_actual = full_candidate(*network_args)
        full_error = errors(full_actual, full_expected)
        full_timing = measure(
            lambda: full_candidate(*network_args), args.warmup, args.repeats
        )
        record(
            "full_candidate",
            full_candidate=full_timing,
            full_error=full_error,
            canonical_full_candidate=full_seed_candidate_error,
        )
        print(
            "Full candidate:",
            full_timing["median_ms"],
            {k: v for k, v in full_error.items() if k != "channel_relative_rms"},
            flush=True,
        )
        for block, conv in zip(strata.pixel_blocks, original_convs):
            block.adaln_bilinear_dw_conv = conv
        for block, method in zip(strata.pixel_blocks, original_compute):
            block._compute_bilinear_dw_adaln_params = method
        restore_precompute()
        apply_experiment(strata, args.baseline_experiment)
        full_baseline, full_seed_after_error = compile_phase(
            network,
            network_args,
            strata,
            args.baseline_experiment,
            original_compute,
            original_convs,
        )
        full_after_error = errors(full_baseline(*network_args), full_expected)
        full_after = measure(
            lambda: full_baseline(*network_args), args.warmup, args.repeats
        )
        record(
            "full_baseline_after",
            full_baseline_after=full_after,
            full_control_error=full_after_error,
            canonical_full_after=full_seed_after_error,
        )
        print("Full baseline after:", full_after["median_ms"], flush=True)
    numerical_pass = all(
        e["finite"]
        and e["relative_rms"] <= 0.005
        and e["max_channel_relative_rms"] <= 0.01
        for e in (
            numerical,
            full_error,
            after_error,
            full_after_error,
            seed_before_error,
            seed_candidate_error,
            seed_after_error,
            full_seed_before_error,
            full_seed_candidate_error,
            full_seed_after_error,
        )
    )
    performance = latency_gate(
        dict(before=before, candidate=candidate_timing, after=after),
        dict(before=full_before, candidate=full_timing, after=full_after),
        isolated_scope=args.scope,
    )
    faster = performance["significant"]
    result = dict(
        time=datetime.now(timezone.utc).isoformat(),
        agent="Cirrus-42 (AI agent)",
        experiment=args.experiment,
        compiler_threads=inductor_config.compile_threads,
        semantic_attention_persistent=args.experiment
        in (
            "semantic-column-persistent",
            "semantic-persistent-gelu-table",
            "semantic-persistent-gelu-all",
            "semantic-persistent-token-layout",
            "semantic-gelu-token-layout",
            "semantic-native-order-norm",
            "semantic-column-fused-output-gate",
            "semantic-column-rms-attention",
            "semantic-rope-packed",
            "semantic-column-rms-residual-layout",
            "semantic-rope-packed-residual-layout",
            "pixel-qkv-packed",
            "pixel-packed-persistent",
            "pixel-packed-persistent-vector-norm",
            "pixel-fused-conditioning-conv-norm",
            "pixel-norm1-modulation",
            "pixel-norm2-modulation",
            "pixel-attention-axis",
            "pixel-attention-axis-last",
            "pixel-state-row-layout",
            "pixel-packed-output-projection",
            "semantic-column-pinned-scale",
            "semantic-normalization-column-schedules",
            "semantic-gelu-address-layout",
        ),
        backbone_normalization=(
            "native Welford order plus immediate BF16 linear input cast"
            if args.experiment
            in (
                "semantic-native-order-norm",
                "semantic-column-fused-output-gate",
                "semantic-column-rms-attention",
                "semantic-rope-packed",
                "semantic-column-rms-residual-layout",
                "semantic-rope-packed-residual-layout",
                "pixel-qkv-packed",
                "pixel-packed-persistent",
                "pixel-packed-persistent-vector-norm",
                "pixel-fused-conditioning-conv-norm",
                "pixel-norm1-modulation",
                "pixel-norm2-modulation",
                "pixel-attention-axis",
                "pixel-attention-axis-last",
                "pixel-state-row-layout",
                "pixel-packed-output-projection",
                "semantic-column-pinned-scale",
                "semantic-normalization-column-schedules",
                "semantic-gelu-address-layout",
            )
            else "native"
        ),
        backbone_token_layout=args.experiment
        in (
            "semantic-column-token-layout",
            "semantic-persistent-token-layout",
            "semantic-gelu-token-layout",
            "semantic-native-order-norm",
            "semantic-column-fused-output-gate",
            "semantic-column-rms-attention",
            "semantic-rope-packed",
            "semantic-column-rms-residual-layout",
            "semantic-rope-packed-residual-layout",
            "pixel-qkv-packed",
            "pixel-packed-persistent",
            "pixel-packed-persistent-vector-norm",
            "pixel-fused-conditioning-conv-norm",
            "pixel-norm1-modulation",
            "pixel-norm2-modulation",
            "pixel-attention-axis",
            "pixel-attention-axis-last",
            "pixel-state-row-layout",
            "pixel-packed-output-projection",
            "semantic-column-pinned-scale",
            "semantic-normalization-column-schedules",
            "semantic-gelu-address-layout",
        ),
        backbone_gelu_lookup=args.experiment
        in (
            "semantic-column-gelu-table",
            "semantic-column-gelu-all",
            "semantic-persistent-gelu-table",
            "semantic-persistent-gelu-all",
            "semantic-gelu-token-layout",
            "semantic-native-order-norm",
            "semantic-column-fused-output-gate",
            "semantic-column-rms-attention",
            "semantic-rope-packed",
            "semantic-column-rms-residual-layout",
            "semantic-rope-packed-residual-layout",
            "pixel-qkv-packed",
            "pixel-packed-persistent",
            "pixel-packed-persistent-vector-norm",
            "pixel-fused-conditioning-conv-norm",
            "pixel-norm1-modulation",
            "pixel-norm2-modulation",
            "pixel-attention-axis",
            "pixel-attention-axis-last",
            "pixel-state-row-layout",
            "pixel-packed-output-projection",
            "semantic-column-pinned-scale",
            "semantic-normalization-column-schedules",
            "semantic-gelu-address-layout",
        ),
        pixel_gelu_lookup=args.experiment
        in (
            "semantic-column-gelu-all",
            "semantic-persistent-gelu-all",
            "semantic-gelu-token-layout",
            "semantic-native-order-norm",
            "semantic-column-fused-output-gate",
            "semantic-column-rms-attention",
            "semantic-rope-packed",
            "semantic-column-rms-residual-layout",
            "semantic-rope-packed-residual-layout",
            "pixel-qkv-packed",
            "pixel-packed-persistent",
            "pixel-packed-persistent-vector-norm",
            "pixel-fused-conditioning-conv-norm",
            "pixel-norm1-modulation",
            "pixel-norm2-modulation",
            "pixel-attention-axis",
            "pixel-attention-axis-last",
            "pixel-state-row-layout",
            "pixel-packed-output-projection",
            "semantic-column-pinned-scale",
            "semantic-normalization-column-schedules",
            "semantic-gelu-address-layout",
        ),
        column_qk_rms_fused=args.experiment
        in (
            "semantic-column-rms-attention",
            "semantic-column-rms-residual-layout",
            "semantic-rope-packed",
            "semantic-rope-packed-residual-layout",
            "pixel-qkv-packed",
            "pixel-packed-persistent",
            "pixel-packed-persistent-vector-norm",
            "pixel-fused-conditioning-conv-norm",
            "pixel-norm1-modulation",
            "pixel-norm2-modulation",
            "pixel-attention-axis",
            "pixel-attention-axis-last",
            "pixel-state-row-layout",
            "pixel-packed-output-projection",
            "semantic-column-pinned-scale",
            "semantic-normalization-column-schedules",
            "semantic-gelu-address-layout",
        ),
        pixel_projected_output_packed=args.experiment
        == "pixel-packed-output-projection",
        pixel_packed_update_rows=pixel_packed_projection_model.UPDATE_ROWS
        if args.experiment == "pixel-packed-output-projection"
        else None,
        pixel_state_row_layout=args.experiment
        in ("pixel-state-row-layout", "pixel-packed-output-projection"),
        pixel_state_norm1_rows=32
        if args.experiment
        in ("pixel-state-row-layout", "pixel-packed-output-projection")
        else None,
        pixel_attention_axis_blocks=[3]
        if args.experiment
        in (
            "pixel-attention-axis-last",
            "pixel-state-row-layout",
            "pixel-packed-output-projection",
        )
        else [0, 1, 2, 3]
        if args.experiment == "pixel-attention-axis"
        else [],
        pixel_attention_axis=args.experiment
        in (
            "pixel-attention-axis",
            "pixel-attention-axis-last",
            "pixel-state-row-layout",
            "pixel-packed-output-projection",
        ),
        pixel_attention_physical_shape=[128, 128, 24]
        if args.experiment
        in (
            "pixel-attention-axis",
            "pixel-attention-axis-last",
            "pixel-state-row-layout",
            "pixel-packed-output-projection",
        )
        else [24, 128, 128],
        pixel_qkv_packed=args.experiment
        in (
            "pixel-qkv-packed",
            "pixel-packed-persistent",
            "pixel-packed-persistent-vector-norm",
            "pixel-fused-conditioning-conv-norm",
            "pixel-norm1-modulation",
            "pixel-norm2-modulation",
            "pixel-attention-axis",
            "pixel-attention-axis-last",
            "pixel-state-row-layout",
            "pixel-packed-output-projection",
            "semantic-column-pinned-scale",
            "semantic-normalization-column-schedules",
            "semantic-gelu-address-layout",
        ),
        pixel_attention_persistent=args.experiment
        in (
            "pixel-packed-persistent",
            "pixel-packed-persistent-vector-norm",
            "pixel-fused-conditioning-conv-norm",
            "pixel-norm1-modulation",
            "pixel-norm2-modulation",
            "pixel-attention-axis",
            "pixel-attention-axis-last",
            "pixel-state-row-layout",
            "pixel-packed-output-projection",
            "semantic-column-pinned-scale",
            "semantic-normalization-column-schedules",
            "semantic-gelu-address-layout",
        ),
        conditioning_conv_norm_fused=args.experiment
        in (
            "pixel-fused-conditioning-conv-norm",
            "pixel-norm1-modulation",
            "pixel-norm2-modulation",
            "pixel-attention-axis",
            "pixel-attention-axis-last",
            "pixel-state-row-layout",
            "pixel-packed-output-projection",
            "semantic-column-pinned-scale",
            "semantic-normalization-column-schedules",
            "semantic-gelu-address-layout",
        ),
        pixel_norm1_modulated=args.experiment
        in (
            "pixel-norm1-modulation",
            "pixel-norm2-modulation",
            "pixel-attention-axis",
            "pixel-attention-axis-last",
            "pixel-state-row-layout",
            "pixel-packed-output-projection",
            "semantic-column-pinned-scale",
            "semantic-normalization-column-schedules",
            "semantic-gelu-address-layout",
        ),
        pixel_norm2_modulated=args.experiment
        in (
            "pixel-norm2-modulation",
            "pixel-attention-axis",
            "pixel-attention-axis-last",
            "pixel-state-row-layout",
            "pixel-packed-output-projection",
            "semantic-column-pinned-scale",
            "semantic-normalization-column-schedules",
            "semantic-gelu-address-layout",
        ),
        column_pinned_scale=args.experiment
        in (
            "pixel-attention-axis",
            "pixel-attention-axis-last",
            "pixel-state-row-layout",
            "pixel-packed-output-projection",
        )
        or args.experiment
        in (
            "semantic-column-pinned-scale",
            "semantic-normalization-column-schedules",
            "semantic-gelu-address-layout",
        ),
        gelu_address_layout=args.experiment == "semantic-gelu-address-layout",
        normalization_loops_unrolled=args.experiment
        in (
            "semantic-normalization-column-schedules",
            "semantic-gelu-address-layout",
            "pixel-attention-axis",
            "pixel-attention-axis-last",
            "pixel-state-row-layout",
            "pixel-packed-output-projection",
        ),
        conditioning_norm_vectorized=args.experiment
        == "pixel-packed-persistent-vector-norm",
        semantic_qkv_packed=args.experiment
        in (
            "semantic-rope-packed",
            "semantic-rope-packed-residual-layout",
            "pixel-qkv-packed",
            "pixel-packed-persistent",
            "pixel-packed-persistent-vector-norm",
            "pixel-fused-conditioning-conv-norm",
            "pixel-norm1-modulation",
            "pixel-norm2-modulation",
            "pixel-attention-axis",
            "pixel-attention-axis-last",
            "pixel-state-row-layout",
            "pixel-packed-output-projection",
            "semantic-column-pinned-scale",
            "semantic-normalization-column-schedules",
            "semantic-gelu-address-layout",
        ),
        baseline_experiment=args.baseline_experiment,
        baseline_commit=args.baseline_commit,
        commit=args.commit,
        batch_size=8,
        tile_size=128,
        amp="bf16",
        mode="eval, inference_mode, internal BF16 AMP, torch.compile default",
        compiler_cache_protocol="Dynamo reset each phase; seed cuda-fused-prepare-reuse once; independently install and compile selected experiment; warmups exclude compilation",
        inputs=args.inputs,
        input_sha256=input_sha256,
        checkpoint=args.checkpoint,
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        trace_prefix=args.trace_prefix,
        warmup=args.warmup,
        repeats=args.repeats,
        baseline_before=before,
        candidate=candidate_timing,
        baseline_after=after,
        full_baseline_before=full_before,
        full_candidate=full_timing,
        full_baseline_after=full_after,
        compiler_transform=(
            "fuse output layout with rounded semantic sigmoid"
            if args.experiment
            in (
                "semantic-attention-fused-gate",
                "semantic-fused-column-attention",
                "semantic-column-native-sum",
                "semantic-column-token-layout",
                "semantic-persistent-token-layout",
                "semantic-gelu-token-layout",
                "semantic-native-order-norm",
                "semantic-column-fused-output-gate",
                "semantic-column-rms-attention",
                "semantic-rope-packed",
                "semantic-column-rms-residual-layout",
                "semantic-rope-packed-residual-layout",
                "pixel-qkv-packed",
                "pixel-packed-persistent",
                "pixel-packed-persistent-vector-norm",
                "pixel-fused-conditioning-conv-norm",
                "pixel-norm1-modulation",
                "pixel-norm2-modulation",
                "pixel-attention-axis",
                "pixel-attention-axis-last",
                "pixel-state-row-layout",
                "pixel-packed-output-projection",
                "semantic-column-pinned-scale",
                "semantic-normalization-column-schedules",
                "semantic-gelu-address-layout",
                "semantic-column-gelu-table",
                "semantic-column-gelu-all",
                "semantic-persistent-gelu-table",
                "semantic-persistent-gelu-all",
                "semantic-column-persistent",
            )
            else "preserve semantic BF16 sigmoid gate"
            if args.experiment == "semantic-attention-rounded-gate"
            else "none"
        ),
        column_attention_kernel=(
            "24-level Triton, log2 scores; six-term per-lane sequential FP32 sum and adjacent lane-pair reduction; BF16 unnormalized probabilities, normalize after value dot"
            if args.experiment
            in (
                "semantic-column-native-sum",
                "semantic-column-token-layout",
                "semantic-persistent-token-layout",
                "semantic-gelu-token-layout",
                "semantic-native-order-norm",
                "semantic-column-fused-output-gate",
                "semantic-column-rms-attention",
                "semantic-rope-packed",
                "semantic-column-rms-residual-layout",
                "semantic-rope-packed-residual-layout",
                "pixel-qkv-packed",
                "pixel-packed-persistent",
                "pixel-packed-persistent-vector-norm",
                "pixel-fused-conditioning-conv-norm",
                "pixel-norm1-modulation",
                "pixel-norm2-modulation",
                "pixel-attention-axis",
                "pixel-attention-axis-last",
                "pixel-state-row-layout",
                "pixel-packed-output-projection",
                "semantic-column-pinned-scale",
                "semantic-normalization-column-schedules",
                "semantic-gelu-address-layout",
                "semantic-column-gelu-table",
                "semantic-column-gelu-all",
                "semantic-persistent-gelu-table",
                "semantic-persistent-gelu-all",
                "semantic-column-persistent",
            )
            else "24-level Triton, BF16 unnormalized probabilities; normalize after value dot"
            if args.experiment == "semantic-fused-column-attention"
            else "native SDPA"
        ),
        isolated_scope=args.scope,
        isolated_output_shape=list(expected.shape),
        isolated_error=numerical,
        full_error=full_error,
        control_after_error={args.scope: after_error, "full": full_after_error},
        canonical_reference_error={
            f"{args.scope}_before": seed_before_error,
            f"{args.scope}_candidate": seed_candidate_error,
            f"{args.scope}_after": seed_after_error,
            "full_before": full_seed_before_error,
            "full_candidate": full_seed_candidate_error,
            "full_after": full_seed_after_error,
        },
        numerical_gate="overall relative RMS <=0.5%, every channel <=1%, all finite",
        numerical_pass=numerical_pass,
        faster=faster,
        performance_gate=performance,
        decision="baseline"
        if args.experiment == args.baseline_experiment
        else "accept"
        if numerical_pass and faster
        else "reject",
    )
    result[f"{args.scope}_error"] = numerical
    output = Path(args.results)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a") as f:
        f.write(json.dumps(result) + "\n")
    record(
        "complete",
        decision=result["decision"],
        numerical_pass=numerical_pass,
        performance_gate=performance,
    )
    print("RESULT", json.dumps(result), flush=True)
