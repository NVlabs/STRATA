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
"""Profile compiled STRATA inference on a real PBSS tile without module hooks.

Run from the repo in its GPU environment, with AUX_DATA_ROOT and SCREAM_* configured:
  PYTHONPATH=. python scripts/benchmark_pixeldit_forward.py --help
"""

import json
import shlex
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from physicsnemo.experimental.models.strata.layers import (
    DepthwiseConv,
    _build_chunked_depthwise_conv,
)
from torch._dynamo.utils import counters
from torch.profiler import ProfilerActivity, profile

import data_catalog
from screamcast.earth2studio_wrappers import ScreamcastModel
from screamcast.forward_stage_benchmark import run_stage_benchmark


def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Importing model and loading checkpoint", flush=True)

    model = ScreamcastModel.from_checkpoint(args.checkpoint, bf16=True).cuda().eval()
    model.set_tile_size(args.tile_size)
    network = model.pipeline.network
    coords = model.input_coords()
    coords["batch"] = np.arange(args.batch_size)
    coords["face"] = np.array([0])
    coords["time"] = np.array([np.datetime64("2020-10-13T00:00:00")])
    source = data_catalog.scream()
    flat = source._compute_flat_indices(0, coords["x"], coords["y"]).ravel()
    tidx = source._datetime_to_tidx(coords["time"][0])
    grouped = defaultdict(list)
    for channel, name in enumerate(coords["variable"]):
        base, level = source._parse_variable(name)
        grouped[base].append((channel, level))
    initial = np.empty(
        (1, len(coords["variable"]), 1, args.tile_size, args.tile_size),
        dtype=np.float32,
    )
    for base, channels in grouped.items():
        print("Fetching IC:", base, flush=True)
        array = source._var_group[base][base]
        values = (
            array.oindex[tidx, flat][None]
            if channels[0][1] is None
            else array.oindex[tidx, [level for _, level in channels], flat]
        )
        initial[0, [channel for channel, _ in channels], 0] = values.reshape(
            -1, args.tile_size, args.tile_size
        )
    if not np.isfinite(initial).all():
        raise ValueError("Input contains nonfinite values")
    state = torch.from_numpy(initial).cuda().repeat(args.batch_size, 1, 1, 1, 1)

    captured = []
    capture = network.register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs)
    )
    with torch.inference_mode():
        reference, _ = model(state, coords)
    capture.remove()
    if not torch.isfinite(reference).all().item():
        raise ValueError("Reference forecast contains nonfinite values")
    network_args = captured[0]
    captured.clear()
    original_output = None
    with torch.inference_mode():
        original_output = network(*network_args)
    original_chunks = []
    for block in network.strata.pixel_blocks:
        conv = block.adaln_bilinear_dw_conv
        if not isinstance(conv, DepthwiseConv):
            raise TypeError("Expected native DepthwiseConv")
        original_chunks.append(conv.chunk_size)
        conv.chunk_size = args.depthwise_chunk_size
        conv._chunked_conv = _build_chunked_depthwise_conv(
            conv, args.depthwise_chunk_size
        )
    print(
        "Native depthwise chunks:",
        original_chunks,
        "->",
        args.depthwise_chunk_size,
        flush=True,
    )

    def measure(call):
        for _ in range(args.warmup):
            call()
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        device_ms, wall_ms = [], []
        for _ in range(args.repeats):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            wall_start = time.perf_counter()
            start.record()
            result = call()
            end.record()
            end.synchronize()
            device_ms.append(start.elapsed_time(end))
            wall_ms.append((time.perf_counter() - wall_start) * 1000)
            del result
        return {
            "cuda_ms_median": statistics.median(device_ms),
            "cuda_ms_min": min(device_ms),
            "cuda_ms_max": max(device_ms),
            "cuda_ms_samples": device_ms,
            "wall_ms_median": statistics.median(wall_ms),
            "resident_allocated_gb": allocated / 1e9,
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            "incremental_peak_gb": (torch.cuda.max_memory_allocated() - allocated)
            / 1e9,
        }

    def errors(actual, expected):
        diff = actual.float() - expected.float()
        rms = expected.float().square().mean().sqrt()
        channel_rms = expected.float().square().mean(dim=(0, 2, 3, 4)).sqrt()
        return {
            "relative_rms": (diff.square().mean().sqrt() / rms).item(),
            "max_channel_relative_rms": (
                diff.square().mean(dim=(0, 2, 3, 4)).sqrt()
                / channel_rms.clamp_min(1e-12)
            )
            .max()
            .item(),
            "max_absolute_difference": diff.abs().max().item(),
        }

    with torch.inference_mode():
        chunk_error = errors(network(*network_args), original_output)
        print("Chunk change error:", chunk_error, flush=True)
        if chunk_error["max_channel_relative_rms"] > 1e-5:
            raise RuntimeError("Native chunk change exceeded numerical tolerance")
        if args.stages:
            run_stage_benchmark(network, network_args, args, measure, errors)
            return
        eager_timing = measure(lambda: network(*network_args))
        print("Eager network:", eager_timing, flush=True)
        eager_forecast_timing = measure(lambda: model(state, coords))
        eager_output = network(*network_args)
        eager_forecast, _ = model(state, coords)
        model.compile()
        network = model.pipeline.network
        torch.cuda.synchronize()
        first_start = time.perf_counter()
        compiled_output = network(*network_args)
        torch.cuda.synchronize()
        first_call_seconds = time.perf_counter() - first_start
        if not torch.isfinite(compiled_output).all().item():
            raise ValueError("Compiled network output contains nonfinite values")
        output_error = errors(compiled_output, eager_output)
        compiled_forecast, _ = model(state, coords)
        if not torch.isfinite(compiled_forecast).all().item():
            raise ValueError("Compiled forecast contains nonfinite values")
        forecast_error = errors(compiled_forecast, eager_forecast)
        print("Compilation plus first forward seconds:", first_call_seconds, flush=True)
        network_timing = measure(lambda: network(*network_args))
        forecast_timing = measure(lambda: model(state, coords))
        compilation_counters = {name: dict(values) for name, values in counters.items()}
        print("Compiled network:", network_timing, flush=True)
        print("Compiled forecast:", forecast_timing, flush=True)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(args.profile_steps):
                network(*network_args)
                prof.step()
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(output_dir / "trace.json"))
    trace = json.loads((output_dir / "trace.json").read_text())
    activities = [
        e
        for e in trace["traceEvents"]
        if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
    ]
    kernel_us, kernel_counts = Counter(), Counter()
    for event in activities:
        kernel_us[event["name"]] += event["dur"]
        kernel_counts[event["name"]] += 1
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "checkpoint": args.checkpoint,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "tile_size": args.tile_size,
        "batch_size": args.batch_size,
        "mode": "eval, inference_mode, torch.compile default inductor, internal BF16 autocast",
        "depthwise_convolution": "native vmap DepthwiseConv",
        "original_chunk_sizes": original_chunks,
        "depthwise_chunk_size": args.depthwise_chunk_size,
        "chunk_change_network_error": chunk_error,
        "source_revision": args.source_revision,
        "input_shape": list(network_args[0].shape),
        "parameter_count": sum(p.numel() for p in network.parameters()),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "profile_steps": args.profile_steps,
        "first_compiled_call_seconds": first_call_seconds,
        "network_timing": network_timing,
        "forecast_timing": forecast_timing,
        "eager_network_timing": eager_timing,
        "eager_forecast_timing": eager_forecast_timing,
        "compiled_vs_eager_network_error": output_error,
        "compiled_vs_eager_forecast_error": forecast_error,
        "compilation_counters": compilation_counters,
        "profile_cuda_activity_ms": sum(e["dur"] for e in activities)
        / args.profile_steps
        / 1000,
        "cuda_activities_per_forward": len(activities) / args.profile_steps,
        "top_kernels": [
            {
                "name": name,
                "cuda_ms": duration / args.profile_steps / 1000,
                "count_per_forward": kernel_counts[name] / args.profile_steps,
            }
            for name, duration in kernel_us.most_common()
        ],
    }
    (output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Saved results:", output_dir, flush=True)
