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
"""Compare native generated references and ordered Gluon norm; Cirrus-42 (AI agent)."""

import argparse
import importlib.util
import json
import re
import sys
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import torch
import triton

from screamcast.cuda_timing import measure
from screamcast.normalization_order import native_order_kernel


def launch_reference(kernel, arguments, config):
    kernel[(196608 // config["XBLOCK"],)](
        *arguments,
        XBLOCK=config["XBLOCK"],
        R0_BLOCK=config["R0_BLOCK"],
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
        enable_fp_fusion=True,
    )


def launch_candidate(a, b, output, semantic, residual):
    xblock = 64 if semantic else 8
    native_order_kernel[(triton.cdiv(196608, xblock),)](
        a,
        b,
        output,
        N=196608,
        SEMANTIC=semantic,
        RESIDUAL=residual,
        COALESCED=residual,
        num_warps=16,
        num_stages=1,
        enable_fp_fusion=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--results", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    torch.manual_seed(20261020)
    result = dict(
        agent="Cirrus-42 (AI agent)",
        time=datetime.now(timezone.utc).isoformat(),
        commit=args.commit,
        experiment="normalization-native-order-component",
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        manifest=args.manifest,
        warmup=5,
        repeats=50,
        cases=[],
        scope="Synthetic FP32 production-shape inputs; exact generated native references at selected configurations; candidate contiguous input and FP32/BF16 outputs; norm kernels only, output dtype differences explicit",
    )
    Path(args.results).write_text(json.dumps(result, indent=2) + "\n")
    with torch.inference_mode():
        a = torch.randn((8, 24576, 1024), device="cuda", dtype=torch.float32)
        strided = a.transpose(1, 2).contiguous().transpose(1, 2)
        b = torch.randn(a.shape, device="cuda", dtype=torch.bfloat16) * 0.1
        for index, row in enumerate(manifest["rows"]):
            if row["persistent"]:
                continue
            path = Path(row["source"])
            if path.read_text() != row["source_code"]:
                raise ValueError("Native generated source changed")
            name = f"cirrus42_norm_reference_{index}"
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            kernel = getattr(module, row["name"]).fn
            config = row["selected_config"]
            semantic = config["R0_BLOCK"] == 64
            residual = "in_ptr1" in kernel.arg_names
            dtype_name = re.search(
                r"'out_ptr2': '\*(fp32|bf16)'", row["source_code"]
            ).group(1)
            dtype = torch.float32 if dtype_name == "fp32" else torch.bfloat16
            expected = torch.empty(a.shape, device="cuda", dtype=dtype)
            extra = torch.empty_like(expected)
            values = dict(
                in_ptr0=strided,
                in_ptr1=b,
                out_ptr2=expected,
                out_ptr3=extra,
                xnumel=196608,
                r0_numel=1024,
            )
            arguments = [
                values[key]
                for key in kernel.arg_names
                if key not in ("XBLOCK", "R0_BLOCK")
            ]
            reference = partial(launch_reference, kernel, arguments, config)
            reference()
            fp32 = torch.empty_like(a)
            bf16 = torch.empty(a.shape, device="cuda", dtype=torch.bfloat16)
            candidate_fp32 = partial(launch_candidate, a, b, fp32, semantic, residual)
            candidate_bf16 = partial(launch_candidate, a, b, bf16, semantic, residual)
            candidate_fp32()
            candidate_bf16()
            fallback_bits = None
            if residual:
                summed = a + b.float()
                fallback = torch.empty(a.shape, device="cuda", dtype=torch.bfloat16)
                native_order_kernel[(triton.cdiv(196608, 8),)](
                    summed,
                    b,
                    fallback,
                    N=196608,
                    SEMANTIC=False,
                    RESIDUAL=False,
                    COALESCED=True,
                    num_warps=16,
                    num_stages=1,
                    enable_fp_fusion=True,
                )
                fallback_bits = int(
                    (
                        fallback.view(torch.int16)
                        != expected.to(torch.bfloat16).view(torch.int16)
                    ).sum()
                )
                del summed, fallback
            target = expected.to(torch.bfloat16)
            item = dict(
                native_name=row["name"],
                native_config=config,
                residual=residual,
                semantic=semantic,
                native_output_dtype=dtype_name,
                presummed_norm2_bf16_different_bits=fallback_bits,
                fp32_raw_different_bits=int(
                    (fp32.view(torch.int32) != expected.view(torch.int32)).sum()
                )
                if dtype == torch.float32
                else None,
                fp32_rounded_bf16_different_bits=int(
                    (
                        fp32.to(torch.bfloat16).view(torch.int16)
                        != target.view(torch.int16)
                    ).sum()
                ),
                bf16_different_bits=int(
                    (bf16.view(torch.int16) != target.view(torch.int16)).sum()
                ),
            )
            item["baseline_before"] = measure(reference, 5, 50)
            item["fp32"] = measure(candidate_fp32, 5, 50)
            item["bf16"] = measure(candidate_bf16, 5, 50)
            item["baseline_after"] = measure(reference, 5, 50)
            result["cases"].append(item)
            Path(args.results).write_text(json.dumps(result, indent=2) + "\n")
            print(
                "NORM",
                row["name"],
                item["fp32_raw_different_bits"],
                item["fp32_rounded_bf16_different_bits"],
                item["bf16_different_bits"],
                {
                    k: item[k]["median_ms"]
                    for k in ("baseline_before", "fp32", "bf16", "baseline_after")
                },
                flush=True,
            )
            del expected, extra, fp32, bf16, target


if __name__ == "__main__":
    main()
