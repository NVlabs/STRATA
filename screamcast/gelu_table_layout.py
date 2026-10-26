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
"""Native GELU table address layouts; Cirrus-42 (AI agent)."""

from datetime import datetime, timezone
from functools import partial
from types import MethodType

import torch
import triton
import triton.language as tl

from screamcast import gelu_table
from screamcast.benchmark_statistics import latency_gate
from screamcast.cuda_timing import measure
from screamcast.gemm_schedule_validation import save

MODES = (16, 64, 256, -1)


@triton.jit
def kernel(INPUT, TABLE, OUT, N: tl.constexpr, MODE: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(INPUT + index, index < N, other=0)
    bits = value.to(tl.uint16, bitcast=True).to(tl.int32)
    if MODE == -1:
        address = ((bits << 1) & 65535) | (bits >> 15)
    else:
        address = bits ^ ((bits >> 15) * MODE)
    result = tl.load(TABLE + address)
    tl.store(OUT + index, result, index < N)


def prepare_table(full, mode):
    bits = torch.arange(65536, dtype=torch.int64, device=full.device)
    address = (
        (((bits << 1) & 65535) | (bits >> 15))
        if mode == -1
        else bits ^ ((bits >> 15) * mode)
    )
    table = torch.empty_like(full)
    table[address] = full
    return table


def evaluate(input, table, mode, block):
    output = torch.empty_like(input)
    kernel[(triton.cdiv(input.numel(), block),)](
        input,
        table,
        output,
        N=input.numel(),
        MODE=mode,
        BLOCK=block,
        num_warps=4,
    )
    return output


@torch.library.custom_op("cirrus42::gelu_address_layout", mutates_args=())
def lookup(
    input: torch.Tensor, table: torch.Tensor, mode: int, block: int
) -> torch.Tensor:
    if (
        input.shape not in ((8, 24576, 4096), (8, 393216, 512))
        or input.dtype != torch.bfloat16
        or not input.is_cuda
        or not input.is_contiguous()
        or table.shape != (65536,)
        or table.dtype != torch.bfloat16
        or table.device != input.device
        or not table.is_contiguous()
        or mode not in MODES
        or block not in (4096, 8192)
    ):
        raise ValueError(
            "Address layout requires released contiguous BF16 activations and a permuted native table"
        )
    return evaluate(input, table, mode, block)


@lookup.register_fake
def lookup_fake(input, table, mode, block):
    return torch.empty_like(input)


def run(args):
    result = dict(
        agent="Cirrus-42 (AI agent)",
        source_commit=args.commit,
        time=datetime.now(timezone.utc).isoformat(),
        experiment="native-gelu-table-address-layout",
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        modes=list(MODES),
        warmup=5,
        repeats=100,
        seeds=[20261046, 20261047],
        cases=[],
        scope="Both actual GELU activation shapes; synthetic BF16 values. Selected registered native-table kernel versus registered permuted-table kernel, same output allocation and dispatch path. Full65536pattern equality required per mode; no model acceptance from components.",
    )
    save(args.results, result)
    with torch.inference_mode():
        full = gelu_table.prepare_table(torch.device("cuda:0"))
        patterns = (
            torch.arange(65536, dtype=torch.int32, device="cuda")
            .to(torch.int16)
            .view(torch.bfloat16)
        )
        for mode in MODES:
            table = prepare_table(full, mode)
            domain_error = int(
                (
                    evaluate(patterns, table, mode, 4096).view(torch.int16)
                    != full.view(torch.int16)
                ).sum()
            )
            if domain_error:
                raise RuntimeError(
                    f"Address mode{mode} changes{domain_error} native GELU patterns"
                )
            for shape in ((8, 24576, 4096), (8, 393216, 512)):
                for block in (4096, 8192):
                    case = dict(
                        mode=mode,
                        shape=list(shape),
                        block=block,
                        all_pattern_different_bf16_bits=domain_error,
                        samples=[],
                    )
                    result["cases"].append(case)
                    for seed in result["seeds"]:
                        torch.manual_seed(seed)
                        x = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
                        baseline = partial(gelu_table.lookup, x, full)
                        candidate = partial(lookup, x, table, mode, block)
                        expected, actual = baseline(), candidate()
                        sample = dict(
                            seed=seed,
                            different_bf16_bits=int(
                                (
                                    actual.view(torch.int16)
                                    != expected.view(torch.int16)
                                ).sum()
                            ),
                        )
                        del expected, actual
                        sample["before"] = measure(baseline, 5, 100)
                        sample["candidate"] = measure(candidate, 5, 100)
                        sample["after"] = measure(baseline, 5, 100)
                        case["samples"].append(sample)
                        save(args.results, result)
                        print(
                            "GELU_ADDRESS_RESULT",
                            mode,
                            list(shape),
                            block,
                            seed,
                            sample["different_bf16_bits"],
                            {
                                k: sample[k]["median_ms"]
                                for k in ("before", "candidate", "after")
                            },
                            flush=True,
                        )
                        del baseline, candidate, x
                    gate = latency_gate(*case["samples"], isolated_scope="seed20261046")
                    gate["comparisons"] = {
                        key.replace("full_", "seed20261047_"): value
                        for key, value in gate["comparisons"].items()
                    }
                    case["gate"] = gate
                    case["component_eligible"] = gate["significant"] and all(
                        sample["different_bf16_bits"] == 0 for sample in case["samples"]
                    )
                    case["status"] = "complete"
                    save(args.results, result)
    result["status"] = "complete"
    save(args.results, result)


_ORIGINALS = []


def _forward(self, input):
    return lookup(input, self._cirrus42_gelu_table, 256, 4096)


def restore():
    for module, had_forward, original_forward, original_table in _ORIGINALS:
        module._cirrus42_gelu_table = original_table
        if had_forward:
            module.forward = original_forward
        else:
            del module.forward
    _ORIGINALS.clear()


def install(strata):
    restore()
    modules = [
        module
        for module in strata.modules()
        if isinstance(module, torch.nn.GELU) and module.approximate == "tanh"
    ]
    owned = {id(module) for module, _, _ in gelu_table._ORIGINALS}
    if (
        strata.training
        or len(modules) != 28
        or {id(module) for module in modules} != owned
        or any(
            module.training or "_cirrus42_gelu_table" not in module._buffers
            for module in modules
        )
    ):
        raise ValueError(
            "Address layout requires28 owned eval tanh GELUs from the confirmed live-backbone baseline"
        )
    full = modules[0]._cirrus42_gelu_table
    if any(
        module._cirrus42_gelu_table.shape != (65536,)
        or module._cirrus42_gelu_table.dtype != torch.bfloat16
        or not module._cirrus42_gelu_table.is_cuda
        or not module._cirrus42_gelu_table.is_contiguous()
        or module._cirrus42_gelu_table.device != full.device
        for module in modules
    ):
        raise ValueError(
            "Address layout requires native contiguous BF16 tables on one CUDA device"
        )
    table = prepare_table(full, 256)
    for module in modules:
        _ORIGINALS.append(
            (
                module,
                "forward" in module.__dict__,
                module.forward,
                module._cirrus42_gelu_table,
            )
        )
        module._cirrus42_gelu_table = table
        module.forward = MethodType(_forward, module)
