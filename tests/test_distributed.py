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
import os

import pytest
import torch
import torch.distributed as dist

from screamcast.inference import (
    broadcast_scalar,
    gather_patches,
    scatter_patches,
)


def set_env(rank, world_size, port=13254):
    os.environ["RANK"] = f"{rank}"
    os.environ["LOCAL_RANK"] = f"{rank % torch.cuda.device_count()}"
    os.environ["WORLD_SIZE"] = f"{world_size}"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = f"{port}"


def run_scatter_gather(rank_, world_size_, verbose=False):
    set_env(rank_, world_size_, port=13254)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        batch_input = torch.zeros([12, 121, 64, 64], dtype=torch.float32, device=device)
        batch_input[0:6, :, :, :] = 1.0
        batch_input[6:12, :, :, :] = 2.0
    else:
        batch_input = None

    batch_local = scatter_patches(batch_input)
    if verbose:
        print(f"Rank {rank}: {batch_local.shape}")

    gathered_input = gather_patches(batch_local)
    if rank == 0:
        if verbose:
            print(f"Rank {rank}: {gathered_input.shape}")
            print(gathered_input[:, 0, 0, 0])
        assert torch.allclose(gathered_input, batch_input)

    dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Not enough GPUs available for distributed test",
)
def test_scatter_gather():
    num_gpus = torch.cuda.device_count()
    assert num_gpus >= 2, "Not enough GPUs available for test"
    world_size = 2
    verbose = False  # Change to True for debug

    torch.multiprocessing.set_start_method("spawn", force=True)

    torch.multiprocessing.spawn(
        run_scatter_gather,
        args=(
            world_size,
            verbose,
        ),
        nprocs=world_size,
        join=True,
        daemon=True,
    )


def run_broadcast_scalar(rank_, world_size_, verbose=False):
    set_env(rank_, world_size_, port=13255)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    value = 1 if rank == 0 else 0

    bcast_value = broadcast_scalar(value)
    if verbose:
        print(f"Rank {rank}: Initial value = {value}")
        print(f"Rank {rank}: Final value = {bcast_value}")
    assert bcast_value == 1

    dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Not enough GPUs available for distributed test",
)
def test_broadcast_scalar():
    num_gpus = torch.cuda.device_count()
    assert num_gpus >= 2, "Not enough GPUs available for test"
    world_size = num_gpus
    verbose = False  # Change to True for debug

    torch.multiprocessing.set_start_method("spawn", force=True)

    torch.multiprocessing.spawn(
        run_broadcast_scalar,
        args=(
            world_size,
            verbose,
        ),
        nprocs=world_size,
        join=True,
        daemon=True,
    )
