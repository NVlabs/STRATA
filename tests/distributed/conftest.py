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


@pytest.fixture(scope="session")
def distributed_cuda_context() -> tuple[int, int, torch.device]:
    if not torch.cuda.is_available():
        pytest.skip("distributed tests require CUDA")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 8:
        pytest.skip("run distributed tests with torchrun --nproc_per_node 8")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://", device_id=device)

    rank = dist.get_rank()
    yield rank, world_size, device

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
