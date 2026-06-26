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
import math

from torch.optim.lr_scheduler import LambdaLR


def get_dampened_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 1,
    decay=0.9,
    last_epoch: int = -1,
    min_ratio=0.01,
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        n_steps = float(max(1, num_training_steps - num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / n_steps

        if progress >= 1.0:
            return 0.0

        cycle = (current_step - num_warmup_steps) // (n_steps / num_cycles)
        if cycle:
            ratio = decay**cycle
        else:
            ratio = 1

        return max(
            min_ratio,
            0.5
            * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1)))
            * ratio,
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_warmup_constant_schedule(optimizer, num_warmup_steps, last_epoch=-1):
    """Linear warmup for num_warmup_steps, then constant LR."""

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch)
