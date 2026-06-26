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
import torch

from screamcast.normalization import RunningNorm2d


# Test the RunningNorm2d class
def test_running_norm2d():
    # Initialize parameters
    channels = 3
    height = 4
    width = 4
    fit_batches = 5
    eps = 1e-9

    # Create an instance of RunningNorm2d
    running_norm = RunningNorm2d(channels, fit_batches, eps)

    # Set the model to training mode
    running_norm.train()

    # Generate some random batches of input data
    batch_size = 2
    num_batches = 5
    inputs = [
        torch.randn(batch_size, channels, height, width) for _ in range(num_batches)
    ]

    expected_mean = torch.stack(inputs).mean(dim=(-1, -2), keepdim=True).mean(0).mean(0)
    expected_var = (
        torch.stack(inputs)
        .var(dim=(0, 1, 3, 4), keepdim=True, unbiased=False)
        .squeeze(dim=(0, 1))
    )

    # Initialize variables to track mean and variance manually

    for batch in inputs:
        running_norm(batch)

        # Assert that the running statistics are close to the manual calculations
    assert torch.allclose(
        running_norm.mean.flatten(), expected_mean.flatten(), atol=1e-5
    )
    assert torch.allclose(running_norm.var.flatten(), expected_var.flatten(), atol=1e-5)
