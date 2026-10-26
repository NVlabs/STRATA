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
"""Verify token identity and a real stride change; Cirrus-42 (AI agent)."""

import torch

from screamcast.token_layout import copy_tokens, restore_tokens


def test_copy_preserves_fp32_bits_and_changes_channel_stride():
    input = (
        torch.arange(1024, dtype=torch.float32).view(1, 1, 1024).expand(8, 24576, 1024)
    )
    output = copy_tokens(input)
    assert output.is_contiguous()
    assert output.stride() == (25165824, 1024, 1)
    assert torch.equal(output.view(torch.int32), input.view(torch.int32))
    assert output.data_ptr() != input.data_ptr()

    restored = restore_tokens(output)
    assert restored.stride() == (25165824, 1, 24576)
    assert torch.equal(restored.view(torch.int32), input.view(torch.int32))
    assert restored.data_ptr() != output.data_ptr()
