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
"""Collect real operands only after fake tracing; Cirrus-42 (AI agent)."""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from screamcast import column_precision


@pytest.mark.parametrize(
    "operation",
    (column_precision.diagnostic, column_precision.selected_attention),
    ids=("diagnostic", "native_sum"),
)
def test_fake_column_diagnostic_does_not_collect_observations(
    tmp_path, monkeypatch, operation
):
    column_precision.records.clear()
    monkeypatch.setattr(column_precision, "operand_directory", tmp_path)
    with FakeTensorMode():
        q = torch.empty((8192, 16, 24, 64), device="cuda", dtype=torch.bfloat16)
        output = operation(q, q, q)
        assert output.shape == q.shape
        assert output.stride() == q.stride()
        assert output.dtype == q.dtype
    assert column_precision.records == []
    assert list(tmp_path.iterdir()) == []
