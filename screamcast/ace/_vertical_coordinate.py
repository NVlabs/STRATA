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

AK_ACE2_8L = torch.tensor(
    [
        0.0,
        5119.89501953125,
        13881.3310546875,
        19343.51171875,
        20087.0859375,
        15596.6953125,
        8880.453125,
        3057.265625,
        0.0,
    ],
    dtype=torch.float64,
)
BK_ACE2_8L = torch.tensor(
    [
        0.0,
        0.0,
        0.005377814639359713,
        0.059728413820266724,
        0.2034912109375,
        0.43839120864868164,
        0.6806430220603943,
        0.8739292621612549,
        1.0,
    ],
    dtype=torch.float64,
)
