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

import torch

from screamcast.dali_ext_src import ScreamV2
from screamcast.model_pipelines import MixedPredictionAsymmetric
from screamcast.normalization import RunningNorm2d

root = os.getenv("PROJECT_ROOT", "")

# No pre-registered (UNet-era) pretrained models remain; the live rollout/eval
# paths construct pipelines via MixedPredictionAsymmetric_init directly.
__all__: list[str] = []


def MixedPredictionAsymmetric_init(
    init_network,
    loss_cls,
    experiment_name: str,
    plevel: int = 4,
    level_start: int = 3,
    level_end: int = 128,
    variables_prognostic: tuple = (),
    variables_forcing: tuple = (),
    variables_diagnostic: tuple = (),
    variables_prognostic_state: tuple = (),
    enable_3d_adapter: bool = False,
    checkpoint_name: str = "best.pth",
    do_qv_softplus: bool = False,
    do_precip_relu: bool = False,
    variables_input_zeroed: tuple = (),
):
    """Initialize MixedPredictionAsymmetric pipeline for ScreamV2 with mixed state+difference prediction mode"""

    def func(pretrained=True):

        in_channels = ScreamV2.num_of_input_channels(
            variables_prognostic,
            variables_forcing,
            plevel,
            level_start,
            level_end,
        )
        out_channels = ScreamV2.num_of_output_channels(
            variables_prognostic,
            variables_diagnostic,
            plevel,
            level_start,
            level_end,
        )

        pipeline = MixedPredictionAsymmetric(
            network=init_network(),
            loss_fn=loss_cls(),
            input_norm=RunningNorm2d(in_channels, fit_batches=20),
            target_norm=RunningNorm2d(out_channels, fit_batches=20),
            plevel=plevel,
            level_start=level_start,
            level_end=level_end,
            variables_prognostic=variables_prognostic,
            variables_forcing=variables_forcing,
            variables_diagnostic=variables_diagnostic,
            variables_prognostic_state=variables_prognostic_state,
            enable_3d_adapter=enable_3d_adapter,
            do_qv_softplus=do_qv_softplus,
            do_precip_relu=do_precip_relu,
            variables_input_zeroed=variables_input_zeroed,
        )

        if pretrained:
            checkpoint = os.path.join(root, experiment_name, "output", checkpoint_name)
            checkpoint_data = torch.load(checkpoint, weights_only=True)
            pipeline.load_checkpoint(checkpoint_data)
        return pipeline

    return func
