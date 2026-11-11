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
"""Quick fixed-input CUDA preparation screening before full-network trials."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument(
        "--baseline-commit", default="0b2ea296e4272a9a69a9c65deb498658a144af19"
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    # Keep expensive GPU/runtime imports at the CLI boundary, after parsing.
    import torch

    from screamcast import conditioning_fused_prepare, conditioning_spatial_reuse
    from screamcast.cuda_timing import measure

    cached = torch.load(args.inputs, weights_only=True, map_location="cpu")
    if (cached["batch_size"], cached["tile_size"]) != (8, 128):
        raise ValueError("Requires fixed batch8/tile128 fixture")
    conditioning = cached["conditioning"].cuda()
    del cached
    conditioning_fused_prepare.initialize()
    conditioning_spatial_reuse.initialize()
    with torch.inference_mode():
        conditioning_spatial_reuse.component_profile(conditioning, args, measure)


if __name__ == "__main__":
    main()
