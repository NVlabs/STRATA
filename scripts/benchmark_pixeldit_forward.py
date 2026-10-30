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
"""Measure native PixelDiT eager and compiled forward inference on a real tile."""

import argparse

from dotenv import load_dotenv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--profile-steps", type=int, default=3)
    parser.add_argument("--depthwise-chunk-size", type=int, default=512)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--stages", action="store_true")
    parser.add_argument("--save-decoder-inputs")
    parser.add_argument("--flop-estimate", help="Shape-specific FLOP JSON for --stages")
    args = parser.parse_args()
    if args.stages and args.flop_estimate is None:
        parser.error("--stages requires --flop-estimate")

    # The data catalog reads environment variables during import.
    load_dotenv(args.env_file)
    # Parse CLI and configure the environment before importing GPU dependencies.
    from screamcast.forward_benchmark import run

    run(args)


if __name__ == "__main__":
    main()
