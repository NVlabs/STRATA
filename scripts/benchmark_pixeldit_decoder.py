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
"""Compare a fixed batch-eight, tile-128 BF16 stage and full network."""

import argparse
import faulthandler
import os
import signal

from dotenv import load_dotenv


def main():
    # Inert during timing; request Python thread stacks with SIGUSR1 when diagnosing a wait.
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("decoder", "backbone"), default="decoder")
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env-file", default="/repo/.env")
    parser.add_argument("--results", required=True)
    parser.add_argument("--experiment", default="pixel-state-row-layout")
    parser.add_argument("--baseline-experiment", default="pixel-state-row-layout")
    parser.add_argument("--commit", required=True)
    parser.add_argument(
        "--baseline-commit", default="68b6d8750965f73569715c0ca238257eb59c25e5"
    )
    parser.add_argument("--trace-prefix")
    parser.add_argument("--profile-traffic", action="store_true")
    parser.add_argument(
        "--numerical-only",
        action="store_true",
        help="Run isolated/full numerical diagnostics without timing or acceptance",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    load_dotenv(args.env_file)
    # Configure before Torch imports; explicit caller/environment settings take precedence.
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    from screamcast.decoder_benchmark import run

    run(args)


if __name__ == "__main__":
    main()
