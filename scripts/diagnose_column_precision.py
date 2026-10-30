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
"""Diagnose column precision on real model operands; Cirrus-42 (AI agent)."""

import argparse

from dotenv import load_dotenv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "checkpoint", "commit", "results"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--env-file", default="/repo/.env")
    parser.add_argument("--operands-dir")
    args = parser.parse_args()
    load_dotenv(args.env_file)
    # Configure checkpoint paths before initializing Earth2Studio imports.
    from screamcast.column_precision_benchmark import run

    run(args)


if __name__ == "__main__":
    main()
