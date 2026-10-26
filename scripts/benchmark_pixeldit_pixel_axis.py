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
"""Screen registered pixel attention axis packing; Cirrus-42 (AI agent)."""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--results", required=True)
    args = parser.parse_args()
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    # Configure compiler workers before importing the Torch benchmark.
    from screamcast.pixel_axis_attention import run

    run(args)


if __name__ == "__main__":
    main()
