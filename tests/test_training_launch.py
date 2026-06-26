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
import subprocess
import sys
import tempfile

import pytest

skip_s3 = (
    os.getenv("GITLAB_CI") == "true"
    or os.getenv("CI") == "true"
    or os.getenv("SCREAM_SKIP_S3_TESTS") in {"1", "true", "True"}
)
pytestmark = pytest.mark.skipif(
    skip_s3, reason="S3/rclone not available in CI environment"
)


def test_training_launch():
    """Test that the 'pytest' config (mock=True, no_save, 2 steps) runs without errors."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = os.getcwd()
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["PROJECT_ROOT"] = temp_dir
        subprocess.check_call(
            [  # noqa: S603
                sys.executable,
                os.path.join(original_cwd, "train.py"),
                "pytest",
            ],
            timeout=180,
            env=env,
        )
