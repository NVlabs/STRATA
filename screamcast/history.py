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
"""Helpers for embedding command provenance in generated artifacts."""

from __future__ import annotations

import datetime
import subprocess
import sys
from collections.abc import Sequence


def git_commit() -> str:
    try:
        return (
            subprocess.check_output(  # noqa: S603
                ["git", "rev-parse", "HEAD"],  # noqa: S607, S603
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def history_entry(
    argv: Sequence[str] | None = None, previous: str | None = None
) -> str:
    command = list(sys.argv if argv is None else argv)
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    entry = f"{timestamp}: {' '.join(command)}"
    if previous:
        return f"{previous}\n{entry}"
    return entry
