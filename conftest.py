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

# Skip (rather than hard-error at collection) the test modules whose imports
# need optional heavy dependencies: NVIDIA DALI and Transformer Engine ship
# with the NGC PyTorch container, fme/earth2studio power the optional
# ACE/rollout workflows. `make install` inside the NGC container (or the
# docker/Dockerfile image) provides all of them; a plain pip environment can
# still run the remaining suite.
import importlib.util


def _missing(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is None
    except ModuleNotFoundError:  # namespace parent itself absent
        return True


collect_ignore = []
if _missing("nvidia.dali"):
    collect_ignore += [
        "tests/test_dali_ext_src.py",
        "tests/test_dataloader.py",
        "tests/test_dataloader_cubesphere.py",
        "tests/test_dataloader_global.py",
        "tests/test_datasource.py",
        "tests/test_earth2studio.py",
        "tests/test_pipeline.py",
        "tests/test_screamv2_cubesphere.py",
    ]
if _missing("fme"):
    collect_ignore += [
        "tests/test_finetune_ace2scream_sfno_rev2.py",
        "tests/test_plev_pipeline.py",
        "tests/test_scream_to_ace.py",
        "tests/test_vertical_interpolate.py",
        # these import earth2studio_wrappers, which pulls in screamcast.ace
        "tests/test_datasource.py",
        "tests/test_earth2studio.py",
    ]
if _missing("earth2studio"):
    collect_ignore += [
        "tests/test_datasource.py",
        "tests/test_earth2studio.py",
    ]
collect_ignore = sorted(set(collect_ignore))
