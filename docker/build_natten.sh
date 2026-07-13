#!/bin/bash
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
set -e

# Build NATTEN from source against the torch in the current environment.
# Required inside NGC PyTorch containers: their torch is a custom build, so
# the prebuilt wheels on https://whl.natten.org ship no matching libnatten.
export NATTEN_VERSION="${NATTEN_VERSION:-0.21.5}"
export CUDA_ARCH="${CUDA_ARCH:-9.0}"  # 9.0=Hopper(H100), 10.0=Blackwell(B200), 8.0=Ampere(A100)
export NATTEN_CUDA_ARCH="${CUDA_ARCH}"
export NATTEN_N_WORKERS="${NATTEN_N_WORKERS:-16}"
export NATTEN_WITH_CUDA="1"
export NATTEN_VERBOSE="1"

echo "Building NATTEN ${NATTEN_VERSION} for CUDA architectures: ${CUDA_ARCH}"

git clone --recursive https://github.com/SHI-Labs/NATTEN /opt/natten
cd /opt/natten
git checkout "v${NATTEN_VERSION}"
make clean uninstall || true
make
pip install --no-deps --no-build-isolation -v -e . 2>&1

echo "NATTEN build complete. Verifying libnatten..."
python -c "import natten; print('HAS_LIBNATTEN:', natten.HAS_LIBNATTEN)"
