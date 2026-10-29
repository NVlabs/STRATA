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
export NATTEN_VERSION="${NATTEN_VERSION:-0.21.7}"
export CUDA_ARCH="${CUDA_ARCH:-9.0}"  # 9.0=Hopper(H100), 10.0=Blackwell(B200), 8.0=Ampere(A100)
export NATTEN_CUDA_ARCH="${CUDA_ARCH}"
export NATTEN_N_WORKERS="${NATTEN_N_WORKERS:-16}"
export NATTEN_WITH_CUDA="1"
export NATTEN_VERBOSE="1"

echo "Building NATTEN ${NATTEN_VERSION} for CUDA architectures: ${CUDA_ARCH}"

# Non-editable install from the PyPI sdist; the source tree is removed after.
src=$(mktemp -d)
cd "${src}"
python -m pip download --no-deps --no-binary natten --no-build-isolation \
    "natten==${NATTEN_VERSION}" -d .
tar xzf "natten-${NATTEN_VERSION}.tar.gz"
cd "natten-${NATTEN_VERSION}"
# torch >= 2.13 headers require C++20 ("#error C++20 or later compatible
# compiler is required to use ATen"), but NATTEN hard-codes C++17 in
# csrc/CMakeLists.txt (the CXX_STD default and the host -std flag) and its
# setup.py cannot pass -DCXX_STD, so patch both before building.
sed -i -e 's/set(CXX_STD "17"/set(CXX_STD "20"/' -e 's/-std=c++17/-std=c++20/' csrc/CMakeLists.txt
grep -q 'CXX_STD "20"' csrc/CMakeLists.txt
python -m pip install --no-deps --no-build-isolation -v . 2>&1
cd /
rm -rf "${src}"

echo "NATTEN build complete. Verifying libnatten..."
python -c "import natten; print('HAS_LIBNATTEN:', natten.HAS_LIBNATTEN); assert natten.HAS_LIBNATTEN"
