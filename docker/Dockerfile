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

# Reproducible training/inference environment for Strata.
#
# Build from the repository root:
#
#     docker build -f docker/Dockerfile -t strata .
#
# The NGC PyTorch base image provides the pieces that cannot come from PyPI:
# a CUDA-tuned torch build, NVIDIA DALI (data pipelines), and Transformer
# Engine. Everything below is layered on top of it. Pulling the base image
# requires a (free) NGC account: https://catalog.ngc.nvidia.com
FROM nvcr.io/nvidia/pytorch:26.01-py3

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System libraries: GEOS/PROJ for cartopy, HDF5/netCDF for the data readers,
# eccodes for pygrib (earth2studio's GRIB reader), ffmpeg for rollout videos.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    pkg-config \
    libgeos-dev \
    libproj-dev \
    proj-bin \
    proj-data \
    libhdf5-dev \
    libnetcdf-dev \
    libcurl4-openssl-dev \
    libeccodes-dev \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install -U pip setuptools wheel

# Plain-pip requirements. NATTEN, earth2grid, and physicsnemo are installed
# separately below because each needs flags a requirements file cannot
# express. This step runs with NGC's PIP_CONSTRAINT still active so pip
# cannot displace the container's torch stack.
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir --prefer-binary \
    -r /tmp/requirements.txt

# NATTEN built from source: NGC PyTorch is a custom build, so the prebuilt
# wheels on https://whl.natten.org do not include a matching libnatten.
# Pass --build-arg CUDA_ARCH=<compute capability> for your GPU
# (9.0=Hopper/H100 default; `nvidia-smi --query-gpu=compute_cap
# --format=csv,noheader` prints it; see build_natten.sh).
ARG CUDA_ARCH=9.0
COPY docker/build_natten.sh /tmp/build_natten.sh
RUN CUDA_ARCH=${CUDA_ARCH} bash /tmp/build_natten.sh

# earth2grid is not on PyPI; install the pinned GitHub archive. --no-deps
# and --no-build-isolation keep it from touching the torch stack.
RUN python3 -m pip install --no-cache-dir --prefer-binary --no-deps --no-build-isolation \
    https://github.com/NVlabs/earth2grid/archive/3415382a37e414a867d4c6ae2519e3e8afdbece1.tar.gz

# NGC pins zarr<3 via PIP_CONSTRAINT; the datasets are zarr v3. Clear the
# constraint for this and the remaining steps.
ENV PIP_CONSTRAINT=""
RUN python3 -m pip install --no-cache-dir --upgrade "zarr>=3"

# fme (AI2 Full Model Emulator) and earth2studio power the rollout/inference
# wrappers. Their runtime deps are installed first so the --no-deps install
# below cannot skip them; --no-deps stops pip from re-resolving the
# torch/CUDA/natten stack already baked into the image.
# torch-harmonics is pinned to 0.8.0 by fme==2026.1.1.
RUN python3 -m pip install --no-cache-dir --prefer-binary \
    dacite \
    dask \
    gcsfs \
    h5netcdf \
    imageio \
    imageio_ffmpeg \
    loguru \
    moviepy \
    plotly \
    pygrib \
    "setuptools<81" \
    tensorly \
    tensorly-torch \
    torch-harmonics==0.8.0
RUN python3 -m pip install --no-cache-dir --prefer-binary --no-deps \
    fme==2026.1.1 \
    earth2studio==0.13.0

# marimo runs the notebooks/ examples.
RUN python3 -m pip install --no-cache-dir --prefer-binary marimo

# physicsnemo pinned to main @ 07cdcc8b (includes the Strata models and the
# continuous/stereographic RoPE helpers from PR #1750; pre-2.2, so PyPI does
# not have it yet). --no-deps is REQUIRED: physicsnemo declares
# torch>=2.10.0, and the NGC torch is a PEP 440 PRE-release that pip
# considers older, so a plain install would replace the container's torch.
# warp-lang >= 1.14 and jaxtyping are the runtime deps that --no-deps skips.
RUN python3 -m pip install --no-cache-dir --prefer-binary \
    "warp-lang>=1.14" \
    jaxtyping
RUN python3 -m pip uninstall -y nvidia-physicsnemo || true
RUN python3 -m pip install --no-cache-dir --no-deps \
    https://github.com/NVIDIA/physicsnemo/archive/07cdcc8b65f10bee733a32a77b4b76e20c70c54c.tar.gz

# The NGC image ships triton with an empty backends/nvidia/bin (no ptxas /
# cuobjdump / nvdisasm), which breaks every torch.compile with
# "RuntimeError: Cannot find ptxas". Link the CUDA toolkit's binaries into
# the location triton expects.
RUN TRITON_BIN=$(python3 -c "import triton, pathlib; print(pathlib.Path(triton.__file__).parent / 'backends' / 'nvidia' / 'bin')") \
 && mkdir -p "$TRITON_BIN" \
 && ln -sf /usr/local/cuda/bin/ptxas /usr/local/cuda/bin/cuobjdump /usr/local/cuda/bin/nvdisasm "$TRITON_BIN"/

# Build-time smoke test: the exact imports the Strata stack does at startup.
RUN python3 - <<'EOF'
import natten
import typer  # noqa: F401
import warp
from earth2grid import healpix_bare  # noqa: F401

import physicsnemo

assert natten.HAS_LIBNATTEN, "natten built without libnatten"
assert physicsnemo.__version__.startswith("2.2"), physicsnemo.__version__
from physicsnemo.experimental.models.strata import (  # noqa: F401
    Strata,
    StrataTransformer3D,
)
from physicsnemo.experimental.nn import (  # noqa: F401
    build_axial_rope_cos_sin_2d_continuous,
    spherical_centroid,
    stereographic_projection,
)

print(f"strata env OK: physicsnemo {physicsnemo.__version__}, warp {warp.__version__}")
EOF

WORKDIR /workspace
