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
# Engine. Everything below is layered on top of it. Anonymous pulls of the
# base image generally work; a free NGC account may be required depending on
# NGC policy: https://catalog.ngc.nvidia.com
FROM nvcr.io/nvidia/pytorch:26.09-py3

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

# Pin the NGC torch / CUDA / numpy stack. NGC 26.09 ships an EMPTY
# /etc/pip/constraint.txt, so without this any package that declares e.g.
# torch>=2.13 would replace NGC's 2.14.0a0 pre-release torch with a PyPI
# wheel. NGC's nvidia-dali-cuda130 (the data pipelines) requires
# packaging<=26.2.
RUN python3 -m pip list --format=freeze \
    | grep -i -E '^(torch|torchvision|torchaudio|triton|numpy|transformer.engine|apex|flash.attn|cuda-|nvidia-|cupti)' \
    > /etc/pip/strata-constraint.txt \
 && echo "packaging<=26.2" >> /etc/pip/strata-constraint.txt \
 && cat /etc/pip/strata-constraint.txt
ENV PIP_CONSTRAINT=/etc/pip/strata-constraint.txt

# Plain-pip requirements. NATTEN, earth2grid, and physicsnemo are installed
# separately below because each needs flags a requirements file cannot
# express. PIP_CONSTRAINT stays active for every step so pip cannot displace
# the container's torch stack.
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir --prefer-binary \
    -r /tmp/requirements.txt

# NATTEN built from source: NGC PyTorch is a custom build, so the prebuilt
# wheels on https://whl.natten.org do not include a matching libnatten.
# Pass --build-arg CUDA_ARCH=<compute capabilities> for your GPUs, e.g.
# "8.0" for A100 (`nvidia-smi --query-gpu=compute_cap --format=csv,noheader`
# prints it; see build_natten.sh). The default covers Ada (RTX 6000 Ada,
# L40S) and Hopper (H100). On a GPU the build does not cover, the first NATTEN
# call aborts the whole process with "no kernel image is available for
# execution on the device" (no Python exception, so pytest exits silently).
ARG CUDA_ARCH="8.9;9.0"
COPY docker/build_natten.sh /tmp/build_natten.sh
RUN CUDA_ARCH=${CUDA_ARCH} bash /tmp/build_natten.sh

# earth2grid is not on PyPI; install the pinned GitHub archive (v2025.11.1).
# --no-deps and --no-build-isolation keep it from touching the torch stack.
RUN python3 -m pip install --no-cache-dir --prefer-binary --no-deps --no-build-isolation \
    https://github.com/NVlabs/earth2grid/archive/6e39b069f0558cd0eff0ec666ede5e125d79b4ff.tar.gz

# The datasets are zarr v3. (NGC 26.01 pinned zarr<3 in its constraint file;
# the strata-constraint.txt above does not constrain zarr.)
RUN python3 -m pip install --no-cache-dir --upgrade "zarr>=3"

# fme (AI2 Full Model Emulator) and earth2studio power the rollout/inference
# wrappers. Their runtime deps are installed first so the --no-deps install
# below cannot skip them; --no-deps stops pip from re-resolving the
# torch/CUDA/natten stack already baked into the image.
# torch-harmonics is pinned to 0.8.0 by fme==2026.1.1. earth2studio 0.13.0
# imports nest_asyncio and requires pandas<3 (NGC 26.09 ships pandas 3).
RUN python3 -m pip install --no-cache-dir --prefer-binary \
    dacite \
    dask \
    gcsfs \
    h5netcdf \
    h5py \
    imageio \
    imageio_ffmpeg \
    loguru \
    moviepy \
    nest-asyncio \
    omegaconf \
    "pandas<3" \
    plotly \
    pygrib \
    "setuptools<81" \
    tensorly \
    tensorly-torch \
    torch-harmonics==0.8.0
RUN python3 -m pip install --no-cache-dir --prefer-binary --no-deps \
    fme==2026.1.1 \
    earth2studio==0.13.0

# marimo runs the notebooks/ examples. It needs a newer Pygments than the
# apt-installed one, which has no pip RECORD file and cannot be uninstalled;
# shadow it in site-packages instead.
RUN python3 -m pip install --no-cache-dir --ignore-installed pygments
RUN python3 -m pip install --no-cache-dir --prefer-binary marimo

# physicsnemo pinned to main @ fae6bb2b (2.3.0a0). It contains PR #2140
# (merge b26a8805): Strata's depthwise conditioning conv defaults to
# chunk_size="auto" instead of fixed 2-channel vmap chunks, which fixes the
# slow pixel-stage conv of NVlabs/STRATA#6 for every config that leaves
# use_chunked_depthwise_conv at its default. No physicsnemo release contains
# it yet, and every commit that does requires torch>=2.13 (hence NGC 26.09).
# --no-deps is REQUIRED: physicsnemo declares torch>=2.13.0, and the NGC
# torch is a PEP 440 PRE-release that pip considers older, so a plain
# install would replace the container's torch.
# The import of physicsnemo.experimental.models.strata transitively pulls in
# all of physicsnemo.nn, which needs warp, tensordict and timm at import
# time (and, since the 2.3 dev line, treelib and termcolor via
# physicsnemo.distributed; GitPython and hydra-core are needed by the
# training entry points); tensordict/timm declare their own torch/torchvision floors, so they
# get the same --no-deps treatment, with their pure-Python runtime deps
# installed explicitly first.
RUN python3 -m pip install --no-cache-dir --prefer-binary \
    "warp-lang>=1.17" \
    jaxtyping \
    cloudpickle \
    importlib_metadata \
    packaging \
    pyvers \
    huggingface_hub \
    safetensors \
    pyyaml \
    "GitPython>=3.1.62" \
    "hydra-core>=1.3.7" \
    "termcolor>=3.3.0" \
    "treelib>=1.8.0"
# NB: tensordict declares orjson, but only as an optional accelerated JSON
# backend imported lazily (stdlib json is the registered default) — verified
# import-clean without it, so it is deliberately NOT installed (MPL-2.0).
RUN python3 -m pip install --no-cache-dir --prefer-binary --no-deps \
    "tensordict>=0.15.3,<0.16.0" \
    "timm>=1.0.30"
RUN python3 -m pip uninstall -y nvidia-physicsnemo || true
RUN python3 -m pip install --no-cache-dir --no-deps \
    https://github.com/NVIDIA/physicsnemo/archive/fae6bb2b0e96488a052d9627d41e6dc94653f06a.tar.gz

# The NGC image chain has shipped triton with an empty backends/nvidia/bin
# (no ptxas / cuobjdump / nvdisasm -> "RuntimeError: Cannot find ptxas") and
# without backends/nvidia/include (no cuda.h -> the CudaUtils gcc build
# behind every torch.compile dies with "fatal error: cuda.h"). Link the CUDA
# toolkit's binaries and header into the locations triton expects; both
# no-ops if a future image ships them.
RUN TRITON_NV=$(python3 -c "import triton, pathlib; print(pathlib.Path(triton.__file__).parent / 'backends' / 'nvidia')") \
 && mkdir -p "$TRITON_NV/bin" "$TRITON_NV/include" \
 && ln -sf /usr/local/cuda/bin/ptxas /usr/local/cuda/bin/cuobjdump /usr/local/cuda/bin/nvdisasm "$TRITON_NV/bin/" \
 && ln -sf /usr/local/cuda/include/cuda.h "$TRITON_NV/include/cuda.h"

# Build-time smoke test: the exact imports the Strata stack does at startup.
RUN python3 - <<'EOF'
import natten
import typer  # noqa: F401
import warp
from earth2grid import healpix_bare  # noqa: F401

import inspect

import physicsnemo

assert natten.HAS_LIBNATTEN, "natten built without libnatten"
assert physicsnemo.__version__.startswith("2.3"), physicsnemo.__version__
from physicsnemo.experimental.models.strata import (  # noqa: F401
    Strata,
    StrataTransformer3D,
)

# physicsnemo PR #2140 (NVlabs/STRATA#6): automatic depthwise-conv chunking.
default = inspect.signature(Strata.__init__).parameters["chunk_size_grouped_conv"].default
assert default == "auto", default
from physicsnemo.experimental.nn import (  # noqa: F401
    build_axial_rope_cos_sin_2d_continuous,
    spherical_centroid,
    stereographic_projection,
)

print(f"strata env OK: physicsnemo {physicsnemo.__version__}, warp {warp.__version__}")
EOF

WORKDIR /workspace
