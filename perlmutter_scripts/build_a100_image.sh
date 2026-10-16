#!/bin/bash
# Build the STRATA image for Perlmutter A100s (compute capability 8.0).
# Run from the repository root on a machine that can docker/podman build
# and push to registry.nersc.gov. Do not reuse screamcast:0.2.
set -euo pipefail
docker build --build-arg CUDA_ARCH=8.0 -f docker/Dockerfile -t strata:a100 .
echo "Push strata:a100 to registry.nersc.gov/m4935/screamcast/strata-v1:a100"
echo "See https://docs.nersc.gov/development/containers/registry/"
