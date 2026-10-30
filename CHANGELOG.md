# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

- Fix the slow pixel-stage depthwise conv (#6): pin physicsnemo main @
  `fae6bb2b`, which contains NVIDIA/physicsnemo#2140. Its Strata
  `DepthwiseConv` defaults to `chunk_size="auto"` instead of fixed
  2-channel chunks, so every config that leaves
  `use_chunked_depthwise_conv` at its default (including checkpoints that
  predate the field) gets the fast path with no config or checkpoint change.
  physicsnemo now requires torch >= 2.13, so the Docker base moves from NGC
  PyTorch 26.01 to 26.09; NATTEN goes to 0.21.7 (built as C++20 for the
  torch 2.14 headers) and earth2grid to v2025.11.1, which fixes NaNs in
  lat/lon regridding west of the first source longitude.
- The model architecture now comes from NVIDIA PhysicsNeMo
  (`physicsnemo.experimental.models.strata`); the local `dit_3d.py` /
  `dit_3d_pixel.py` implementations were replaced by thin wrappers
  (`screamcast/strata_wrappers.py`) that keep geometry, wind rotation, and
  legacy-checkpoint compatibility (`screamcast/checkpoint_compat.py`).
  Pre-migration checkpoints load transparently. The `nvidia-modulus`
  dependency was replaced by `nvidia-physicsnemo` (pinned source archive
  until the 2.2 release).
- Initial public Apache-2.0 release of Strata.
