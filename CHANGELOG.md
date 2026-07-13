# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

- The model architecture now comes from NVIDIA PhysicsNeMo
  (`physicsnemo.experimental.models.strata`); the local `dit_3d.py` /
  `dit_3d_pixel.py` implementations were replaced by thin wrappers
  (`screamcast/strata_wrappers.py`) that keep geometry, wind rotation, and
  legacy-checkpoint compatibility (`screamcast/checkpoint_compat.py`).
  Pre-migration checkpoints load transparently. The `nvidia-modulus`
  dependency was replaced by `nvidia-physicsnemo` (pinned source archive
  until the 2.2 release).
- Initial public Apache-2.0 release of Strata.
