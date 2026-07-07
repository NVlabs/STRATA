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
"""Backend selection for the Strata architecture classes.

This module is the ONE deliberate divergence point between the public STRATA
repo and the internal screamcast repo. Everything else in
``strata_wrappers.py`` (and the rest of the shared code) imports the model
classes from here so those files can stay byte-identical across the repos.

Public repo (this file): the plain physicsnemo Strata classes; there is no
cross-attention pixel conditioning and no domain-parallel support.

Internal repo: exports domain-parallel-aware subclasses (identical behavior on
plain tensors) and the cross-attention pixel model as ``CROSSATTN_CLS``.
"""

try:
    from physicsnemo.experimental.models.strata import (  # noqa: F401
        Strata,
        StrataTransformer3D,
    )
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "screamcast requires physicsnemo with the Strata models "
        "(physicsnemo >= 2.2 / main commit 07cdcc8b or newer). "
        "Install it via the pinned source archive in requirements.txt, or "
        "update your container."
    ) from exc

BACKBONE_CLS = StrataTransformer3D
STRATA_CLS = Strata
# Cross-attention pixel conditioning is an internal-only feature.
CROSSATTN_CLS = None
