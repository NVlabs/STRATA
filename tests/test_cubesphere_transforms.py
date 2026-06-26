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
import torch

from screamcast.cubesphere_transforms import (
    create_padded_faces_batched,
    halo_dst_flat_indices,
    halo_dst_flat_indices_from_gridfile,
    reorder_cubesphere_to_2d_tensor,
    unstructured_to_6faces,
)


def test_reorder_cubesphere_to_2d_tensor():
    array = torch.randn(1, 2048**2)
    assert reorder_cubesphere_to_2d_tensor(array, ne=1024, npg=2).shape == (
        1,
        2048,
        2048,
    )


def test_unstructured_to_6faces():
    array = torch.randn(1, 6 * 2048**2)
    faces = unstructured_to_6faces(array, ne=1024, npg=2)
    assert faces.shape == (1, 6, 2048, 2048)

    faces_padded = create_padded_faces_batched(faces, pad_width=64)
    assert faces_padded.shape == (1, 6, 2176, 2176)

    faces_padded = create_padded_faces_batched(faces, pad_width=0)
    assert faces_padded.shape == (1, 6, 2048, 2048)


def test_halo_dst_flat_indices():
    index = halo_dst_flat_indices(2048, 64)
    expected_shape = (
        (2048 * 64 + 64 * 64) * 4 * 6
    )  # halo includes 4 edges+corners per face, 6 faces
    assert index.shape == (expected_shape,)


def test_halo_dst_flat_indices_from_gridfile():
    index, meta = halo_dst_flat_indices_from_gridfile(2048, 2048 + 1024, 64)
    expected_shape = (
        (2048 * 64 + 64 * 64) * 4 * 6
    )  # halo includes 4 edges+corners per face, 6 faces
    assert index.shape == (expected_shape,)
