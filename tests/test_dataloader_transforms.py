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

from screamcast import dataloader_transforms


def test_batch():
    def gen():
        for i in range(10):
            yield torch.ones(10), torch.ones(10)

    for k, (a, b) in enumerate(dataloader_transforms.Batch(gen(), 5)):
        assert a.shape == (5, 10)
        assert b.shape == (5, 10)
    assert k + 1 == 2  # 2 iterations


def test_unbatch():
    def gen():
        for i in range(10):
            yield torch.ones(2, 10), torch.ones(2, 10)

    for k, (a, b) in enumerate(dataloader_transforms.Unbatch(gen(), 2)):
        assert a.shape == (10,)
        assert b.shape == (10,)
    assert k + 1 == 20


def test_cross_face_subtiler_shapes_and_len():
    # Build a single padded face: B=1, C=2, Hpad=Wpad=8
    B, C, Hpad = 1, 2, 8
    tile_size = 4
    overlap = 2
    nside = Hpad - 2 * overlap  # implied by padding formula in cross-face flow

    s0 = torch.arange(B * C * Hpad * Hpad, dtype=torch.float32).reshape(
        B, C, Hpad, Hpad
    )
    s1 = s0 + 1
    index = torch.zeros(B, Hpad, Hpad, dtype=torch.long)
    j = torch.tensor([5], dtype=torch.long)

    upstream = [(s0, s1, index, j)]

    subt = dataloader_transforms.CrossFaceSubTiler(
        upstream,
        nside=nside,
        tile_size=tile_size,
        shuffle_tiles=False,
        num_steps=1,
        skip_corner_tiles=False,
    )

    tiles = list(subt)
    # stride = tile_size - overlap = 2; Hpad=8 -> cx=cy=3 => 9 tiles
    assert len(tiles) == 9
    for t in tiles:
        assert len(t) == 4
        ip, tp, idxp, jp = t
        assert ip.shape == (B, C, tile_size, tile_size)
        assert tp.shape == (B, C, tile_size, tile_size)
        assert idxp.shape == (B, tile_size, tile_size)
        assert jp.shape == j.shape

    # __len__ reflects tiles per upstream sample
    assert len(subt) == len(upstream) * 9

    # test skip_corner_tiles=False
    subt = dataloader_transforms.CrossFaceSubTiler(
        upstream,
        nside=nside,
        tile_size=tile_size,
        shuffle_tiles=False,
        num_steps=1,
        skip_corner_tiles=True,
    )
    tiles = list(subt)
    assert len(tiles) == 5  # 3*3 - 4 (corners) = 5


def test_cross_face_subtiler_balance_cross_face():
    """balance_cross_face subsamples the larger group to match the smaller one."""
    # nside=8, tile_size=4 → overlap=2, stride=2, padded=12, cx=cy=5
    # With skip_corner_tiles=True: 21 tiles total
    #   cross-face (border, non-corner): 12 tiles
    #   interior (3x3 inner grid):        9 tiles
    # Balanced → subsample cross-face to 9, total = 9 + 9 = 18 per upstream sample
    nside = 8
    tile_size = 4
    Hpad = nside + tile_size  # 12
    B, C = 1, 2

    s0 = torch.randn(B, C, Hpad, Hpad)
    s1 = torch.randn(B, C, Hpad, Hpad)
    index = torch.zeros(B, Hpad, Hpad, dtype=torch.long)
    j = torch.tensor([0], dtype=torch.long)

    upstream = [(s0, s1, index, j)]

    # --- without balancing ---
    subt_unbal = dataloader_transforms.CrossFaceSubTiler(
        upstream,
        nside=nside,
        tile_size=tile_size,
        shuffle_tiles=False,
        num_steps=1,
        skip_corner_tiles=True,
        balance_cross_face=False,
    )
    assert len(subt_unbal.cross_face_tiles) == 12  # 5*5 - 3*3 - 4 (corners skipped)
    assert len(subt_unbal.interior_tiles) == 9
    assert len(subt_unbal) == 21  # all tiles

    # --- with balancing ---
    subt_bal = dataloader_transforms.CrossFaceSubTiler(
        upstream,
        nside=nside,
        tile_size=tile_size,
        shuffle_tiles=False,
        num_steps=1,
        skip_corner_tiles=True,
        balance_cross_face=True,
    )
    n_min = min(12, 9)  # 9
    assert len(subt_bal) == 2 * n_min  # 18

    tiles = list(subt_bal)
    assert len(tiles) == 2 * n_min  # actual yield matches __len__

    # Verify shapes are correct
    for t in tiles:
        ip, tp, idxp, jp = t
        assert ip.shape == (B, C, tile_size, tile_size)
        assert tp.shape == (B, C, tile_size, tile_size)
