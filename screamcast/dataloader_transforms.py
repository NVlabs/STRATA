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
import random

import torch


class Unbatch:
    def __init__(self, iterator, batch_size: int | None = None):
        self.iterator = iterator
        self.batch_size = batch_size

    def __len__(self) -> int:
        return len(self.iterator) * self.batch_size

    def __iter__(self):
        for batch in self.iterator:
            lens = [item.shape[0] for item in batch]
            if len(set(lens)) != 1:
                raise RuntimeError("all items must have the same length")

            n = lens[0]

            for i in range(n):
                yield [item[i] for item in batch]


class Batch:
    def __init__(self, iterator, batch_size: int):
        self.iterator = iterator
        self.batch_size = batch_size

    def __len__(self) -> int:
        return len(self.iterator) // self.batch_size

    def __iter__(self):
        batch = []
        for item in self.iterator:
            batch.append(item)

            if len(batch) == self.batch_size:
                yield [torch.stack(xs) for xs in zip(*batch)]
                batch.clear()


class SubTiler:
    """Subtile a dataset which loads faces"""

    def __init__(
        self,
        face_iterator,
        nside: int,
        tile_size: int,
        shuffle=True,
        num_steps: int = 1,
    ):
        self.face_iterator = face_iterator
        self.nside = nside
        self.tile_size = tile_size
        self.shuffle = shuffle
        self.num_steps = max(1, int(num_steps))
        n = self.tile_size
        self.tiles = [
            (slice(i * n, (i + 1) * n), slice(j * n, (j + 1) * n))
            for i in range(self.nside // n)
            for j in range(self.nside // n)
        ]

    def __len__(self):
        return len(self.face_iterator) * len(self.tiles)

    def __iter__(self):
        for batch in self.face_iterator:
            # Parse batch based on num_steps:
            # num_steps=1: (inputs, targets, index, j) - 4 elements
            # num_steps>1: (inputs, targets [T,C,H,W], index, j, forcings [T-1,C,H,W]) - 5 elements
            if self.num_steps == 1:
                if len(batch) != 4:
                    raise RuntimeError(
                        f"Expected 4 elements for num_steps=1, got {len(batch)}"
                    )
                inputs, targets, index, j = batch
                forcings = None
            else:
                if len(batch) != 5:
                    raise RuntimeError(
                        f"Expected 5 elements for num_steps>1, got {len(batch)}"
                    )
                inputs, targets, index, j, forcings = batch

            if self.shuffle:
                random.shuffle(self.tiles)
            for xt, yt in self.tiles:
                input_patch = inputs[..., yt, xt]
                # targets shape: [T, C, H, W] for multistep, [C, H, W] for single step
                target_patch = targets[..., yt, xt]
                index_patch = index[..., yt, xt]

                if self.num_steps == 1:
                    yield input_patch, target_patch, index_patch, j
                else:
                    forcing_patch = forcings[..., yt, xt]
                    yield input_patch, target_patch, index_patch, j, forcing_patch


class ToTuple:
    def __init__(self, dali_iterator, num_steps: int = 1):
        self.dali_iterator = dali_iterator
        self.num_steps = max(1, int(num_steps))

    def __len__(self):
        return len(self.dali_iterator)

    def __iter__(self):
        for data in self.dali_iterator:
            batch = data[0]
            if self.num_steps == 1:
                yield batch["s0"], batch["s1"], batch["index"], batch["j"]
            else:
                # For multistep: s0, s_outputs [T,C,H,W], index, j, s_forcings [T-1,C,H,W]
                yield (
                    batch["s0"],
                    batch["s_outputs"],
                    batch["index"],
                    batch["j"],
                    batch["s_forcings"],
                )


class WithLength:
    def __init__(self, iterator, len):
        self.iterator = iterator
        self._len = len

    def __len__(self):
        return self._len

    def __iter__(self):
        return iter(self.iterator)


class CrossFaceSubTiler:
    """
    Subtile padded face tensors into overlapped tiles.
    Expects items shaped:
      - s0: [B, C, Hpad, Wpad], index: [B, Hpad, Wpad], j: [1]
      - num_steps=1: s1 [B, C, Hpad, Wpad]
      - num_steps>1: s_outputs [B, T, C, Hpad, Wpad], s_forcings [B, T-1, C, Hpad, Wpad]
    Emits (preserving leading batch B so Unbatch can split):
      - num_steps=1: [B, C, H, W], [B, C, H, W], [B, H, W], j
      - num_steps>1: [B, C, H, W], [B, T, C, H, W], [B, H, W], j, [B, T-1, C, H, W]
    """

    def __init__(
        self,
        iterator,
        *,
        nside: int,
        tile_size: int,
        shuffle_tiles: bool = True,
        num_steps: int = 1,
        skip_corner_tiles: bool = True,
        stride: int = None,
        balance_cross_face: bool = False,
    ):
        self.iterator = iterator
        self.nside = nside
        self.tile_size = tile_size
        if tile_size % 2 != 0:
            raise ValueError(f"tile_size must be even, got {tile_size}")
        self.overlap = tile_size // 2
        self.stride = tile_size - self.overlap
        self.shuffle_tiles = shuffle_tiles
        self.num_steps = max(1, int(num_steps))
        self.skip_corner_tiles = skip_corner_tiles
        self.balance_cross_face = balance_cross_face
        # Precompute tiling grid on padded faces
        padded_size = self.nside + 2 * self.overlap
        self.cx = (padded_size - self.tile_size) // self.stride + 1
        self.cy = (padded_size - self.tile_size) // self.stride + 1
        cross_face_tiles = []
        interior_tiles = []
        for iy in range(self.cy):
            for ix in range(self.cx):
                if skip_corner_tiles and (
                    (ix == 0 and iy == 0)
                    or (ix == 0 and iy == self.cy - 1)
                    or (ix == self.cx - 1 and iy == 0)
                    or (ix == self.cx - 1 and iy == self.cy - 1)
                ):
                    continue
                # A tile is cross-face if it overlaps the halo region
                is_cross_face = (
                    ix == 0 or ix == self.cx - 1 or iy == 0 or iy == self.cy - 1
                )
                if is_cross_face:
                    cross_face_tiles.append((ix, iy))
                else:
                    interior_tiles.append((ix, iy))
        self.cross_face_tiles = cross_face_tiles
        self.interior_tiles = interior_tiles
        self.tiles = cross_face_tiles + interior_tiles

    def _balanced_tiles(self):
        """Subsample the larger group to match the smaller one."""
        n_cross = len(self.cross_face_tiles)
        n_interior = len(self.interior_tiles)
        if n_cross <= n_interior:
            selected_interior = random.sample(self.interior_tiles, n_cross)
            return self.cross_face_tiles + selected_interior
        else:
            selected_cross = random.sample(self.cross_face_tiles, n_interior)
            return selected_cross + self.interior_tiles

    def __len__(self):
        if self.balance_cross_face:
            n = min(len(self.cross_face_tiles), len(self.interior_tiles))
            return len(self.iterator) * 2 * n
        return len(self.iterator) * len(self.tiles)

    def __iter__(self):
        stride = self.stride
        for batch in self.iterator:
            # Parse batch based on num_steps:
            # num_steps=1: (s0, s1, index, j) - 4 elements
            # num_steps>1: (s0, s_outputs [B,T,C,H,W], index, j, s_forcings [B,T-1,C,H,W]) - 5 elements
            if self.num_steps == 1:
                if len(batch) != 4:
                    raise RuntimeError(
                        f"Expected 4 elements for num_steps=1, got {len(batch)}"
                    )
                s0, s1, index, j = batch
                s_outputs = None
                s_forcings = None
            else:
                if len(batch) != 5:
                    raise RuntimeError(
                        f"Expected 5 elements for num_steps>1, got {len(batch)}"
                    )
                s0, s_outputs, index, j, s_forcings = batch

            tiles = self._balanced_tiles() if self.balance_cross_face else self.tiles
            if self.shuffle_tiles:
                random.shuffle(tiles)

            for ix, iy in tiles:
                xs = ix * stride
                ys = iy * stride
                xe = xs + self.tile_size
                ye = ys + self.tile_size
                input_patch = s0[:, :, ys:ye, xs:xe]
                index_patch = index[..., ys:ye, xs:xe]

                if self.num_steps == 1:
                    target_patch = s1[:, :, ys:ye, xs:xe]
                    yield input_patch, target_patch, index_patch, j
                else:
                    # s_outputs: [B, T, C, Hpad, Wpad] -> [B, T, C, H, W]
                    target_patch = s_outputs[:, :, :, ys:ye, xs:xe]
                    # s_forcings: [B, T-1, C, Hpad, Wpad] -> [B, T-1, C, H, W]
                    forcing_patch = s_forcings[:, :, :, ys:ye, xs:xe]
                    yield input_patch, target_patch, index_patch, j, forcing_patch
