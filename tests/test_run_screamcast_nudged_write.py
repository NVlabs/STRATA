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
from collections import OrderedDict

import numpy as np
import torch
import zarr

from screamcast import zarr_writer


def test_reindex_grouped_variables_compacts_channel_indices():
    grouped = OrderedDict(
        [
            ("B", [(0, 3), (1, 4)]),
            ("D", [(None, 7)]),
        ]
    )

    local_grouped = zarr_writer.reindex_grouped_variables(grouped)

    assert local_grouped == OrderedDict(
        [
            ("B", [(0, 0), (1, 1)]),
            ("D", [(None, 2)]),
        ]
    )


class _FakeTopology:
    def __init__(self, *, world_size: int, rank: int, tile_size: int = 2):
        self.world_size = world_size
        self.rank = rank
        self.tile_size = tile_size
        self.tiles_per_rank = 6 // world_size
        self.total_tiles = self.tiles_per_rank * world_size
        self.face_size = tile_size
        self.n_faces = 6
        self.global_tiles = [(face, 0, 0) for face in range(self.total_tiles)]


def test_face_zarr_write_step_selects_owned_channels_and_writes_faces(monkeypatch):
    grouped = OrderedDict(
        [
            ("U", [(0, 1), (1, 2)]),
            ("PRECT", [(None, 4)]),
            ("T", [(None, 0)]),
            ("V", [(None, 3)]),
        ]
    )
    topology = _FakeTopology(world_size=2, rank=0)
    store = zarr.open_group(zarr.storage.MemoryStore(), mode="w")
    store.create_array(
        "U",
        shape=(1, 2, 2, 6, 2, 2),
        chunks=(1, 1, 2, 1, 2, 2),
        dtype="f4",
        fill_value=0.0,
    )
    store.create_array(
        "PRECT",
        shape=(1, 2, 6, 2, 2),
        chunks=(1, 1, 1, 2, 2),
        dtype="f4",
        fill_value=0.0,
    )
    write_step = zarr_writer.FaceZarrWriteStep(
        output_steps=2,
        topology=topology,
        grouped_variables=grouped,
        store=store,
    )
    out_tensor = torch.arange(1 * 5 * 3 * 2 * 2, dtype=torch.float32).reshape(
        1, 5, 3, 2, 2
    )
    remote_owned_tiles = (
        torch.arange(1 * 3 * 3 * 2 * 2, dtype=torch.float32).reshape(1, 3, 3, 2, 2)
        + 1000.0
    )

    call_count = {"value": 0}

    def fake_all_to_all_single(
        output, input, output_split_sizes=None, input_split_sizes=None, group=None
    ):
        del group
        call_count["value"] += 1
        assert input_split_sizes == [36, 24]
        assert output_split_sizes == [36, 36]
        torch.testing.assert_close(input[:36], out_tensor[:, [1, 2, 4]].reshape(-1))
        torch.testing.assert_close(input[36:], out_tensor[:, [0, 3]].reshape(-1))
        output.copy_(
            torch.cat(
                [out_tensor[:, [1, 2, 4]].reshape(-1), remote_owned_tiles.reshape(-1)]
            )
        )

    monkeypatch.setattr(torch.distributed, "all_to_all_single", fake_all_to_all_single)
    write_step(step_index=1, out_tensor=out_tensor)
    assert call_count["value"] == 0
    assert np.all(store["PRECT"][:] == 0.0)
    assert np.all(store["U"][:] == 0.0)

    write_step(step_index=2, out_tensor=out_tensor)
    write_step.close()

    assert call_count["value"] == 1
    expected_owned_tiles = torch.cat(
        [out_tensor[:, [1, 2, 4]], remote_owned_tiles], dim=2
    )
    np.testing.assert_allclose(store["U"][0, 1, 0], expected_owned_tiles[0, 0].numpy())
    np.testing.assert_allclose(store["U"][0, 1, 1], expected_owned_tiles[0, 1].numpy())
    np.testing.assert_allclose(store["PRECT"][0, 1], expected_owned_tiles[0, 2].numpy())


def test_face_zarr_write_step_empty_owner_still_participates_in_collective(
    monkeypatch,
):
    grouped = OrderedDict(
        [
            ("A", [(None, 0)]),
            ("B", [(None, 1)]),
        ]
    )
    topology = _FakeTopology(world_size=3, rank=2)
    store = zarr.open_group(zarr.storage.MemoryStore(), mode="w")
    store.create_array(
        "A",
        shape=(1, 1, 6, 2, 2),
        chunks=(1, 1, 1, 2, 2),
        dtype="f4",
        fill_value=0.0,
    )
    store.create_array(
        "B",
        shape=(1, 1, 6, 2, 2),
        chunks=(1, 1, 1, 2, 2),
        dtype="f4",
        fill_value=0.0,
    )
    write_step = zarr_writer.FaceZarrWriteStep(
        output_steps=1,
        topology=topology,
        grouped_variables=grouped,
        store=store,
    )
    out_tensor = torch.arange(1 * 2 * 2 * 2 * 2, dtype=torch.float32).reshape(
        1, 2, 2, 2, 2
    )

    call_count = {"value": 0}

    def fake_all_to_all_single(
        output, input, output_split_sizes=None, input_split_sizes=None, group=None
    ):
        del group, input
        call_count["value"] += 1
        assert output.numel() == 0
        assert input_split_sizes == [8, 8, 0]
        assert output_split_sizes == [0, 0, 0]

    monkeypatch.setattr(torch.distributed, "all_to_all_single", fake_all_to_all_single)
    write_step(step_index=0, out_tensor=out_tensor)
    write_step.close()

    assert call_count["value"] == 1
    assert np.all(store["A"][:] == 0.0)
    assert np.all(store["B"][:] == 0.0)
