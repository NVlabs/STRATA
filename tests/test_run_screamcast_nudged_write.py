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
import pytest
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
    # With output_steps=2, step_index 0 (lead 1) is skipped and step_index 1
    # (lead 2) goes to frame 0.
    write_step(step_index=0, out_tensor=out_tensor)
    assert call_count["value"] == 0
    assert np.all(store["PRECT"][:] == 0.0)
    assert np.all(store["U"][:] == 0.0)

    write_step(step_index=1, out_tensor=out_tensor)
    write_step.close()

    assert call_count["value"] == 1
    expected_owned_tiles = torch.cat(
        [out_tensor[:, [1, 2, 4]], remote_owned_tiles], dim=2
    )
    np.testing.assert_allclose(store["U"][0, 0, 0], expected_owned_tiles[0, 0].numpy())
    np.testing.assert_allclose(store["U"][0, 0, 1], expected_owned_tiles[0, 1].numpy())
    np.testing.assert_allclose(store["PRECT"][0, 0], expected_owned_tiles[0, 2].numpy())
    assert np.all(store["U"][0, 1] == 0.0)
    assert np.all(store["PRECT"][0, 1] == 0.0)


@pytest.mark.parametrize("output_steps", [1, 2, 3, 6])
def test_face_zarr_write_step_frames_hold_the_lead_their_step_label_names(
    tmp_path, output_steps
):
    """Each saved frame holds the forecast at the lead its ``step`` coordinate names.

    The forecast for zero-based ``step_index`` is at lead ``step_index + 1``;
    the test fills it with that lead (in units of dt) and checks the store
    against its own ``step`` labels from ``prepare_output_store``.
    """
    n_steps = 6
    dt = np.timedelta64(600, "s")
    grouped = OrderedDict([("lead", [(None, 0)])])
    path = str(tmp_path / "forecast.zarr")
    zarr_writer.prepare_output_store(
        output_path=path,
        grouped_variables=grouped,
        n_times=1,
        n_steps=n_steps // output_steps,
        nside=2,
        tile_size=2,
        time_values=np.array([np.datetime64("2020-10-13T00")]),
        lat=np.zeros((6, 2, 2)),
        lon=np.zeros((6, 2, 2)),
        dt=dt * output_steps,
    )
    store = zarr.open_group(path, mode="a")
    write_step = zarr_writer.FaceZarrWriteStep(
        output_steps=output_steps,
        topology=_FakeTopology(world_size=1, rank=0),
        grouped_variables=grouped,
        store=store,
    )
    for step_index in range(n_steps):
        lead = float(step_index + 1)
        write_step(step_index, torch.full((1, 1, 6, 2, 2), lead))
    write_step.close()

    labelled_leads = store["step"][:] / dt
    np.testing.assert_array_equal(
        labelled_leads, np.arange(1, n_steps // output_steps + 1) * output_steps
    )
    written = store["lead"][0]
    for frame, lead in enumerate(labelled_leads):
        np.testing.assert_array_equal(written[frame], np.full((6, 2, 2), lead))


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
