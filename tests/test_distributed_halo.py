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
import numpy as np
import pytest
import torch

import screamcast.distributed_halo as distributed_halo
from screamcast.distributed_halo import (
    DistributedTileKNNHaloPadding_AllGather,
    TileTopology,
)


def test_tile_topology_partitions_tiles_by_rank():
    topology = TileTopology(
        world_size=4,
        rank=1,
        face_size=8,
        tile_size=4,
        halo_width=2,
    )

    assert topology.tiles_per_dim == 2
    assert topology.total_tiles == 24
    assert topology.tiles_per_rank == 6
    assert topology.padded_tile_size == 8
    assert topology.local_tiles == [
        (1, 1, 0),
        (1, 1, 1),
        (2, 0, 0),
        (2, 0, 1),
        (2, 1, 0),
        (2, 1, 1),
    ]


def test_tile_topology_coord_transforms():
    topology = TileTopology(
        world_size=1,
        rank=0,
        face_size=4,
        tile_size=2,
        halo_width=1,
    )
    base_coords = {"variable": np.array(["A", "B"]), "lead_time": np.array([0])}

    interior = topology.interior_coords(
        base_coords, time=np.array([np.datetime64("2020-01-01")])
    )
    padded = topology.pad_coords(interior)
    cropped = topology.crop_coords(padded)

    np.testing.assert_array_equal(interior["face"], np.arange(topology.total_tiles))
    np.testing.assert_array_equal(interior["x"], np.arange(2))
    np.testing.assert_array_equal(interior["y"], np.arange(2))
    np.testing.assert_array_equal(padded["x"], np.arange(4))
    np.testing.assert_array_equal(padded["y"], np.arange(4))
    np.testing.assert_array_equal(cropped["x"], np.arange(2))
    np.testing.assert_array_equal(cropped["y"], np.arange(2))
    np.testing.assert_array_equal(cropped["time"], interior["time"])


class _FakeRegrid(torch.nn.Module):
    def __init__(self, output_size: int):
        super().__init__()
        self.output_size = output_size
        self.inputs: list[torch.Tensor] = []
        self.last_input = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.inputs.append(x.clone())
        self.last_input = x.clone()
        leading_shape = x.shape[:-1]
        values = torch.arange(
            int(np.prod(leading_shape)) * self.output_size,
            device=x.device,
            dtype=x.dtype,
        )
        return values.reshape(*leading_shape, self.output_size)


class _SliceRegrid(torch.nn.Module):
    def __init__(self, output_size: int):
        super().__init__()
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., : self.output_size]


def test_halo_exchange_pad_and_crop(monkeypatch):
    topology = TileTopology(
        world_size=2,
        rank=0,
        face_size=4,
        tile_size=2,
        halo_width=1,
    )
    regrid = _FakeRegrid(
        output_size=topology.tiles_per_rank * topology.padded_tile_size**2
    )
    halo_exchange = DistributedTileKNNHaloPadding_AllGather(
        topology=topology,
        regrid=regrid,
        lat_deg=torch.zeros(
            topology.tiles_per_rank,
            topology.padded_tile_size,
            topology.padded_tile_size,
        ),
        lon_deg=torch.zeros(
            topology.tiles_per_rank,
            topology.padded_tile_size,
            topology.padded_tile_size,
        ),
    )

    def fake_all_gather(tensor_list, x):
        tensor_list[0].copy_(x)
        tensor_list[1].copy_(x + 100.0)

    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)

    x = torch.arange(
        1 * 3 * topology.tiles_per_rank * 2 * 2, dtype=torch.float32
    ).reshape(1, 3, topology.tiles_per_rank, 2, 2)
    padded = halo_exchange(x)

    assert padded.shape == (1, 3, topology.tiles_per_rank, 4, 4)
    expected_input = torch.cat([x, x + 100.0], dim=-3).reshape(1, 3, -1)
    assert len(regrid.inputs) == 1
    torch.testing.assert_close(regrid.inputs[0], expected_input)

    cropped = topology.crop(padded)
    assert cropped.shape == (1, 3, topology.tiles_per_rank, 2, 2)
    torch.testing.assert_close(cropped, padded[..., 1:-1, 1:-1])


def test_normalized_halo_adjoint_reduce_scatter_accumulates_remote_contributions(
    monkeypatch,
):
    topology = TileTopology(
        world_size=2,
        rank=0,
        face_size=2,
        tile_size=2,
        halo_width=0,
    )
    halo_exchange = DistributedTileKNNHaloPadding_AllGather(
        topology=topology,
        regrid=_SliceRegrid(topology.tiles_per_rank * topology.padded_tile_size**2),
        lat_deg=torch.zeros(
            topology.tiles_per_rank,
            topology.padded_tile_size,
            topology.padded_tile_size,
        ),
        lon_deg=torch.zeros(
            topology.tiles_per_rank,
            topology.padded_tile_size,
            topology.padded_tile_size,
        ),
    )
    calls: list[list[torch.Tensor]] = []

    def fake_reduce_scatter(output, input_list, op=None, group=None):
        del op, group
        calls.append([tensor.clone() for tensor in input_list])
        remote = calls[-1][0] * 3.0
        output.copy_(input_list[0] + remote)

    monkeypatch.setattr(torch.distributed, "reduce_scatter", fake_reduce_scatter)

    halo_pseudoinverse = distributed_halo.NormalizedHaloAdjoint(halo_exchange)
    padded = torch.arange(
        topology.tiles_per_rank * topology.padded_tile_size**2,
        dtype=torch.float32,
    ).reshape(
        1,
        1,
        topology.tiles_per_rank,
        topology.padded_tile_size,
        topology.padded_tile_size,
    )

    unpadded = halo_pseudoinverse(padded)

    assert len(calls) == 2
    local_flat = padded.reshape(1, 1, -1)
    expected = (local_flat * 4.0 / 4.0).reshape_as(unpadded)
    torch.testing.assert_close(unpadded, expected)


def test_halo_exchange_pad_validates_interior_shape():
    topology = TileTopology(
        world_size=1,
        rank=0,
        face_size=4,
        tile_size=2,
        halo_width=1,
    )
    halo_exchange = DistributedTileKNNHaloPadding_AllGather(
        topology=topology,
        regrid=_FakeRegrid(topology.tiles_per_rank * topology.padded_tile_size**2),
        lat_deg=torch.zeros(
            topology.tiles_per_rank,
            topology.padded_tile_size,
            topology.padded_tile_size,
        ),
        lon_deg=torch.zeros(
            topology.tiles_per_rank,
            topology.padded_tile_size,
            topology.padded_tile_size,
        ),
    )

    x = torch.zeros(1, 1, topology.tiles_per_rank, 3, 2)
    with pytest.raises(ValueError, match="Expected interior tile shape"):
        halo_exchange(x)


def test_halo_exchange_is_module_with_buffers():
    topology = TileTopology(
        world_size=1,
        rank=0,
        face_size=4,
        tile_size=2,
        halo_width=1,
    )
    halo_exchange = DistributedTileKNNHaloPadding_AllGather(
        topology=topology,
        regrid=_FakeRegrid(topology.tiles_per_rank * topology.padded_tile_size**2),
        lat_deg=torch.ones(
            topology.tiles_per_rank,
            topology.padded_tile_size,
            topology.padded_tile_size,
        ),
        lon_deg=torch.zeros(
            topology.tiles_per_rank,
            topology.padded_tile_size,
            topology.padded_tile_size,
        ),
    )

    assert isinstance(halo_exchange, torch.nn.Module)
    buffers = dict(halo_exchange.named_buffers())
    assert "lat_deg" in buffers
    assert "lon_deg" in buffers
    torch.testing.assert_close(buffers["lat_deg"], halo_exchange.lat_deg)
    torch.testing.assert_close(buffers["lon_deg"], halo_exchange.lon_deg)


def test_gather_tiles_to_faces_single_rank_identity(monkeypatch):
    """With world_size=1, gather should place every local tile into its face slot."""
    topology = TileTopology(
        world_size=1,
        rank=0,
        face_size=4,
        tile_size=2,
        halo_width=1,
    )

    def fake_all_gather(tensor_list, x):
        tensor_list[0].copy_(x)

    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)

    # Each tile filled with a distinct value equal to its global index.
    x = torch.zeros(
        2, 3, topology.tiles_per_rank, topology.tile_size, topology.tile_size
    )
    for i in range(topology.tiles_per_rank):
        x[..., i, :, :] = float(i)

    out = topology.gather_tiles_to_faces(x)
    assert out.shape == (2, 3, topology.n_faces, topology.face_size, topology.face_size)
    ts = topology.tile_size
    for global_index, (face, ti, tj) in enumerate(topology.global_tiles):
        block = out[..., face, ti * ts : (ti + 1) * ts, tj * ts : (tj + 1) * ts]
        assert torch.all(
            block == float(global_index)
        ), f"tile {(face, ti, tj)} missing expected value {global_index}"


def test_gather_tiles_to_faces_multi_rank_interleave(monkeypatch):
    """With world_size=2, gather order matches global_tiles (rank 0 first, then rank 1)."""
    topology = TileTopology(
        world_size=2,
        rank=0,
        face_size=4,
        tile_size=2,
        halo_width=0,
    )
    # rank 0 owns the first tiles_per_rank global tiles.
    tpr = topology.tiles_per_rank

    def fake_all_gather(tensor_list, x):
        # tensor_list[0] <- rank 0's tiles (as provided by x)
        tensor_list[0].copy_(x)
        # Simulate rank 1's tiles as shifted values (+ tpr) so we can track them.
        shifted = x + float(tpr)
        tensor_list[1].copy_(shifted)

    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)

    x = torch.zeros(1, topology.tiles_per_rank, topology.tile_size, topology.tile_size)
    for i in range(tpr):
        x[:, i, :, :] = float(i)

    out = topology.gather_tiles_to_faces(x)
    assert out.shape == (1, topology.n_faces, topology.face_size, topology.face_size)
    ts = topology.tile_size
    for global_index, (face, ti, tj) in enumerate(topology.global_tiles):
        block = out[:, face, ti * ts : (ti + 1) * ts, tj * ts : (tj + 1) * ts]
        expected = float(global_index)
        assert torch.all(
            block == expected
        ), f"tile {(face, ti, tj)} missing expected value {expected}"


def test_gather_tiles_to_faces_validates_shape(monkeypatch):
    topology = TileTopology(
        world_size=1,
        rank=0,
        face_size=4,
        tile_size=2,
        halo_width=0,
    )

    def fake_all_gather(tensor_list, x):
        tensor_list[0].copy_(x)

    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)

    wrong_shape = torch.zeros(1, topology.tiles_per_rank, 3, 2)
    with pytest.raises(ValueError, match="Expected interior tile shape"):
        topology.gather_tiles_to_faces(wrong_shape)

    wrong_tpr = torch.zeros(1, topology.tiles_per_rank - 1, 2, 2)
    with pytest.raises(ValueError, match="local tiles"):
        topology.gather_tiles_to_faces(wrong_tpr)


def test_halo_exchange_from_padded_face_grid_builds_expected_latlon(monkeypatch):
    topology = TileTopology(
        world_size=1,
        rank=0,
        face_size=4,
        tile_size=2,
        halo_width=1,
    )

    face_extent = topology.face_size + 2
    lon = torch.arange(6 * face_extent * face_extent, dtype=torch.float32).reshape(
        6, face_extent, face_extent
    )
    lat = lon + 1000.0

    calls = {}

    class _FakeInterpolator(torch.nn.Module):
        def __init__(self, src_lon, src_lat, tgt_lon, tgt_lat, k, eps):
            super().__init__()
            calls["src_lon"] = src_lon.clone()
            calls["src_lat"] = src_lat.clone()
            calls["tgt_lon"] = tgt_lon.clone()
            calls["tgt_lat"] = tgt_lat.clone()
            calls["k"] = k
            calls["eps"] = eps

        def float(self):
            return self

        def to(self, device):
            calls["device"] = device
            return self

    monkeypatch.setattr(distributed_halo, "KNNS2Interpolator", _FakeInterpolator)

    halo_exchange = DistributedTileKNNHaloPadding_AllGather.from_padded_face_grid(
        topology=topology,
        lon=lon,
        lat=lat,
        pad_width_data=1,
        device=torch.device("cpu"),
    )

    assert halo_exchange.lon_deg.shape == (
        topology.tiles_per_rank,
        topology.padded_tile_size,
        topology.padded_tile_size,
    )
    assert halo_exchange.lat_deg.shape == (
        topology.tiles_per_rank,
        topology.padded_tile_size,
        topology.padded_tile_size,
    )
    assert calls["src_lon"].numel() == topology.total_tiles * topology.tile_size**2
    assert calls["src_lat"].numel() == topology.total_tiles * topology.tile_size**2
    assert (
        calls["tgt_lon"].numel()
        == topology.tiles_per_rank * topology.padded_tile_size**2
    )
    assert (
        calls["tgt_lat"].numel()
        == topology.tiles_per_rank * topology.padded_tile_size**2
    )
    assert calls["k"] == 4
    assert calls["eps"] == 1e-7


def test_jvp_inference_mode_interaction():
    # Create tensors outside
    x = torch.ones(3)
    _, vjp_fn = torch.func.vjp(torch.sin, x)

    with torch.inference_mode():
        # ok to create or run the vjp in inference mode
        (o,) = vjp_fn(x)
        assert not torch.all(o == 0.0)
