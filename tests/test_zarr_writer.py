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
"""Unit tests for ZarrWriter, _split_variable_levels, and prepare_output_store."""

from collections import OrderedDict

import numpy as np
import torch
import zarr

from screamcast.zarr_writer import (
    ZarrWriter,
    _split_variable_levels,
    partition_grouped_variables,
    prepare_latlon_plev_store,
    prepare_output_store,
)

TILE = 4
N_TIMES = 2
N_STEPS = 3
N_FACES = 6
NSIDE = 8


def _make_store(grouped_variables: OrderedDict) -> zarr.Group:
    store = zarr.open_group(zarr.storage.MemoryStore(), mode="w")
    for base_name, entries in grouped_variables.items():
        if entries[0][0] is None:
            store.create_array(
                base_name,
                shape=(N_TIMES, N_STEPS, N_FACES, NSIDE, NSIDE),
                chunks=(1, 1, 1, TILE, TILE),
                dtype="f4",
                fill_value=0.0,
            )
        else:
            store.create_array(
                base_name,
                shape=(N_TIMES, N_STEPS, len(entries), N_FACES, NSIDE, NSIDE),
                chunks=(1, 1, len(entries), 1, TILE, TILE),
                dtype="f4",
                fill_value=0.0,
            )
    return store


def _tile(value: float, n_channels: int = 1) -> torch.Tensor:
    """Make a [n_channels, TILE, TILE] tensor filled with value."""
    return torch.full((n_channels, TILE, TILE), value)


def test_write_2d_variable():
    """ZarrWriter writes a 2D variable tile to the correct zarr location."""
    grouped = OrderedDict([("T2m", [(None, 0)])])
    store = _make_store(grouped)
    writer = ZarrWriter(store, grouped, tile_size=TILE)

    writer.write(out_index=0, step_index=1, face=2, x0=0, y0=4, tensor=_tile(42.0))
    writer.close()

    result = store["T2m"][0, 1, 2, 0:TILE, 4 : 4 + TILE]
    assert np.all(result == 42.0), f"Expected 42.0, got {result}"


def test_write_3d_variable():
    """ZarrWriter writes a 3D variable (multiple levels) to the correct location."""
    n_levels = 3
    grouped = OrderedDict([("T", [(i, i) for i in range(n_levels)])])
    store = _make_store(grouped)
    writer = ZarrWriter(store, grouped, tile_size=TILE)

    # channels: level0=1.0, level1=2.0, level2=3.0
    data = torch.stack([_tile(float(v))[0] for v in [1.0, 2.0, 3.0]])  # [3, TILE, TILE]
    writer.write(out_index=1, step_index=0, face=0, x0=0, y0=0, tensor=data)
    writer.close()

    result = store["T"][1, 0, :, 0, 0:TILE, 0:TILE]
    assert result.shape == (n_levels, TILE, TILE)
    for lev_idx, expected in enumerate([1.0, 2.0, 3.0]):
        assert np.all(
            result[lev_idx] == expected
        ), f"Level {lev_idx}: expected {expected}, got {result[lev_idx]}"


def test_write_multiple_steps():
    """ZarrWriter correctly indexes the step dimension for multiple writes."""
    grouped = OrderedDict([("U", [(None, 0)])])
    store = _make_store(grouped)
    writer = ZarrWriter(store, grouped, tile_size=TILE)

    for step in range(N_STEPS):
        writer.write(
            out_index=0, step_index=step, face=0, x0=0, y0=0, tensor=_tile(float(step))
        )
    writer.close()

    for step in range(N_STEPS):
        result = store["U"][0, step, 0, 0:TILE, 0:TILE]
        assert np.all(
            result == float(step)
        ), f"Step {step}: expected {step}, got {result}"


def test_write_mixed_2d_3d():
    """ZarrWriter handles a mix of 2D and 3D variables in one pass."""
    grouped = OrderedDict(
        [
            ("T2m", [(None, 0)]),
            ("T", [(0, 1), (1, 2)]),
        ]
    )
    store = _make_store(grouped)
    writer = ZarrWriter(store, grouped, tile_size=TILE)

    # ch0=T2m=7.0, ch1=T_lev0=8.0, ch2=T_lev1=9.0
    data = torch.stack([_tile(v)[0] for v in [7.0, 8.0, 9.0]])  # [3, TILE, TILE]
    writer.write(out_index=0, step_index=0, face=0, x0=0, y0=0, tensor=data)
    writer.close()

    assert np.all(store["T2m"][0, 0, 0, 0:TILE, 0:TILE] == 7.0)
    assert np.all(store["T"][0, 0, 0, 0, 0:TILE, 0:TILE] == 8.0)
    assert np.all(store["T"][0, 0, 1, 0, 0:TILE, 0:TILE] == 9.0)


# ---------------------------------------------------------------------------
# _split_variable_levels
# ---------------------------------------------------------------------------


def test_split_variable_levels_2d():
    names = ["PRECT", "T2m"]
    grouped = _split_variable_levels(names)
    assert list(grouped.keys()) == ["PRECT", "T2m"]
    assert grouped["PRECT"] == [(None, 0)]
    assert grouped["T2m"] == [(None, 1)]


def test_split_variable_levels_3d():
    names = ["U_0", "U_1", "U_2"]
    grouped = _split_variable_levels(names)
    assert list(grouped.keys()) == ["U"]
    assert grouped["U"] == [(0, 0), (1, 1), (2, 2)]


def test_split_variable_levels_mixed():
    names = ["PRECT", "U_0", "U_1", "T2m"]
    grouped = _split_variable_levels(names)
    assert list(grouped.keys()) == ["PRECT", "U", "T2m"]
    assert grouped["PRECT"] == [(None, 0)]
    assert grouped["U"] == [(0, 1), (1, 2)]
    assert grouped["T2m"] == [(None, 3)]


def test_split_variable_levels_preserves_order():
    names = ["Z_2", "Z_0", "Z_1"]
    grouped = _split_variable_levels(names)
    # Entries should reflect channel index order, not sorted by level
    assert grouped["Z"] == [(2, 0), (0, 1), (1, 2)]


def test_partition_grouped_variables_balances_base_variables():
    grouped = OrderedDict(
        [
            ("A", [(None, 0)]),
            ("B", [(0, 1), (1, 2)]),
            ("C", [(None, 3)]),
            ("D", [(None, 4)]),
            ("E", [(0, 5), (1, 6)]),
        ]
    )

    parts = partition_grouped_variables(grouped, 3)

    assert [list(part.keys()) for part in parts] == [
        ["A", "B"],
        ["C", "D"],
        ["E"],
    ]


# ---------------------------------------------------------------------------
# prepare_output_store
# ---------------------------------------------------------------------------

_NSIDE = 8
_TILE = 4
_N_TIMES = 2
_N_STEPS = 3


def _make_lat_lon():
    lat = np.zeros((6, _NSIDE, _NSIDE), dtype=np.float32)
    lon = np.ones((6, _NSIDE, _NSIDE), dtype=np.float32)
    return lat, lon


def test_prepare_output_store_2d(tmp_path):
    grouped = _split_variable_levels(["PRECT", "T2m"])
    lat, lon = _make_lat_lon()
    times = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[s]")

    store = prepare_output_store(
        output_path=str(tmp_path / "out.zarr"),
        grouped_variables=grouped,
        n_times=_N_TIMES,
        n_steps=_N_STEPS,
        nside=_NSIDE,
        tile_size=_TILE,
        time_values=times,
        lat=lat,
        lon=lon,
    )

    for var in ["PRECT", "T2m"]:
        assert var in store
        assert store[var].shape == (_N_TIMES, _N_STEPS, 6, _NSIDE, _NSIDE)
        assert store[var].chunks == (1, 1, 1, _TILE, _TILE)

    assert store["lat"].shape == (6, _NSIDE, _NSIDE)
    assert store["lon"].shape == (6, _NSIDE, _NSIDE)
    assert store["time"].shape == (_N_TIMES,)
    assert store["step"].shape == (_N_STEPS,)


def test_prepare_output_store_3d(tmp_path):
    grouped = _split_variable_levels(["U_0", "U_1"])
    lat, lon = _make_lat_lon()
    times = np.array(["2020-01-01"], dtype="datetime64[s]")

    store = prepare_output_store(
        output_path=str(tmp_path / "out.zarr"),
        grouped_variables=grouped,
        n_times=1,
        n_steps=_N_STEPS,
        nside=_NSIDE,
        tile_size=_TILE,
        time_values=times,
        lat=lat,
        lon=lon,
    )

    assert store["U"].shape == (1, _N_STEPS, 2, 6, _NSIDE, _NSIDE)
    assert store["U"].chunks == (1, 1, 1, 1, _TILE, _TILE)


def test_prepare_output_store_with_dt(tmp_path):
    grouped = _split_variable_levels(["PRECT"])
    lat, lon = _make_lat_lon()
    times = np.array(["2020-01-01"], dtype="datetime64[s]")
    dt = np.timedelta64(10 * 60, "s")

    store = prepare_output_store(
        output_path=str(tmp_path / "out.zarr"),
        grouped_variables=grouped,
        n_times=1,
        n_steps=_N_STEPS,
        nside=_NSIDE,
        tile_size=_TILE,
        time_values=times,
        lat=lat,
        lon=lon,
        dt=dt,
    )

    steps = store["step"][:]
    expected = np.arange(1, _N_STEPS + 1) * dt
    np.testing.assert_array_equal(steps, expected)


def test_prepare_output_store_with_hyam_hybm(tmp_path):
    grouped = _split_variable_levels(["T_0", "T_1", "T_2"])
    lat, lon = _make_lat_lon()
    times = np.array(["2020-01-01"], dtype="datetime64[s]")
    hyam = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    hybm = np.array([0.9, 0.8, 0.7], dtype=np.float32)

    store = prepare_output_store(
        output_path=str(tmp_path / "out.zarr"),
        grouped_variables=grouped,
        n_times=1,
        n_steps=_N_STEPS,
        nside=_NSIDE,
        tile_size=_TILE,
        time_values=times,
        lat=lat,
        lon=lon,
        hyam=hyam,
        hybm=hybm,
    )

    assert "level" in store
    assert "hyam" in store
    assert "hybm" in store
    np.testing.assert_array_equal(store["level"][:], [0, 1, 2])
    np.testing.assert_array_almost_equal(store["hyam"][:], hyam)
    np.testing.assert_array_almost_equal(store["hybm"][:], hybm)


def test_prepare_output_store_attrs(tmp_path):
    grouped = _split_variable_levels(["PRECT"])
    lat, lon = _make_lat_lon()
    times = np.array(["2020-01-01"], dtype="datetime64[s]")

    store = prepare_output_store(
        output_path=str(tmp_path / "out.zarr"),
        grouped_variables=grouped,
        n_times=1,
        n_steps=1,
        nside=_NSIDE,
        tile_size=_TILE,
        time_values=times,
        lat=lat,
        lon=lon,
        attrs={"dataset_type": "test", "pixel_ordering": "cubesphere_faces_2d"},
    )

    assert store.attrs["dataset_type"] == "test"
    assert store.attrs["pixel_ordering"] == "cubesphere_faces_2d"


# ---------------------------------------------------------------------------
# prepare_latlon_plev_store
# ---------------------------------------------------------------------------

_N_LAT = 6
_N_LON = 8
_N_PLEV = 3


def test_prepare_latlon_plev_store_2d_and_3d(tmp_path):
    grouped = _split_variable_levels(["PRECT", "T_0", "T_1", "T_2"])
    lat = np.linspace(-80, 80, _N_LAT, dtype=np.float32)
    lon = np.linspace(0, 360, _N_LON, endpoint=False, dtype=np.float32)
    plev = np.array([100000.0, 50000.0, 20000.0], dtype=np.float32)
    times = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[s]")

    store = prepare_latlon_plev_store(
        output_path=str(tmp_path / "out.zarr"),
        grouped_variables=grouped,
        n_times=_N_TIMES,
        n_steps=_N_STEPS,
        time_values=times,
        lat=lat,
        lon=lon,
        plev=plev,
    )

    assert store["PRECT"].shape == (_N_TIMES, _N_STEPS, _N_LAT, _N_LON)
    assert store["PRECT"].chunks == (1, 1, _N_LAT, _N_LON)
    assert store["T"].shape == (_N_TIMES, _N_STEPS, _N_PLEV, _N_LAT, _N_LON)
    assert store["T"].chunks == (1, 1, 1, _N_LAT, _N_LON)

    np.testing.assert_array_equal(store["lat"][:], lat)
    np.testing.assert_array_equal(store["lon"][:], lon)
    np.testing.assert_array_equal(store["plev"][:], plev)
    assert store["lat"].attrs["units"] == "degrees_north"
    assert store["lon"].attrs["units"] == "degrees_east"
    assert store["plev"].attrs["units"] == "Pa"


def test_prepare_latlon_plev_store_with_dt_and_attrs(tmp_path):
    grouped = _split_variable_levels(["U_0", "U_1", "U_2"])
    lat = np.linspace(-80, 80, _N_LAT, dtype=np.float32)
    lon = np.linspace(0, 360, _N_LON, endpoint=False, dtype=np.float32)
    plev = np.array([100000.0, 50000.0, 20000.0], dtype=np.float32)
    times = np.array(["2020-01-01"], dtype="datetime64[s]")
    dt = np.timedelta64(10 * 60, "s")

    store = prepare_latlon_plev_store(
        output_path=str(tmp_path / "out.zarr"),
        grouped_variables=grouped,
        n_times=1,
        n_steps=_N_STEPS,
        time_values=times,
        lat=lat,
        lon=lon,
        plev=plev,
        dt=dt,
        attrs={"grid": "ace2_latlon", "P0_Pa": 100000.0},
    )

    steps = store["step"][:]
    np.testing.assert_array_equal(steps, np.arange(1, _N_STEPS + 1) * dt)
    assert store.attrs["grid"] == "ace2_latlon"
    assert store.attrs["P0_Pa"] == 100000.0
    # plev axis is present for 3D
    assert store["U"].shape[2] == _N_PLEV
    assert store["U"].chunks == (1, 1, 1, _N_LAT, _N_LON)


# ---------------------------------------------------------------------------
# ZarrWriter
# ---------------------------------------------------------------------------


def test_queue_backpressure():
    """ZarrWriter with maxsize=1 blocks when the queue is full, then drains correctly."""
    grouped = OrderedDict([("V", [(None, 0)])])
    store = _make_store(grouped)
    writer = ZarrWriter(store, grouped, tile_size=TILE, maxsize=1)

    for step in range(N_STEPS):
        writer.write(
            out_index=0, step_index=step, face=0, x0=0, y0=0, tensor=_tile(float(step))
        )
    writer.close()

    for step in range(N_STEPS):
        assert np.all(store["V"][0, step, 0, 0:TILE, 0:TILE] == float(step))
