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
from __future__ import annotations

import numpy as np
import pytest
import torch
import zarr

from initial_conditions import fast_ic as F
from initial_conditions.layout import NLEV, NLEV_SUBSET, VARS_3D
from initial_conditions.run_hiccup_eamxx import GRAV, KAPPA, RD

P0 = 1.0e5
P_SOURCE = torch.tensor(
    [100.0, 500.0, 1000.0, 5000.0, 1.0e4, 3.0e4, 5.0e4, 7.0e4, 8.5e4, 1.0e5],
    dtype=torch.float64,
)
LOG_P_SOURCE = torch.log(P_SOURCE)


def _interp(p: torch.Tensor, f: torch.Tensor, lapse: bool) -> torch.Tensor:
    stencil = F.log_pressure_stencil(p, LOG_P_SOURCE)
    return F.interp_level(stencil, f, P_SOURCE, LOG_P_SOURCE, lapse)


def test_interp_reproduces_fields_linear_in_log_pressure():
    rng = np.random.default_rng(0)
    ncol = 64
    slope = torch.from_numpy(rng.uniform(-5.0, 5.0, ncol))
    f = (300.0 + slope[None, :] * LOG_P_SOURCE[:, None]).float()
    p = torch.from_numpy(rng.uniform(150.0, 9.9e4, ncol))

    got = _interp(p, f, lapse=False).double()

    expected = 300.0 + slope * torch.log(p)
    torch.testing.assert_close(got, expected, rtol=2e-6, atol=0.0)


def test_interp_clamps_outside_the_source_range_and_lapses_temperature_down():
    f = torch.linspace(200.0, 290.0, P_SOURCE.numel(), dtype=torch.float32)[:, None]
    f = f.expand(-1, 2).contiguous()
    p = torch.tensor([50.0, 1.2e5], dtype=torch.float64)

    clamped = _interp(p, f, lapse=False)
    lapsed = _interp(p, f, lapse=True)

    assert clamped.tolist() == [200.0, 290.0]
    assert lapsed[0].item() == 200.0
    # 6.5 K per 100 hPa below the lowest source level
    assert abs(lapsed[1].item() - (290.0 + 13.0)) < 1e-4


def test_pure_pressure_stencil_matches_per_column_stencil():
    rng = np.random.default_rng(1)
    f = torch.from_numpy(rng.uniform(-50.0, 50.0, (P_SOURCE.numel(), 32))).float()
    scalar = torch.tensor([23456.7], dtype=torch.float64)
    per_column = scalar.expand(32).contiguous()

    for lapse in (False, True):
        assert torch.equal(_interp(scalar, f, lapse), _interp(per_column, f, lapse))


def test_hybrid_pressure_is_one_element_on_pure_pressure_levels():
    ps = torch.tensor([9.0e4, 1.0e5], dtype=torch.float64)

    assert F.hybrid_pressure(0.25, 0.0, P0, ps).tolist() == [0.25 * P0]
    assert F.hybrid_pressure(0.1, 0.5, P0, ps).tolist() == [5.5e4, 6.0e4]


def test_surface_pressure_adjustment_follows_the_standard_atmosphere():
    # ICAO-style atmosphere with HICCUP's lapse rate: T(z) = T0 - L z and
    # p(z) = p_ref (T(z)/T0)^(g/(R L)). With T known at the source surface,
    # HICCUP's adjustment is exact for this atmosphere.
    lapse, r, g = F.STD_LAPSE, F.RD_AIR, F.GRAVIT
    p_ref = 101325.0
    t0 = float(np.float32(288.15))
    phis_dst = torch.tensor(
        [-200.0 * g, 0.0, 5.0e-4, 800.0 * g, 3000.0 * g], dtype=torch.float32
    )
    n = phis_dst.numel()

    ps = F.adjust_surface_pressure(
        torch.full((n,), p_ref, dtype=torch.float32),
        torch.zeros(n, dtype=torch.float32),
        torch.full((n,), t0, dtype=torch.float32),
        phis_dst,
        p_ref,
    )

    z_dst = phis_dst.double() / g
    expected = p_ref * ((t0 - lapse * z_dst) / t0) ** (g / (r * lapse))
    # |del_phis| within the 1e-3 threshold leaves ps untouched
    expected[1:3] = p_ref
    torch.testing.assert_close(ps, expected, rtol=1e-7, atol=0.0)


def test_supersaturation_clips_to_bolton_saturation():
    p_hpa = torch.tensor([1000.0], dtype=torch.float64)
    t = torch.full((4,), 273.0, dtype=torch.float32)
    # at 273.0 K Bolton's exponent is zero, so es = 6.112 hPa
    eps = 18.0 / 28.966
    r_sat = eps * 6.112 / (1000.0 - 6.112)
    qv_sat = r_sat / (1.0 + r_sat)
    qv = torch.tensor([0.5 * qv_sat, qv_sat * 0.999, 2.0 * qv_sat, -1e-3]).float()

    out = F.remove_supersaturation(qv, t, p_hpa)

    assert out.dtype == torch.float64
    assert out[0].item() == qv[0].item()
    assert out[1].item() == qv[1].item()
    assert abs(out[2].item() - qv_sat) < 1e-6 * qv_sat
    assert out[3].item() == F.QV_MIN


def _random_row_stochastic_csr(n_b: int, n_a: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    counts = rng.integers(1, 4, n_b)
    crow = np.concatenate([[0], np.cumsum(counts)])
    col = np.concatenate([np.sort(rng.choice(n_a, c, replace=False)) for c in counts])
    val = rng.uniform(0.1, 1.0, crow[-1])
    for i in range(n_b):
        val[crow[i] : crow[i + 1]] /= val[crow[i] : crow[i + 1]].sum()
    return torch.sparse_csr_tensor(
        torch.from_numpy(crow),
        torch.from_numpy(col),
        torch.from_numpy(val),
        size=(n_b, n_a),
    )


def test_horizontal_remap_preserves_constants_and_ignores_chunking():
    remap = _random_row_stochastic_csr(n_b=50, n_a=20, seed=2)
    rng = np.random.default_rng(3)
    src = torch.from_numpy(rng.uniform(-1.0, 1.0, (20, 4))).float()
    src[:, 0] = 271.5

    whole = F.remap_horizontal(remap, src, rows_per_chunk=1000)
    rowwise = F.remap_horizontal(remap, src, rows_per_chunk=1)

    assert whole.shape == (4, 50)
    assert torch.equal(whole, rowwise)
    torch.testing.assert_close(whole[0], torch.full((50,), 271.5), rtol=0, atol=1e-4)


def _toy_vert() -> dict[str, np.ndarray]:
    """128 levels like L128: 46 pure pressure levels on top, hybrid below."""
    eta = np.linspace(0.003, 1.0, NLEV + 1)
    s = np.clip((eta - eta[46]) / (1.0 - eta[46]), 0.0, 1.0)
    hyai = np.where(np.arange(NLEV + 1) <= 46, 0.5 * eta, (1.0 - s) * 0.5 * eta[46])
    hybi = s
    return {
        "hyai": hyai,
        "hybi": hybi,
        "hyam": 0.5 * (hyai[:-1] + hyai[1:]),
        "hybm": 0.5 * (hybi[:-1] + hybi[1:]),
    }


def test_isothermal_atmosphere_at_rest_gives_closed_form_ic():
    t_iso = 250.0
    ps = np.array([9.5e4, 9.0e4, 8.0e4, 6.0e4, 9.9e4], dtype=np.float32)
    phis = (np.array([0.0, 500.0, 1500.0, 3500.0, 50.0]) * GRAV).astype(np.float32)
    ncol = ps.size
    nk = P_SOURCE.numel()

    def level_field(value: float) -> np.ndarray:
        return np.full((nk, 1, ncol), value, dtype=np.float32)

    era5 = F.Era5(
        atm={
            "T_mid": level_field(t_iso),
            "qv": level_field(0.0),
            "U": level_field(7.0),
            "V": level_field(-3.0),
            "omega": level_field(0.25),
        },
        sfc={
            "ps": ps.reshape(1, ncol),
            "phis": phis.reshape(1, ncol),
            "T_2m": np.full((1, ncol), 281.0, dtype=np.float32),
        },
        plev_pa=P_SOURCE.numpy(),
    )
    identity = torch.sparse_csr_tensor(
        torch.arange(ncol + 1),
        torch.arange(ncol),
        torch.ones(ncol, dtype=torch.float64),
        size=(ncol, ncol),
    )
    vert = _toy_vert()
    emitted: dict[tuple[str, int], np.ndarray] = {}

    def emit(name: str, j: int, field: torch.Tensor) -> None:
        assert (name, j) not in emitted
        emitted[name, j] = field.numpy()

    F.build_ic(era5, identity, torch.from_numpy(phis), vert, P0, 2, emit)

    assert len(emitted) == len(VARS_3D) * NLEV_SUBSET + 2
    assert np.array_equal(emitted["ps", 0], ps)
    assert np.array_equal(emitted["T_2m", 0], np.full(ncol, 281.0, np.float32))
    ps64 = ps.astype(np.float64)
    for j in range(NLEV_SUBSET):
        k = 3 + 4 * j
        p_mid = vert["hyam"][k] * P0 + vert["hybm"][k] * ps64
        assert np.all(emitted["U", j] == 7.0)
        assert np.all(emitted["V", j] == -3.0)
        assert np.all(emitted["omega", j] == 0.25)
        assert np.all(emitted["qv", j] == 0.0)
        np.testing.assert_allclose(
            emitted["PotentialTemperature", j],
            t_iso * (P0 / p_mid) ** KAPPA,
            rtol=1e-6,
        )
        # isothermal and dry: the interface sums telescope to the hypsometric
        # height of the mid level above the surface
        np.testing.assert_allclose(
            emitted["z_mid", j],
            phis / GRAV + (RD * t_iso / GRAV) * np.log(ps64 / p_mid),
            rtol=1e-6,
        )


def test_chunk_writer_round_trips_every_chunk_through_zarr(tmp_path):
    ncol = 10
    rng = np.random.default_rng(4)
    expected = {}
    out = str(tmp_path / "ic.zarr")
    with F.ThreadPoolExecutor(4) as pool:
        writer = F.ChunkWriter(out, ncol, "2016-03-20T12:00:00", "test", "h", pool)
        for name in ("phis", "lat", "lon", "sgh30", "landfrac", "T_2m", "ps", "coszr"):
            expected[name] = rng.uniform(-1.0, 1.0, ncol).astype(np.float32)
            writer.submit(name, 0, expected[name])
        for name in VARS_3D:
            expected[name] = rng.uniform(-1.0, 1.0, (NLEV_SUBSET, ncol)).astype(
                np.float32
            )
            for j in range(NLEV_SUBSET):
                writer.submit(name, j, torch.from_numpy(expected[name][j]))
        writer.close()

    root = zarr.open_consolidated(out, mode="r")
    for name, values in expected.items():
        got = np.asarray(root[name][...])
        assert np.array_equal(got.reshape(values.shape), values), name
    assert root.attrs["initial_time"] == "2016-03-20T12:00:00"


def test_surface_pressure_adjustment_extrapolates_t_below_the_lowest_level():
    # Same atmosphere, but the source surface (p_ref) is below the lowest
    # source level p_bot, so T at the surface comes from T(p_bot). HICCUP
    # linearises T(p) = T(p_bot) (p/p_bot)^alpha, which is exact to ~1e-5 here.
    lapse, r, g = F.STD_LAPSE, F.RD_AIR, F.GRAVIT
    alpha = lapse * r / g
    p_ref, p_bot, t0 = 101325.0, 1.0e5, 288.15
    t_bot = t0 * (p_bot / p_ref) ** alpha
    phis_dst = torch.tensor(
        [-150.0 * g, 800.0 * g, 1500.0 * g, 3000.0 * g], dtype=torch.float32
    )
    n = phis_dst.numel()

    ps = F.adjust_surface_pressure(
        torch.full((n,), p_ref, dtype=torch.float32),
        torch.zeros(n, dtype=torch.float32),
        torch.full((n,), t_bot, dtype=torch.float32),
        phis_dst,
        p_bot,
    )

    z_dst = phis_dst.double() / g
    expected = p_ref * ((t0 - lapse * z_dst) / t0) ** (g / (r * lapse))
    torch.testing.assert_close(ps, expected, rtol=2e-5, atol=0.0)


def _identity_map(ncol: int) -> torch.Tensor:
    return torch.sparse_csr_tensor(
        torch.arange(ncol + 1),
        torch.arange(ncol),
        torch.ones(ncol, dtype=torch.float64),
        size=(ncol, ncol),
    )


def _run_build_ic(era5: F.Era5, phis_dst: np.ndarray) -> dict:
    emitted: dict = {}

    def emit(name: str, j: int, field: torch.Tensor) -> None:
        emitted[name, j] = field.numpy()

    ncol = phis_dst.size
    F.build_ic(
        era5, _identity_map(ncol), torch.from_numpy(phis_dst), _toy_vert(), P0, 2, emit
    )
    return emitted


def test_build_ic_moves_ps_to_the_scream_surface_in_a_standard_atmosphere():
    # ERA5 surface at sea level (p_ref) in the standard atmosphere; SCREAM's
    # surface is higher or lower. The adjusted ps must be the standard-atmosphere
    # pressure at the SCREAM height, using T from the lowest source level.
    lapse, r, g = F.STD_LAPSE, F.RD_AIR, F.GRAVIT
    alpha = lapse * r / g
    p_ref, t0 = 101325.0, 288.15
    z_dst = np.array([-100.0, 500.0, 1500.0, 2500.0])
    ncol, nk = z_dst.size, P_SOURCE.numel()
    t_src = t0 * (P_SOURCE.numpy() / p_ref) ** alpha

    def level_field(values: np.ndarray) -> np.ndarray:
        return np.repeat(values[:, None, None], ncol, axis=2).astype(np.float32)

    era5 = F.Era5(
        atm={
            "T_mid": level_field(t_src),
            "qv": level_field(np.zeros(nk)),
            "U": level_field(np.zeros(nk)),
            "V": level_field(np.zeros(nk)),
            "omega": level_field(np.zeros(nk)),
        },
        sfc={
            "ps": np.full((1, ncol), p_ref, np.float32),
            "phis": np.zeros((1, ncol), np.float32),
            "T_2m": np.full((1, ncol), t0, np.float32),
        },
        plev_pa=P_SOURCE.numpy(),
    )

    emitted = _run_build_ic(era5, (z_dst * g).astype(np.float32))

    expected = p_ref * ((t0 - lapse * z_dst) / t0) ** (g / (r * lapse))
    np.testing.assert_allclose(emitted["ps", 0], expected, rtol=2e-5)


def _qsat_murphy_koop(t: float, p_pa: np.ndarray) -> np.ndarray:
    """Specific humidity at liquid saturation, Murphy and Koop (2005) eq. 10."""
    log_es = (
        54.842763
        - 6763.22 / t
        - 4.210 * np.log(t)
        + 0.000367 * t
        + np.tanh(0.0415 * (t - 218.8))
        * (53.878 - 1331.22 / t - 9.44523 * np.log(t) + 0.014025 * t)
    )
    es = np.exp(log_es)
    return 0.622 * es / (p_pa - 0.378 * es)


def test_build_ic_keeps_subsaturated_qv_and_clips_supersaturated_qv():
    t_iso = 250.0
    q0 = 2.0e-4  # below saturation at 250 K on every level from 1000 hPa up
    ps = np.array([9.8e4, 9.0e4, 9.5e4, 8.5e4], dtype=np.float32)
    phis = (np.array([0.0, 700.0, 300.0, 1300.0]) * GRAV).astype(np.float32)
    ncol, nk = ps.size, P_SOURCE.numel()
    moist, wet = [0, 1], [2, 3]

    qv = np.zeros((nk, 1, ncol), np.float32)
    qv[:, 0, moist] = q0
    qv[:, 0, wet] = 3.0 * _qsat_murphy_koop(t_iso, P_SOURCE.numpy())[:, None]

    def level_field(value: float) -> np.ndarray:
        return np.full((nk, 1, ncol), value, dtype=np.float32)

    era5 = F.Era5(
        atm={
            "T_mid": level_field(t_iso),
            "qv": qv,
            "U": level_field(0.0),
            "V": level_field(0.0),
            "omega": level_field(0.0),
        },
        sfc={
            "ps": ps.reshape(1, ncol),
            "phis": phis.reshape(1, ncol),
            "T_2m": np.full((1, ncol), t_iso, np.float32),
        },
        plev_pa=P_SOURCE.numpy(),
    )

    emitted = _run_build_ic(era5, phis)

    vert = _toy_vert()
    ps64 = ps.astype(np.float64)
    tv = t_iso * (1.0 + (461.5 / 287.05 - 1.0) * q0)
    for j in range(NLEV_SUBSET):
        k = 3 + 4 * j
        p_mid = vert["hyam"][k] * P0 + vert["hybm"][k] * ps64
        qv_out = emitted["qv", j].astype(np.float64)
        np.testing.assert_allclose(qv_out[moist], q0, rtol=1e-6)
        # hypsometric height with the virtual temperature of the moist columns
        np.testing.assert_allclose(
            emitted["z_mid", j][moist],
            (phis / GRAV + (RD * tv / GRAV) * np.log(ps64 / p_mid))[moist],
            rtol=1e-6,
        )
        # clipped to saturation: Bolton (in HICCUP's form) is within 2.5% of
        # Murphy-Koop at 250 K; well above es so the formula is meaningful
        high = p_mid[wet] >= 1000.0
        ratio = qv_out[wet] / _qsat_murphy_koop(t_iso, p_mid[wet])
        assert np.all(np.abs(ratio[high] - 1.0) < 0.025), (j, ratio)


def test_load_map_reads_one_based_links_in_any_order(tmp_path):
    import netCDF4

    # dst 0 = src 1; dst 1 = 0.25 src 0 + 0.75 src 3; dst 2 = src 2 (1-based in file)
    links = [(2, 1, 0.25), (1, 2, 1.0), (3, 3, 1.0), (2, 4, 0.75)]
    path = str(tmp_path / "map.nc")
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("n_a", 4)
        ds.createDimension("n_b", 3)
        ds.createDimension("n_s", len(links))
        for name, dtype, column in (("row", "i4", 0), ("col", "i4", 1), ("S", "f8", 2)):
            ds.createVariable(name, dtype, ("n_s",))[:] = [
                link[column] for link in links
            ]

    remap = F.load_map(path, torch.device("cpu"))
    src = torch.tensor([[10.0, -1.0], [20.0, -2.0], [30.0, -3.0], [40.0, -4.0]])
    out = F.remap_horizontal(remap, src, rows_per_chunk=2)

    expected = torch.tensor([[20.0, 32.5, 30.0], [-2.0, -3.25, -3.0]])
    torch.testing.assert_close(out, expected, rtol=0.0, atol=0.0)


ARCO_LAT_NS = np.array([60.0, 30.0, 0.0, -30.0, -60.0])
ARCO_LON = np.arange(8) * 45.0
ARCO_LEVELS = np.array([500.0, 850.0, 1000.0])
ARCO_TIMES = ["2021-07-15T11", "2021-07-15T12"]


def _write_arco_like_store(path, omega_units: str = "Pa s**-1") -> None:
    """A tiny store in the ARCO ERA5 layout (north-to-south, hours since 1900)."""
    root = zarr.open_group(str(path), mode="w", zarr_format=2)
    nt, nlev, nlat, nlon = len(ARCO_TIMES), ARCO_LEVELS.size, 5, 8

    def put(name, data, units="", dims=None):
        arr = root.create_array(name, shape=data.shape, dtype=data.dtype)
        arr[...] = data
        arr.attrs["units"] = units
        arr.attrs["_ARRAY_DIMENSIONS"] = dims or [name]

    epoch = np.datetime64("1900-01-01T00", "h")
    hours = [
        int((np.datetime64(t, "h") - epoch) / np.timedelta64(1, "h"))
        for t in ARCO_TIMES
    ]
    put("time", np.array(hours, np.int64), "hours since 1900-01-01 00:00:00")
    put("latitude", ARCO_LAT_NS)
    put("longitude", ARCO_LON)
    put("level", ARCO_LEVELS)
    dims3 = ["time", "level", "latitude", "longitude"]
    dims2 = ["time", "latitude", "longitude"]
    shape3, shape2 = (nt, nlev, nlat, nlon), (nt, nlat, nlon)
    # temperature differs between the two hours so the time lookup is visible
    temp = np.stack([np.full(shape3[1:], 250.0 + 10.0 * i) for i in range(nt)])
    put("temperature", temp.astype(np.float32), "K", dims3)
    # u carries the latitude, so a flip would be visible
    u = np.broadcast_to(ARCO_LAT_NS[None, None, :, None], shape3)
    put("u_component_of_wind", u.astype(np.float32), "m s**-1", dims3)
    for name in ("v_component_of_wind", "specific_humidity"):
        put(name, np.zeros(shape3, np.float32), "", dims3)
    put("vertical_velocity", np.zeros(shape3, np.float32), omega_units, dims3)
    put("surface_pressure", np.full(shape2, 1.0e5, np.float32), "Pa", dims2)
    put("geopotential_at_surface", np.zeros(shape2, np.float32), "m**2 s**-2", dims2)
    put("2m_temperature", np.full(shape2, 290.0, np.float32), "K", dims2)


def test_fetch_era5_arco_selects_the_hour_and_keeps_north_to_south(tmp_path):
    store = tmp_path / "arco.zarr"
    _write_arco_like_store(store)
    era5 = F.fetch_era5_arco("2021-07-15T12:00:00", str(store), workers=2)

    np.testing.assert_array_equal(era5.plev_pa, ARCO_LEVELS * 100.0)
    np.testing.assert_array_equal(era5.atm["T_mid"], 260.0)
    np.testing.assert_array_equal(era5.atm["U"][:, 0, :], 60.0)
    np.testing.assert_array_equal(era5.atm["U"][:, -1, :], -60.0)
    np.testing.assert_array_equal(era5.sfc["ps"], 1.0e5)

    with pytest.raises(SystemExit, match="not on the hour"):
        F.fetch_era5_arco("2021-07-15T12:30:00", str(store), workers=1)
    with pytest.raises(SystemExit, match="is not in"):
        F.fetch_era5_arco("2021-07-15T13:00:00", str(store), workers=1)
    bad = tmp_path / "arco_hpa.zarr"
    _write_arco_like_store(bad, omega_units="hPa s**-1")
    with pytest.raises(SystemExit, match="vertical velocity units"):
        F.fetch_era5_arco("2021-07-15T12:00:00", str(bad), workers=1)
