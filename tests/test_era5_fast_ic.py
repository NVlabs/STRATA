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
import torch

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
    import zarr

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
