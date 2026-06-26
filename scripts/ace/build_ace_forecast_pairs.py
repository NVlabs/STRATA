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
"""Build ACE-grid forecast/truth netCDF from preprocessed zarr inputs.

This stage does not perform horizontal regridding. It reuses:

- an ACE-grid forecast zarr produced from `scripts/ace/run_pipeline.py` / `scripts/ace/regrid_zarr.py`
- an ACE-grid truth/state zarr produced from `scripts/ace/regrid_zarr.py`

It aligns forecast valid times with the truth/state store and writes:

- `scream_input_state`: `(time, scream_channel, lat, lon)` at valid_time - 6h
- `forecast_state`: `(time, scream_channel, lat, lon)`
- `truth_state`: `(time, scream_channel, lat, lon)`
- `time`: forecast valid time as `datetime64[s]`
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor

import netCDF4 as nc
import numpy as np
import zarr

import data_catalog
from screamcast.ace import P0_SCREAM
from screamcast.history import history_entry

REV2_3D_VARIABLE_NAMES = (
    "PotentialTemperature",
    "U",
    "V",
    "z_mid",
    "omega",
    "qv",
)
REV2_SURFACE_VARIABLE_NAMES = ("T_2m", "ps")


def _truth_datetimes(
    values: np.ndarray,
    *,
    start_time: np.datetime64,
    timestep_minutes: int,
) -> np.ndarray:
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype("datetime64[s]")
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"Unsupported truth time dtype {arr.dtype}.")
    return (
        start_time + arr.astype(np.int64) * np.timedelta64(int(timestep_minutes), "m")
    ).astype("datetime64[s]")


def _forecast_step_numbers(
    store: zarr.Group,
    *,
    timestep_minutes: int,
) -> np.ndarray | None:
    if "step" not in store:
        return None

    raw = np.asarray(store["step"]).reshape(-1)
    if raw.size == 0:
        return np.zeros((0,), dtype=np.int64)

    step_delta = np.timedelta64(int(timestep_minutes), "m")
    if np.issubdtype(raw.dtype, np.timedelta64):
        step_numbers = raw / step_delta
    elif np.issubdtype(raw.dtype, np.integer):
        step_numbers = raw.astype(np.float64)
    else:
        raise TypeError(f"Unsupported forecast step dtype {raw.dtype}.")

    rounded = np.rint(step_numbers).astype(np.int64)
    if not np.allclose(step_numbers, rounded):
        raise ValueError(
            "Forecast step coordinate is not an integer multiple of the native timestep: "
            f"{raw.tolist()}"
        )
    return rounded


def _resolve_forecast_step(
    store: zarr.Group,
    *,
    forecast_step: int,
    timestep_minutes: int,
) -> tuple[int, int]:
    n_steps_avail = int(store["PotentialTemperature"].shape[1])
    if forecast_step < 1:
        raise ValueError(f"forecast_step must be >= 1, got {forecast_step}")

    stored_steps = _forecast_step_numbers(store, timestep_minutes=timestep_minutes)
    if stored_steps is not None:
        matches = np.nonzero(stored_steps == int(forecast_step))[0]
        if matches.size:
            return int(matches[0]), int(forecast_step)

    step_index = forecast_step - 1
    if step_index < n_steps_avail:
        return step_index, int(forecast_step)

    available = (
        stored_steps.tolist()
        if stored_steps is not None and stored_steps.size
        else list(range(1, n_steps_avail + 1))
    )
    raise ValueError(
        f"forecast_step={forecast_step} exceeds available step count {n_steps_avail}; "
        f"available physical steps: {available}"
    )


def _read_grid_coord(
    primary: zarr.Group,
    secondary: zarr.Group,
    name: str,
) -> np.ndarray:
    if name in primary:
        return np.asarray(primary[name], dtype=np.float32).reshape(-1)
    if name in secondary:
        return np.asarray(secondary[name], dtype=np.float32).reshape(-1)
    raise KeyError(
        f"Neither forecast nor truth zarr contains required coordinate {name!r}."
    )


def _levels_from_store(store: zarr.Group, field_levels: int) -> np.ndarray:
    if "level" in store:
        levels = np.asarray(store["level"], dtype=np.int64).reshape(-1)
        if levels.shape[0] == field_levels:
            return levels
    return np.arange(field_levels, dtype=np.int64)


def _select_matching_levels(
    field: np.ndarray,
    *,
    target_levels: np.ndarray,
    source_levels: np.ndarray | None,
) -> np.ndarray:
    if field.ndim < 3:
        return field

    target_levels = np.asarray(target_levels, dtype=np.int64).reshape(-1)
    if field.shape[0] == len(target_levels):
        return field

    if source_levels is not None:
        source_levels = np.asarray(source_levels, dtype=np.int64).reshape(-1)
        if source_levels.shape[0] == field.shape[0]:
            src_lookup = {
                int(level): i for i, level in enumerate(source_levels.tolist())
            }
            try:
                return np.stack(
                    [field[src_lookup[int(level)]] for level in target_levels], axis=0
                )
            except KeyError as exc:
                raise ValueError(
                    f"Missing target level {int(exc.args[0])} in source levels {source_levels.tolist()}."
                ) from exc

    if np.max(target_levels, initial=-1) < field.shape[0]:
        return field[target_levels]

    raise ValueError(
        "Unable to align vertical levels: "
        f"field has {field.shape[0]} levels, target levels are {target_levels.tolist()}, "
        f"source levels are {None if source_levels is None else source_levels.tolist()}."
    )


def _stack_sample(
    source: zarr.Group,
    *,
    time_index: int,
    step_index: int | None,
    target_levels: np.ndarray,
    source_levels: np.ndarray | None,
) -> np.ndarray:
    channels: list[np.ndarray] = []
    for name in REV2_3D_VARIABLE_NAMES:
        arr = source[name]
        value = np.asarray(
            arr[time_index] if step_index is None else arr[time_index, step_index],
            dtype=np.float32,
        )
        value = _select_matching_levels(
            value, target_levels=target_levels, source_levels=source_levels
        )
        channels.append(value)

    for name in REV2_SURFACE_VARIABLE_NAMES:
        arr = source[name]
        value = np.asarray(
            arr[time_index] if step_index is None else arr[time_index, step_index],
            dtype=np.float32,
        )
        channels.append(value[None])

    return np.concatenate(channels, axis=0)


def build_ace_forecast_pairs(
    *,
    forecast_data: str,
    truth_data: str,
    output: str,
    time_start: int = 0,
    time_end: int | None = None,
    n_times: int | None = None,
    min_init_time: str = "",
    forecast_step: int = 36,
    truth_start_time: str | None = None,
    truth_timestep_minutes: int = 10,
    prefetch: int = 2,
) -> None:
    history = history_entry()
    if truth_start_time is None:
        truth_start_time = str(data_catalog.scream_sdecadal.reference_time)

    forecast = zarr.open_group(forecast_data, mode="r")
    truth = zarr.open_group(truth_data, mode="r")

    required = [*REV2_3D_VARIABLE_NAMES, *REV2_SURFACE_VARIABLE_NAMES]
    forecast_missing = [name for name in required if name not in forecast]
    truth_missing = [name for name in required if name not in truth]
    if forecast_missing:
        raise KeyError(
            f"Forecast zarr is missing required rev2 variables: {forecast_missing}"
        )
    if truth_missing:
        raise KeyError(
            f"Truth zarr is missing required rev2 variables: {truth_missing}"
        )

    step_index, forecast_step = _resolve_forecast_step(
        forecast,
        forecast_step=int(forecast_step),
        timestep_minutes=truth_timestep_minutes,
    )

    forecast_valid_time = np.asarray(forecast["time"])
    if not np.issubdtype(forecast_valid_time.dtype, np.datetime64):
        raise TypeError(
            "Forecast zarr time coordinate must be datetime64; "
            f"got {forecast_valid_time.dtype}."
        )
    forecast_valid_time = forecast_valid_time.astype("datetime64[s]")
    input_time = forecast_valid_time - np.timedelta64(
        int(forecast_step * truth_timestep_minutes), "m"
    )
    if min_init_time:
        min_init_time_np = np.datetime64(min_init_time, "s")
        keep = input_time >= min_init_time_np
        forecast_valid_time = forecast_valid_time[keep]
        input_time = input_time[keep]
        selected_forecast_indices = np.nonzero(keep)[0]
    else:
        selected_forecast_indices = np.arange(len(forecast_valid_time), dtype=np.int64)

    t_start = max(0, int(time_start))
    if n_times is not None:
        t_end = t_start + int(n_times)
    elif time_end is not None:
        t_end = int(time_end)
    else:
        t_end = len(selected_forecast_indices)
    t_end = min(t_end, len(selected_forecast_indices))
    if t_start >= t_end:
        raise ValueError(f"Empty time selection: start={t_start}, end={t_end}")
    selected_forecast_indices = selected_forecast_indices[t_start:t_end]
    forecast_valid_time = forecast_valid_time[t_start:t_end]
    input_time = input_time[t_start:t_end]

    truth_time = _truth_datetimes(
        np.asarray(truth["time"]),
        start_time=np.datetime64(truth_start_time),
        timestep_minutes=truth_timestep_minutes,
    )
    truth_lookup = {int(t.astype(np.int64)): i for i, t in enumerate(truth_time)}
    missing_valid_times = [
        str(t)
        for t in forecast_valid_time
        if int(t.astype(np.int64)) not in truth_lookup
    ]
    if missing_valid_times:
        preview = ", ".join(missing_valid_times[:5])
        extra = (
            ""
            if len(missing_valid_times) <= 5
            else f" ... (+{len(missing_valid_times) - 5} more)"
        )
        raise KeyError(f"Truth zarr is missing forecast valid times: {preview}{extra}")
    missing_input_times = [
        str(t) for t in input_time if int(t.astype(np.int64)) not in truth_lookup
    ]
    if missing_input_times:
        preview = ", ".join(missing_input_times[:5])
        extra = (
            ""
            if len(missing_input_times) <= 5
            else f" ... (+{len(missing_input_times) - 5} more)"
        )
        raise KeyError(f"Truth zarr is missing required input times: {preview}{extra}")

    levels = np.asarray(forecast["level"], dtype=np.int64).reshape(-1)
    forecast_hyam = np.asarray(forecast["hyam"], dtype=np.float64).reshape(-1)
    forecast_hybm = np.asarray(forecast["hybm"], dtype=np.float64).reshape(-1)

    truth_field_levels = int(np.asarray(truth["PotentialTemperature"][0]).shape[0])
    truth_levels = _levels_from_store(truth, truth_field_levels)
    lat = _read_grid_coord(forecast, truth, "lat")
    lon = _read_grid_coord(forecast, truth, "lon")

    rev2_variable_names = [
        f"{name}_{level}"
        for name in REV2_3D_VARIABLE_NAMES
        for level in levels.tolist()
    ] + list(REV2_SURFACE_VARIABLE_NAMES)
    n_times = len(selected_forecast_indices)
    n_channels = len(rev2_variable_names)

    with nc.Dataset(output, "w") as out:
        out.createDimension("time", n_times)
        out.createDimension("scream_channel", n_channels)
        out.createDimension("lat", len(lat))
        out.createDimension("lon", len(lon))
        out.createDimension("level", len(levels))
        out.history = history
        out.dataset_type = "ace_rev2_training_pairs"
        out.forecast_step = int(forecast_step)
        out.p0_scream = float(P0_SCREAM)
        out.min_init_time = min_init_time
        out.scream_variable_names = ",".join(rev2_variable_names)
        out.forecast_levels = ",".join(str(int(level)) for level in levels.tolist())
        out.truth_data = truth_data
        out.forecast_data = forecast_data

        time_input_var = out.createVariable("time_input", "i8", ("time",))
        time_input_var[:] = input_time.astype(np.int64)
        time_input_var.units = "seconds since 1970-01-01 00:00:00"
        time_input_var.calendar = "proleptic_gregorian"
        time_var = out.createVariable("time", "i8", ("time",))
        time_var[:] = forecast_valid_time.astype(np.int64)
        time_var.units = "seconds since 1970-01-01 00:00:00"
        time_var.calendar = "proleptic_gregorian"
        out.createVariable("lat", "f4", ("lat",))[:] = lat
        out.createVariable("lon", "f4", ("lon",))[:] = lon
        out.createVariable("hyam", "f8", ("level",))[:] = forecast_hyam
        out.createVariable("hybm", "f8", ("level",))[:] = forecast_hybm
        out.createVariable(
            "scream_input_state", "f4", ("time", "scream_channel", "lat", "lon")
        )
        out.createVariable(
            "forecast_state", "f4", ("time", "scream_channel", "lat", "lon")
        )
        out.createVariable(
            "truth_state", "f4", ("time", "scream_channel", "lat", "lon")
        )

        selected_truth_indices = np.asarray(
            [truth_lookup[int(t.astype(np.int64))] for t in forecast_valid_time],
            dtype=np.int64,
        )
        selected_input_indices = np.asarray(
            [truth_lookup[int(t.astype(np.int64))] for t in input_time],
            dtype=np.int64,
        )

        def load_pair(sel_i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            forecast_i = int(selected_forecast_indices[sel_i])
            truth_i = int(selected_truth_indices[sel_i])
            input_i = int(selected_input_indices[sel_i])
            scream_input_state = _stack_sample(
                truth,
                time_index=input_i,
                step_index=None,
                target_levels=levels,
                source_levels=truth_levels,
            )
            forecast_state = _stack_sample(
                forecast,
                time_index=forecast_i,
                step_index=step_index,
                target_levels=levels,
                source_levels=levels,
            )
            truth_state = _stack_sample(
                truth,
                time_index=truth_i,
                step_index=None,
                target_levels=levels,
                source_levels=truth_levels,
            )
            return scream_input_state, forecast_state, truth_state

        with ThreadPoolExecutor(max_workers=max(1, prefetch)) as pool:
            futures: list[Future[tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
            cursor = 0

            def _submit_next() -> None:
                nonlocal cursor
                if cursor < n_times:
                    futures.append(pool.submit(load_pair, cursor))
                    cursor += 1

            for _ in range(min(prefetch, n_times)):
                _submit_next()

            for i in range(n_times):
                scream_input_state, forecast_state, truth_state = futures.pop(
                    0
                ).result()
                _submit_next()
                print(
                    f"pair {i + 1}/{n_times} valid_time={str(forecast_valid_time[i])}",
                    flush=True,
                )
                out["scream_input_state"][i] = scream_input_state
                out["forecast_state"][i] = forecast_state
                out["truth_state"][i] = truth_state

    print(f"Done: {output}")
