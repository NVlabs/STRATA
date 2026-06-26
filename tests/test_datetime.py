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
"""Tests for screamcast.datetime converters."""

import cftime
import numpy as np

from screamcast.datetime import (
    as_calday_year,
    as_cftime_no_leap,
    as_py_datetime,
    cftime_to_calday,
)


def test_cftime_to_calday():
    """Test conversion from cftime to calendar day."""
    dt = cftime.DatetimeNoLeap(2020, 1, 1, 12, 0, 0, 0, has_year_zero=True)
    assert cftime_to_calday(dt) == 1.5


def test_as_py_datetime_from_string():
    """Test conversion from string to Python datetime."""
    py_dt = as_py_datetime("2020-01-15 12:30:45")
    assert py_dt.year == 2020
    assert py_dt.month == 1
    assert py_dt.day == 15


def test_as_py_datetime_from_datetime64():
    """Test conversion from numpy datetime64 to Python datetime."""
    np_dt = np.datetime64("2020-01-15T12:30:45")
    py_dt = as_py_datetime(np_dt)
    assert py_dt.year == 2020
    assert py_dt.month == 1
    assert py_dt.day == 15


def test_as_cftime_no_leap():
    """Test conversion to cftime.DatetimeNoLeap."""
    cft_dt = as_cftime_no_leap("2020-01-15 12:30:45")
    assert isinstance(cft_dt, cftime.DatetimeNoLeap)
    assert cft_dt.year == 2020


def test_as_calday_year():
    """Test getting calday and year from datetime string."""
    calday, year = as_calday_year("2020-01-02 00:00:00")
    assert calday == 2.0
    assert year == 2020
