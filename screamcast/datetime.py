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
"""Time conversion utilities for SCREAM data handling."""

from datetime import datetime, timezone

import cftime
import numpy as np


def cftime_to_calday(time_obj):
    """
    Convert cftime object to calendar day (Jan 1 = 1.0).
    Handles DatetimeNoLeap objects properly.
    """
    # Create January 1st of the same year
    jan1 = cftime.DatetimeNoLeap(time_obj.year, 1, 1, 0, 0, 0, 0, has_year_zero=True)

    # Calculate days since Jan 1 (this will be 0 for Jan 1)
    days_since_jan1 = (time_obj - jan1).days

    # Add fractional day from hours/minutes/seconds
    fractional_day = (
        time_obj.hour + time_obj.minute / 60.0 + time_obj.second / 3600.0
    ) / 24.0

    # Calendar day (Jan 1 = 1.0)
    calday = days_since_jan1 + 1.0 + fractional_day

    return calday


def as_py_datetime(time: np.datetime64 | str | datetime) -> datetime:
    """Convert numpy datetime64 or string to Python datetime."""
    if isinstance(time, datetime):
        return time
    elif isinstance(time, np.datetime64):
        ts = (time - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).replace(tzinfo=None)
    elif isinstance(time, str):
        return datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
    else:
        raise NotImplementedError()


def as_cftime_no_leap(time):
    """Convert time to cftime.DatetimeNoLeap object."""
    dt = as_py_datetime(time)
    return cftime.DatetimeNoLeap(
        dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, 0, has_year_zero=True
    )


def as_calday_year(time):
    """Convert time to (calday, year) tuple."""
    time_obj = as_cftime_no_leap(time)
    calday = cftime_to_calday(time_obj)
    year = time_obj.year
    return calday, year
