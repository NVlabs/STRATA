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

from datetime import datetime

import numpy as np

from screamcast.datetime import as_calday_year


class SolarZenithCalculator:
    """
    Calculate solar zenith angles consistent with SCREAM/CESM orbital calculations.
    Replicates the shr_orb_* Fortran routines used in SCREAM.
    """

    def __init__(
        self,
        orbital_year=None,
        orbital_eccen=None,
        orbital_obliq=None,
        orbital_mvelp=None,
    ):
        self.orbital_year = orbital_year
        self.orbital_eccen = orbital_eccen
        self.orbital_obliq = orbital_obliq
        self.orbital_mvelp = orbital_mvelp

    def orbital_params(self, year):
        """Compute orbital parameters for given year."""
        if (
            self.orbital_eccen is not None
            and self.orbital_eccen >= 0
            and self.orbital_obliq is not None
            and self.orbital_obliq >= 0
            and self.orbital_mvelp is not None
            and self.orbital_mvelp >= 0
        ):
            eccen = self.orbital_eccen
            obliq = self.orbital_obliq
            mvelp = self.orbital_mvelp
        else:
            if self.orbital_year is not None and self.orbital_year >= 0:
                year = self.orbital_year

            years_since_1950 = year - 1950.0
            eccen = 0.0167086 - 0.0000004 * years_since_1950
            obliq = 23.439291 - 0.0130042 * years_since_1950 / 100.0
            mvelp = 282.94719 + 1.7195269 * years_since_1950 / 100.0
            mvelp = mvelp % 360.0

        obliqr = np.radians(obliq)
        mvelpp = np.radians(mvelp)
        lambm0 = mvelpp

        return eccen, obliqr, lambm0, mvelpp

    def solar_declination(self, calday, eccen, mvelpp, lambm0, obliqr):
        """Compute solar declination and eccentricity factor."""
        mean_longitude = 2 * np.pi * (calday - 81.0) / 365.25
        mean_anomaly = mean_longitude - lambm0
        equation_of_center = 2 * eccen * np.sin(mean_anomaly) + (
            5 / 4
        ) * eccen**2 * np.sin(2 * mean_anomaly)
        true_longitude = mean_longitude + equation_of_center
        delta = np.arcsin(np.sin(obliqr) * np.sin(true_longitude))

        true_anomaly = mean_anomaly + equation_of_center
        eccf = ((1 + eccen * np.cos(true_anomaly)) / (1 - eccen**2)) ** 2

        alpha = np.arctan2(
            np.cos(obliqr) * np.sin(true_longitude), np.cos(true_longitude)
        )
        eot = (mean_longitude - alpha + np.pi) % (2 * np.pi) - np.pi

        return delta, eccf, eot

    def solar_zenith_cosine(self, calday, lat, lon, delta, dt_avg=0.0, eot=0.0):
        """Compute cosine of solar zenith angle."""
        hours_since_midnight = ((calday - 1) % 1) * 24.0
        solar_time_hours = hours_since_midnight + np.degrees(lon) / 15.0
        hour_angle = np.radians(15.0 * (solar_time_hours - 12.0)) + eot

        if dt_avg > 0:
            dt_hours = dt_avg / 3600.0
            dt_radians = np.radians(15.0 * dt_hours)
            h1 = hour_angle - dt_radians / 2
            h2 = hour_angle + dt_radians / 2

            sin_lat_sin_delta = np.sin(lat) * np.sin(delta)
            cos_lat_cos_delta = np.cos(lat) * np.cos(delta)
            integral = sin_lat_sin_delta * (h2 - h1) + cos_lat_cos_delta * (
                np.sin(h2) - np.sin(h1)
            )
            cosz = integral / (h2 - h1)
        else:
            cosz = np.sin(lat) * np.sin(delta) + np.cos(lat) * np.cos(delta) * np.cos(
                hour_angle
            )

        return np.maximum(cosz, 0.0)


def calculate_cosine_zenith_direct(
    time: str | datetime | np.datetime64,
    lat,
    lon,
    orbital_params=None,
    dt_avg=0.0,
    correct_eot: bool = False,
):
    """Calculate cosine of solar zenith angle for direct inputs."""
    calday, year = as_calday_year(time)

    if orbital_params is None:
        calc = SolarZenithCalculator()
    else:
        calc = SolarZenithCalculator(**orbital_params)

    lats_rad = np.radians(lat)
    lons_rad = np.radians(lon)
    eccen, obliqr, lambm0, mvelpp = calc.orbital_params(year)
    delta, _eccf, eot = calc.solar_declination(calday, eccen, mvelpp, lambm0, obliqr)

    if not correct_eot:
        eot = 0.0

    return calc.solar_zenith_cosine(calday, lats_rad, lons_rad, delta, dt_avg, eot)


def calculate_dswrftoa(
    time,
    lat: np.ndarray,
    lon: np.ndarray,
    S0: float = 1361.0,
    correct_eot: bool = False,
) -> np.ndarray:
    """Approximate DSWRFtoa as the daytime-weighted 6-hour average over [t-6h, t]."""
    calday, year = as_calday_year(time)
    calc = SolarZenithCalculator()
    eccen, obliqr, lambm0, mvelpp = calc.orbital_params(year)
    delta, eccf, eot = calc.solar_declination(calday, eccen, mvelpp, lambm0, obliqr)

    lats_rad = np.radians(lat)
    lons_rad = np.radians(lon)

    hours_since_midnight = ((calday - 1) % 1) * 24.0
    solar_time = hours_since_midnight + np.degrees(lons_rad) / 15.0
    h_center = np.radians(15.0 * (solar_time - 12.0)) - np.radians(15.0 * 3.0)
    if correct_eot:
        h_center = h_center + eot
    h_center = (h_center + np.pi) % (2 * np.pi) - np.pi

    dt_rad = np.radians(15.0 * 6.0)
    h1 = h_center - dt_rad / 2
    h2 = h_center + dt_rad / 2

    sin_ll = np.sin(lats_rad) * np.sin(delta)
    cos_ll = np.cos(lats_rad) * np.cos(delta)
    cos_h0 = np.clip(-np.tan(lats_rad) * np.tan(delta), -1, 1)
    h0 = np.arccos(cos_h0)
    ha = np.maximum(h1, -h0)
    hb = np.minimum(h2, h0)

    integral = np.where(
        ha < hb, sin_ll * (hb - ha) + cos_ll * (np.sin(hb) - np.sin(ha)), 0.0
    )
    cosz_avg = np.maximum(integral / (h2 - h1), 0.0)

    return cosz_avg * eccf * S0
