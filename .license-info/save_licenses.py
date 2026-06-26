#!/usr/bin/env python3
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

import csv
import subprocess
import sys
from importlib import metadata
from pathlib import Path

from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parent.parent
LICENSE_DIR = ROOT / ".license-info"
RAW_CSV = LICENSE_DIR / "third_party_licenses.regen.csv"
FINAL_CSV = LICENSE_DIR / "third_party_licenses.csv"
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"

UNKNOWN_LICENSE_OVERRIDES = {
    "natten": "MIT",
    "google-crc32c": "Apache-2.0",
    "loro": "MIT",
    "matplotlib-inline": "BSD-3-Clause",
    "transformer-engine": "Apache-2.0",
    "cuda-toolkit": "LicenseRef-NVIDIA-SOFTWARE-LICENSE",
}

EXCLUDED_PACKAGE_PREFIXES = (
    "nvidia-",
    "cuda-",
)
EXCLUDED_PACKAGE_NAMES = {
    "nvtx",
}


def run_piplicenses() -> None:
    python = str(VENV_PYTHON if VENV_PYTHON.exists() else sys.executable)
    cmd = [
        python,
        "-m",
        "piplicenses",
        "--format=csv",
        "--with-urls",
        "--python",
        python,
        "--output-file",
        str(RAW_CSV),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)  # noqa: S603


def build_url_map() -> dict[str, str]:
    url_map: dict[str, str] = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if not name:
            continue
        key = canonicalize_name(name)
        md = dist.metadata
        homepage = md.get("Home-page")
        project_urls = md.get_all("Project-URL") or []

        candidates: list[str] = []
        if homepage:
            candidates.append(homepage.strip())

        parsed: list[tuple[str, str]] = []
        for item in project_urls:
            if "," in item:
                label, url = item.split(",", 1)
                parsed.append((label.strip().lower(), url.strip()))
            else:
                parsed.append(("", item.strip()))

        for label in [
            "homepage",
            "home",
            "source",
            "repository",
            "documentation",
            "docs",
            "github",
            "home-page",
        ]:
            for parsed_label, url in parsed:
                if parsed_label == label:
                    candidates.append(url)

        for _, url in parsed:
            candidates.append(url)

        for candidate in candidates:
            if candidate.startswith("http"):
                url_map[key] = candidate
                break
    return url_map


def normalize_csv() -> None:
    rows = list(csv.DictReader(RAW_CSV.open()))
    url_map = build_url_map()
    filtered_rows = []

    for row in rows:
        key = canonicalize_name(row["Name"])

        if key.startswith(EXCLUDED_PACKAGE_PREFIXES) or key in EXCLUDED_PACKAGE_NAMES:
            continue

        if key == "nvidia-dali-cuda130":
            row["Name"] = "nvidia-dali"
            key = "nvidia-dali"

        if row["License"] == "UNKNOWN":
            row["License"] = UNKNOWN_LICENSE_OVERRIDES.get(key, row["License"])

        if not row["URL"] or row["URL"] == "UNKNOWN":
            row["URL"] = url_map.get(
                key, f"https://pypi.org/project/{key}/{row['Version']}/"
            )

        filtered_rows.append(row)

    with FINAL_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Name", "Version", "License", "URL"])
        writer.writeheader()
        writer.writerows(filtered_rows)


def main() -> int:
    run_piplicenses()
    normalize_csv()
    print(FINAL_CSV)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
