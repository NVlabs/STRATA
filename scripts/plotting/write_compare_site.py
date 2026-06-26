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

import argparse
from pathlib import Path

REGIONS = ("global", "land", "ocean", "tropics", "extratropics")
VARIABLES = ("qv", "omega", "U")
LABELS = (
    ("tile64_h32_b4_retry2", "tile64 h32 b4 retry2"),
    ("tile128_h16_b4", "tile128 h16 b4"),
    ("tile256_h32_b1", "tile256 h32 b1"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a simple static comparison site with site/index.html and site/images/."
    )
    parser.add_argument("--site-dir", required=True, help="Site root directory.")
    return parser.parse_args()


def _image_card(title: str, href: str) -> str:
    return f"""
    <article class="card">
      <h3>{title}</h3>
      <p><a href="{href}">Open image</a></p>
      <img src="{href}" alt="{title}">
    </article>
"""


def _gallery(cards: list[str]) -> str:
    return '<div class="gallery">' + "".join(cards) + "</div>"


def main() -> None:
    args = parse_args()
    site_dir = Path(args.site_dir)
    site_dir.mkdir(parents=True, exist_ok=True)
    images_dir = site_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    precip_cards = [
        _image_card("Global precipitation", "images/precip_global_compare.png"),
        _image_card("Regional precipitation", "images/precip_region_compare.png"),
    ]

    sections: list[str] = [
        """
    <section>
      <h2>Precipitation</h2>
      <p>
        Total precipitation is computed from
        <code>precip_liq_surf_mass_flux + precip_ice_surf_mass_flux</code>
        and converted from <code>m/s</code> to <code>mm/day</code>. The
        precipitation plots include the catalog-derived truth regional averages
        alongside the three rollout configurations.
      </p>
""",
        *precip_cards,
        "    </section>",
    ]

    for variable in VARIABLES:
        region_blocks: list[str] = []
        for region in REGIONS:
            cards = [
                _image_card(
                    label,
                    f"images/{variable}_{region}_{slug}_contours.png",
                )
                for slug, label in LABELS
            ]
            region_blocks.append(
                f"""
      <h3>{region.title()}</h3>
      {_gallery(cards)}
"""
            )
        sections.extend(
            [
                f"""
    <section>
      <h2>{variable.upper()} contours</h2>
      <p>
        Each image is separate now. Lead time is on the x-axis, level index is on
        the y-axis, and the color scale is shared within each region across tile
        configurations for direct comparison.
      </p>
""",
                *region_blocks,
                "    </section>",
            ]
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Tile Comparison Site</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f4f1ea;
      --card: #fffdfa;
      --fg: #1f2933;
      --muted: #52606d;
      --line: #d8dee6;
      --link: #0b5cab;
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      color: var(--fg);
      background:
        radial-gradient(circle at top left, rgba(11, 92, 171, 0.08), transparent 28%),
        linear-gradient(180deg, #f8f5ef 0%, var(--bg) 100%);
      font-family: Georgia, "Times New Roman", serif;
    }}
    main {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 32px 20px 56px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 2.2rem;
    }}
    p {{
      color: var(--muted);
      max-width: 80ch;
    }}
    section {{
      margin-top: 28px;
      padding: 18px;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: rgba(255, 253, 250, 0.92);
      box-shadow: 0 14px 36px rgba(31, 41, 51, 0.08);
    }}
    .card {{
      margin-top: 18px;
      padding-top: 6px;
    }}
    .gallery {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 18px;
      align-items: start;
    }}
    img {{
      display: block;
      width: 100%;
      height: auto;
      border-radius: 10px;
      border: 1px solid var(--line);
      background: white;
    }}
    a {{
      color: var(--link);
    }}
  </style>
</head>
<body>
  <main>
    <h1>Tile Comparison Site</h1>
    <p>
      Comparing truth precipitation averages against the tile64/halo32,
      tile128/halo16, and tile256/halo32 rollout outputs using precomputed
      regional-average NetCDF products.
    </p>
    {''.join(sections)}
  </main>
</body>
</html>
"""
    output = site_dir / "index.html"
    output.write_text(html, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
