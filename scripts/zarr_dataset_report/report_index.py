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

import html
import json
from pathlib import Path
from typing import Protocol


class TaskResultLike(Protocol):
    name: str
    returncode: int
    elapsed: int
    outputs: tuple[Path, ...]
    skipped: bool


def _display_task_name(task_name: str) -> str:
    return task_name.replace("_", " ")


def _display_artifact_name(path: Path) -> str:
    return path.stem.replace("_", " ")


def _section_id(name: str) -> str:
    return name.lower().replace(" ", "-").replace("_", "-")


def _child_report_links(root_out: Path, href: str) -> list[tuple[str, str]]:
    report_dir = root_out / Path(href).parent
    if not report_dir.is_dir():
        return []
    children: list[tuple[str, str]] = []
    for child in sorted(report_dir.iterdir()):
        child_index = child / "index.html"
        if child.is_dir() and child_index.exists():
            rel = child_index.relative_to(root_out)
            children.append((_display_artifact_name(child), str(rel)))
    return children


def write_index_html(
    *,
    root_out: Path,
    output_leaf: str,
    pred_zarr: Path,
    pred_zarr_attrs: dict[str, object],
    truth_zarr: Path,
    hostname: str,
    created_at: str | None,
    tasks: tuple[str, ...],
    workers: int,
    results: list[TaskResultLike],
    failures: list[str],
) -> None:
    result_by_name = {result.name: result for result in results}
    quick_links: list[tuple[str, str, list[tuple[str, str]]]] = []
    sections: list[str] = []

    for task_name in tasks:
        result = result_by_name.get(task_name)
        if result is None:
            continue

        outputs_by_stem: dict[str, dict[str, str]] = {}
        for output in result.outputs:
            rel = output.relative_to(root_out)
            label = _display_artifact_name(rel)
            if (
                rel.suffix == ".html"
                and rel.stem == "index"
                and rel.parent != Path(".")
            ):
                label = _display_artifact_name(rel.parent)
            entry = outputs_by_stem.setdefault(
                str(rel.with_suffix("")),
                {
                    "label": label,
                },
            )
            entry[output.suffix.lstrip(".")] = str(rel)

        html_only_entries = [
            entry
            for entry in outputs_by_stem.values()
            if "html" in entry and "png" not in entry and "pdf" not in entry
        ]
        if html_only_entries and len(html_only_entries) == len(outputs_by_stem):
            quick_links.extend(
                [
                    (
                        entry["label"],
                        entry["html"],
                        _child_report_links(root_out, entry["html"]),
                    )
                    for entry in html_only_entries
                ]
            )
            continue

        cards: list[str] = []
        for stem in sorted(outputs_by_stem):
            entry = outputs_by_stem[stem]
            png_href = entry.get("png")
            pdf_href = entry.get("pdf")
            html_href = entry.get("html")
            preview = (
                f'<a class="preview" href="{html.escape(png_href)}">'
                f'<img src="{html.escape(png_href)}" '
                f'alt="{html.escape(entry["label"])}"></a>'
                if png_href is not None
                else (
                    f'<a class="preview report" href="{html.escape(html_href)}">'
                    '<div class="preview missing">Open HTML report</div></a>'
                    if html_href is not None
                    else '<div class="preview missing">No PNG preview</div>'
                )
            )
            links: list[str] = []
            if png_href is not None:
                links.append(f'<a href="{html.escape(png_href)}">png</a>')
            if pdf_href is not None:
                links.append(f'<a href="{html.escape(pdf_href)}">pdf</a>')
            if html_href is not None:
                links.append(f'<a href="{html.escape(html_href)}">html</a>')
            cards.append(
                "\n".join(
                    [
                        '<article class="artifact">',
                        preview,
                        f'<div class="artifact-name">{html.escape(entry["label"])}</div>',
                        f'<div class="artifact-links">{" | ".join(links)}</div>',
                        "</article>",
                    ]
                )
            )

        status: list[str] = []
        if result.skipped:
            status.append("cached")
        if result.returncode != 0:
            status.append(f"failed ({result.returncode})")
        elif not result.skipped:
            status.append(f"{result.elapsed}s")
        status_html = (
            f'<span class="task-status">{html.escape(", ".join(status))}</span>'
            if status
            else ""
        )
        section_id = _section_id(task_name)

        sections.append(
            "\n".join(
                [
                    f'<section id="{html.escape(section_id)}">',
                    f"<h2>{html.escape(_display_task_name(task_name))} {status_html}</h2>",
                    '<div class="artifact-grid">',
                    *(
                        cards
                        if cards
                        else ['<p class="empty">No outputs recorded.</p>']
                    ),
                    "</div>",
                    "</section>",
                ]
            )
        )

    failures_html = ""
    if failures:
        failures_html = "\n".join(
            [
                '<div class="failures">',
                "<strong>Failures:</strong>",
                "<ul>",
                *[f"<li>{html.escape(failure)}</li>" for failure in failures],
                "</ul>",
                "</div>",
            ]
        )

    nav_groups: list[str] = ['<a class="nav-home" href="index.html">Main</a>']
    if quick_links:
        nav_groups.append(
            "\n".join(
                [
                    '<details class="nav-group">',
                    "<summary>Regional Average</summary>",
                    '<div class="nav-menu">',
                    *(
                        [
                            f'<a href="{html.escape(href)}">{html.escape(label)}</a>'
                            for label, href, _ in quick_links
                        ]
                        + [
                            f'<a class="nav-subitem" href="{html.escape(child_href)}">{html.escape(child_label)}</a>'
                            for _, _, children in quick_links
                            for child_label, child_href in children
                        ]
                    ),
                    "</div>",
                    "</details>",
                ]
            )
        )
    nav_html = '<nav class="navbar">' + "".join(nav_groups) + "</nav>"

    attr_rows = []
    for key, value in sorted(pred_zarr_attrs.items()):
        rendered_value = (
            value if isinstance(value, str) else json.dumps(value, sort_keys=True)
        )
        attr_rows.append(
            "\n".join(
                [
                    "<tr>",
                    f"<th>{html.escape(str(key))}</th>",
                    f"<td><code>{html.escape(rendered_value)}</code></td>",
                    "</tr>",
                ]
            )
        )

    attrs_html = "\n".join(
        [
            '<section id="zarr-attributes">',
            "<h2>Zarr attributes</h2>",
            (
                '<table class="attrs"><thead><tr><th>Attribute</th><th>Value</th></tr></thead>'
                f"<tbody>{''.join(attr_rows)}</tbody></table>"
                if attr_rows
                else '<p class="empty">No zarr attributes found.</p>'
            ),
            "</section>",
        ]
    )

    page = "\n".join(
        [
            "<!DOCTYPE html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{html.escape(output_leaf)} report</title>",
            "<style>",
            "body { font-family: sans-serif; margin: 24px; background: #f7f7f7; color: #111; }",
            "a { color: #0b57d0; }",
            "header { margin-bottom: 24px; }",
            ".navbar { position: sticky; top: 0; z-index: 10; display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-bottom: 20px; padding: 10px 12px; background: rgba(255,255,255,0.95); border: 1px solid #ddd; border-radius: 10px; backdrop-filter: blur(8px); }",
            ".nav-home, .nav-group summary { display: inline-block; padding: 8px 12px; background: #fff; border: 1px solid #ddd; border-radius: 8px; cursor: pointer; text-decoration: none; color: #111; list-style: none; }",
            ".nav-group { position: relative; }",
            ".nav-group summary::-webkit-details-marker { display: none; }",
            ".nav-group[open] summary { border-bottom-left-radius: 0; border-bottom-right-radius: 0; }",
            ".nav-menu { position: absolute; top: calc(100% - 1px); left: 0; min-width: 220px; padding: 8px; background: #fff; border: 1px solid #ddd; border-radius: 0 10px 10px 10px; box-shadow: 0 8px 24px rgba(0,0,0,0.08); }",
            ".nav-menu a { display: block; padding: 6px 8px; border-radius: 6px; text-decoration: none; }",
            ".nav-menu a:hover { background: #f3f7ff; }",
            ".nav-subitem { padding-left: 18px !important; color: #355b8c; }",
            "h1, h2 { margin-bottom: 8px; }",
            ".meta { color: #444; margin: 6px 0; }",
            ".meta code { background: #eee; padding: 1px 4px; border-radius: 4px; }",
            ".failures { background: #fff1f0; border: 1px solid #f3b3ad; padding: 12px 16px; border-radius: 8px; margin-bottom: 24px; }",
            "section { margin: 28px 0; }",
            ".task-status { font-size: 0.8em; font-weight: normal; color: #666; }",
            ".artifact-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }",
            ".artifact { background: white; border: 1px solid #ddd; border-radius: 10px; padding: 12px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }",
            ".preview { display: block; background: #fafafa; border: 1px solid #e5e5e5; border-radius: 8px; overflow: hidden; aspect-ratio: 4 / 3; }",
            ".preview.report { text-decoration: none; color: inherit; }",
            ".preview img { display: block; width: 100%; height: 100%; object-fit: contain; background: white; }",
            ".preview.missing { display: grid; place-items: center; color: #777; }",
            ".artifact-name { margin-top: 10px; font-weight: 600; }",
            ".artifact-links { margin-top: 6px; color: #555; }",
            ".attrs { width: 100%; border-collapse: collapse; background: white; border: 1px solid #ddd; border-radius: 10px; overflow: hidden; }",
            ".attrs th, .attrs td { padding: 10px 12px; border-bottom: 1px solid #eee; text-align: left; vertical-align: top; }",
            ".attrs th { width: 220px; background: #fafafa; }",
            ".attrs code { white-space: pre-wrap; word-break: break-word; }",
            ".empty { color: #666; }",
            "</style>",
            "</head>",
            "<body>",
            '<header id="top">',
            f"<h1>{html.escape(output_leaf)} report</h1>",
            f'<div class="meta"><strong>Pred zarr:</strong> <code>{html.escape(str(pred_zarr))}</code></div>',
            f'<div class="meta"><strong>Truth zarr:</strong> <code>{html.escape(str(truth_zarr))}</code></div>',
            f'<div class="meta"><strong>Output root:</strong> <code>{html.escape(str(root_out))}</code></div>',
            f'<div class="meta"><strong>Hostname:</strong> <code>{html.escape(hostname)}</code></div>',
            (
                f'<div class="meta"><strong>Date created:</strong> <code>{html.escape(created_at)}</code></div>'
                if created_at is not None
                else ""
            ),
            '<div class="meta"><a href="metadata.json">metadata.json</a></div>',
            "</header>",
            nav_html,
            failures_html,
            attrs_html,
            *(sections if sections else ["<p>No task results available.</p>"]),
            "</body>",
            "</html>",
        ]
    )
    (root_out / "index.html").write_text(page + "\n", encoding="utf-8")
