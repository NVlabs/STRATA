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
"""
Pure Python generator that mimics TempestRemap's GenerateCSMesh and
GenerateVolumetricMesh and writes Exodus .g files.

Notes:
- This is a direct translation of the TempestRemap logic (cubed-sphere,
  equi-angular edges, unit-sphere normalization).
- Output is Exodus-style NetCDF that matches the fields written in
  Mesh::Write (coord, connect1, global_id1, edge_type1, etc.).
- For large resolutions (e.g., res=1024), memory usage is very high.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
from netCDF4 import Dataset

REFERENCE_TOL = 1.0e-12
LEN_STRING = 33


@dataclass
class Mesh:
    nodes: List[Tuple[float, float, float]]
    faces: List[Tuple[int, int, int, int]]


def _normalize(x: float, y: float, z: float) -> Tuple[float, float, float]:
    r = math.sqrt(x * x + y * y + z * z)
    return (x / r, y / r, z / r)


def _insert_cs_subnode(
    ix0: int,
    ix1: int,
    alpha: float,
    nodes: List[Tuple[float, float, float]],
) -> int:
    x0, y0, z0 = nodes[ix0]
    x1, y1, z1 = nodes[ix1]
    dx = x0 + (x1 - x0) * alpha
    dy = y0 + (y1 - y0) * alpha
    dz = z0 + (z1 - z0) * alpha
    nodes.append(_normalize(dx, dy, dz))
    return len(nodes) - 1


def _generate_cs_multiedge_vertices(
    n_resolution: int,
    n_halo: int,
    ix0: int,
    ix1: int,
    nodes: List[Tuple[float, float, float]],
) -> List[int]:
    edge: List[int] = []
    for i in range(-n_halo, n_resolution + n_halo + 1):
        if i == 0:
            edge.append(ix0)
            continue
        if i == n_resolution:
            edge.append(ix1)
            continue
        alpha = i / n_resolution
        alpha = 0.5 * (math.tan(0.25 * math.pi * (2.0 * alpha - 1.0)) + 1.0)
        edge.append(_insert_cs_subnode(ix0, ix1, alpha, nodes))
    return edge


def _generate_faces_from_quad(
    n_resolution: int,
    n_halo: int,
    edge0: List[int],
    edge1: List[int],
    edge2: List[int],
    edge3: List[int],
    nodes: List[Tuple[float, float, float]],
    faces: List[Tuple[int, int, int, int]],
) -> None:
    n_face_resolution = n_resolution + 2 * n_halo
    edge_bot = _generate_cs_multiedge_vertices(
        n_resolution, n_halo, edge1[0], edge2[0], nodes
    )
    for j in range(n_face_resolution):
        if (j != n_halo - 1) and (j != n_resolution + n_halo - 1):
            ix0 = edge1[j + 1]
            ix1 = edge2[j + 1]
            edge_top = _generate_cs_multiedge_vertices(
                n_resolution, n_halo, ix0, ix1, nodes
            )
        elif j == n_halo - 1:
            edge_top = edge0
        elif j == n_resolution + n_halo - 1:
            edge_top = edge3

        for i in range(n_face_resolution):
            faces.append(  # noqa: PERF401
                (edge_bot[i + 1], edge_top[i + 1], edge_top[i], edge_bot[i])
            )
        edge_bot = edge_top


def generate_cs_mesh(n_resolution: int, n_halo: int = 0) -> Mesh:
    nodes: List[Tuple[float, float, float]] = []
    faces: List[Tuple[int, int, int, int]] = []

    inv = 1.0 / math.sqrt(3.0)
    nodes.extend(
        [
            (inv, -inv, -inv),
            (inv, inv, -inv),
            (-inv, inv, -inv),
            (-inv, -inv, -inv),
            (inv, -inv, inv),
            (inv, inv, inv),
            (-inv, inv, inv),
            (-inv, -inv, inv),
        ]
    )

    edges = [None] * 12
    edges[0] = _generate_cs_multiedge_vertices(n_resolution, n_halo, 0, 1, nodes)
    edges[1] = _generate_cs_multiedge_vertices(n_resolution, n_halo, 1, 2, nodes)
    edges[2] = _generate_cs_multiedge_vertices(n_resolution, n_halo, 2, 3, nodes)
    edges[3] = _generate_cs_multiedge_vertices(n_resolution, n_halo, 3, 0, nodes)

    edges[4] = _generate_cs_multiedge_vertices(n_resolution, n_halo, 0, 4, nodes)
    edges[5] = _generate_cs_multiedge_vertices(n_resolution, n_halo, 1, 5, nodes)
    edges[6] = _generate_cs_multiedge_vertices(n_resolution, n_halo, 2, 6, nodes)
    edges[7] = _generate_cs_multiedge_vertices(n_resolution, n_halo, 3, 7, nodes)

    edges[8] = _generate_cs_multiedge_vertices(n_resolution, n_halo, 4, 5, nodes)
    edges[9] = _generate_cs_multiedge_vertices(n_resolution, n_halo, 5, 6, nodes)
    edges[10] = _generate_cs_multiedge_vertices(n_resolution, n_halo, 6, 7, nodes)
    edges[11] = _generate_cs_multiedge_vertices(n_resolution, n_halo, 7, 4, nodes)

    _generate_faces_from_quad(
        n_resolution, n_halo, edges[0], edges[4], edges[5], edges[8], nodes, faces
    )
    _generate_faces_from_quad(
        n_resolution, n_halo, edges[1], edges[5], edges[6], edges[9], nodes, faces
    )
    _generate_faces_from_quad(
        n_resolution, n_halo, edges[2], edges[6], edges[7], edges[10], nodes, faces
    )
    _generate_faces_from_quad(
        n_resolution, n_halo, edges[3], edges[7], edges[4], edges[11], nodes, faces
    )

    _generate_faces_from_quad(
        n_resolution,
        n_halo,
        list(reversed(edges[2])),
        edges[3],
        list(reversed(edges[1])),
        edges[0],
        nodes,
        faces,
    )

    _generate_faces_from_quad(
        n_resolution,
        n_halo,
        edges[8],
        list(reversed(edges[11])),
        edges[9],
        list(reversed(edges[10])),
        nodes,
        faces,
    )

    faces = [(f[3], f[0], f[1], f[2]) for f in faces]

    return Mesh(nodes=nodes, faces=faces)


def _interpolate_quadrilateral_node(
    node0: Tuple[float, float, float],
    node1: Tuple[float, float, float],
    node2: Tuple[float, float, float],
    node3: Tuple[float, float, float],
    d_a: float,
    d_b: float,
) -> Tuple[float, float, float]:
    x = (
        (1.0 - d_a) * (1.0 - d_b) * node0[0]
        + d_a * (1.0 - d_b) * node1[0]
        + d_a * d_b * node2[0]
        + (1.0 - d_a) * d_b * node3[0]
    )
    y = (
        (1.0 - d_a) * (1.0 - d_b) * node0[1]
        + d_a * (1.0 - d_b) * node1[1]
        + d_a * d_b * node2[1]
        + (1.0 - d_a) * d_b * node3[1]
    )
    z = (
        (1.0 - d_a) * (1.0 - d_b) * node0[2]
        + d_a * (1.0 - d_b) * node1[2]
        + d_a * d_b * node2[2]
        + (1.0 - d_a) * d_b * node3[2]
    )
    return _normalize(x, y, z)


def _accumulated_weights(n_p: int, uniform: bool) -> List[float]:
    if uniform:
        d_w = [1.0 / n_p] * n_p
    else:
        raise ValueError("Non-uniform weights not implemented")
    d_acc = [0.0]
    for w in d_w:
        d_acc.append(d_acc[-1] + w)
    return d_acc


def generate_volumetric_mesh(mesh_in: Mesh, n_p: int, uniform: bool = True) -> Mesh:
    if n_p < 2:
        raise ValueError("--np must be >= 2")

    d_acc = _accumulated_weights(n_p, uniform)

    nodes_out: List[Tuple[float, float, float]] = []
    faces_out: List[Tuple[int, int, int, int]] = []

    node_map = {}

    def _key(node: Tuple[float, float, float]) -> Tuple[float, float, float]:
        return (round(node[0], 12), round(node[1], 12), round(node[2], 12))

    for face in mesh_in.faces:
        node0 = mesh_in.nodes[face[0]]
        node1 = mesh_in.nodes[face[1]]
        node2 = mesh_in.nodes[face[2]]
        node3 = mesh_in.nodes[face[3]]

        for q in range(n_p):
            for p in range(n_p):
                face_nodes = []
                for i in range(4):
                    px = p + ((i + 1) // 2) % 2
                    qx = q + (i // 2)
                    node = _interpolate_quadrilateral_node(
                        node0,
                        node1,
                        node2,
                        node3,
                        d_acc[px],
                        d_acc[qx],
                    )
                    k = _key(node)
                    ix = node_map.get(k)
                    if ix is None:
                        ix = len(nodes_out)
                        node_map[k] = ix
                        nodes_out.append(node)
                    face_nodes.append(ix)
                faces_out.append(tuple(face_nodes))

    return Mesh(nodes=nodes_out, faces=faces_out)


def _write_string_array(var, strings: Iterable[str], length: int) -> None:
    # Build the char array manually: newer netCDF4/numpy combinations reject
    # bytes arrays in stringtochar (numpy.bytes_ lost .encode).
    items = list(strings)
    arr = np.zeros((len(items), length), dtype="S1")
    for i, text in enumerate(items):
        raw = text.encode("ascii")[:length]
        arr[i, : len(raw)] = np.frombuffer(raw, dtype="S1")
    var[:] = arr


def write_exodus(path: str, mesh: Mesh) -> None:
    n_nodes = len(mesh.nodes)
    n_faces = len(mesh.faces)

    with Dataset(path, "w") as ds:
        ds.createDimension("len_string", LEN_STRING)
        ds.createDimension("len_line", 81)
        ds.createDimension("four", 4)
        ds.createDimension("time_step", None)
        ds.createDimension("num_dim", 3)
        ds.createDimension("num_nodes", n_nodes)
        ds.createDimension("num_elem", n_faces)
        ds.createDimension("num_qa_rec", 1)
        ds.createDimension("num_el_blk", 1)
        ds.createDimension("num_el_in_blk1", n_faces)
        ds.createDimension("num_nod_per_el1", 4)
        ds.createDimension("num_att_in_blk1", 1)

        ds.api_version = np.float32(5.00)
        ds.version = np.float32(5.00)
        ds.floating_point_word_size = 8
        ds.file_size = 0

        now = time.localtime()
        date_str = time.strftime("%m/%d/%Y", now)
        time_str = time.strftime("%X", now)
        ds.title = f"tempest({path}) {date_str}: {time_str}"

        ds.createVariable("time_whole", "f8", ("time_step",))

        qa = ds.createVariable("qa_records", "S1", ("num_qa_rec", "four", "len_string"))
        _write_string_array(qa, ["Tempest", "14.0", date_str, time_str], LEN_STRING)

        coor = ds.createVariable("coor_names", "S1", ("num_dim", "len_string"))
        _write_string_array(coor, ["x", "y", "z"], LEN_STRING)

        eb_names = ds.createVariable("eb_names", "S1", ("num_el_blk", "len_string"))
        _write_string_array(eb_names, ["block1"], LEN_STRING)

        eb_status = ds.createVariable("eb_status", "i4", ("num_el_blk",))
        eb_status[:] = np.array([1], dtype=np.int32)

        eb_prop1 = ds.createVariable("eb_prop1", "i4", ("num_el_blk",))
        eb_prop1[:] = np.array([1], dtype=np.int32)
        eb_prop1.setncattr("name", "ID")

        attrib = ds.createVariable(
            "attrib1", "f8", ("num_el_in_blk1", "num_att_in_blk1")
        )
        attrib[:, 0] = 1.0

        connect = ds.createVariable(
            "connect1", "i4", ("num_el_in_blk1", "num_nod_per_el1")
        )
        connect.elem_type = "SHELL4"
        connect[:] = np.array(mesh.faces, dtype=np.int32) + 1

        global_id = ds.createVariable("global_id1", "i4", ("num_el_in_blk1",))
        global_id[:] = np.arange(1, n_faces + 1, dtype=np.int32)

        edge_type = ds.createVariable(
            "edge_type1", "i4", ("num_el_in_blk1", "num_nod_per_el1")
        )
        edge_type[:] = 0

        coord = ds.createVariable("coord", "f8", ("num_dim", "num_nodes"))
        nodes = np.array(mesh.nodes, dtype=np.float64)
        coord[0, :] = nodes[:, 0]
        coord[1, :] = nodes[:, 1]
        coord[2, :] = nodes[:, 2]


def build_mesh_stem(resolution: int, halo: int, np_subdiv: int | None = None) -> str:
    stem = f"ne{resolution}"
    if halo > 0:
        stem += f"halo{halo}"
    if np_subdiv is not None:
        stem += f"pg{np_subdiv}"
    return stem


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate TempestRemap-style Exodus meshes in pure Python."
    )
    parser.add_argument(
        "--res", type=int, default=1024, help="Cubed-sphere resolution (e.g., 1024)"
    )
    parser.add_argument(
        "--np", type=int, default=2, help="Subdivision parameter for volumetric mesh"
    )
    parser.add_argument(
        "--halo", type=int, default=0, help="Number of halo layers per face"
    )
    parser.add_argument("--out-dir", default=".", help="Output directory")
    args = parser.parse_args()

    cs_mesh = generate_cs_mesh(args.res, args.halo)
    out_cs = f"{args.out_dir}/{build_mesh_stem(args.res, args.halo)}.g"
    write_exodus(out_cs, cs_mesh)

    pg_mesh = generate_volumetric_mesh(cs_mesh, args.np, uniform=True)
    out_pg = f"{args.out_dir}/{build_mesh_stem(args.res, args.halo, args.np)}.g"
    write_exodus(out_pg, pg_mesh)

    print("Wrote:", out_cs)
    print("Wrote:", out_pg)


if __name__ == "__main__":
    main()
