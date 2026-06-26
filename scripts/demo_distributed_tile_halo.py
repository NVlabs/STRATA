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
Demo: Distributed tile-level halo padding on multiple GPUs.

Runs DistributedTileKNNHaloPadding across GPUs using torch.multiprocessing.spawn,
then generates three visualization figures:

  1. Reassembled padded faces (6 panels) — shows halo continuity
  2. Individual tile detail (4 panels) — interior, edge, corner, cross-face tiles
  3. Halo error vs face-level reference (6 panels)

Usage:
    python scripts/demo_distributed_tile_halo.py --num-gpus 8
    python scripts/demo_distributed_tile_halo.py --num-gpus 1   # single-GPU debug
"""

from __future__ import annotations

import argparse
import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist

from screamcast.cubesphere_transforms import (
    create_padded_faces_batched,
    faces_to_unstructured,
    reorder_2d_tensor_to_cubesphere,
    unstructured_to_6faces,
    unstructured_to_padded_faces_knn,
)
from screamcast.distributed_cubesphere_transforms import (
    DistributedTileKNNHaloPadding,
    _tile_id_to_tuple,
    build_distributed_tile_halo_plan,
)

# Grid parameters
NE = 8
NPG = 2
FACE_SIZE = NE * NPG  # 16
TILE_SIZE = 4
TILES_PER_DIM = FACE_SIZE // TILE_SIZE  # 4
HALO_WIDTH = 2
TOTAL_TILES = 6 * TILES_PER_DIM**2  # 96
PADDED_TILE = TILE_SIZE + 2 * HALO_WIDTH  # 8
PADDED_FACE = FACE_SIZE + 2 * HALO_WIDTH  # 20


def make_cubesphere_lonlat() -> dict:
    """Create synthetic lon/lat for the cubed-sphere grid."""
    face_size = FACE_SIZE
    lons, lats = [], []
    for f in range(6):
        lon_base = 60.0 * f
        lat_base = -80.0 + f * 20.0
        lon_1d = np.linspace(lon_base, lon_base + 50.0, face_size)
        lat_1d = np.linspace(lat_base, lat_base + 50.0, face_size)
        lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d, indexing="ij")

        for arr_2d, out_list in [(lon_2d, lons), (lat_2d, lats)]:
            face_t = torch.from_numpy(
                arr_2d.reshape(1, face_size, face_size).astype(np.float32)
            )
            cs = reorder_2d_tensor_to_cubesphere(face_t, ne=NE, npg=NPG)
            out_list.append(cs.numpy().flatten())

    return {
        "lon": np.concatenate(lons).astype(np.float32),
        "lat": np.concatenate(lats).astype(np.float32),
    }


def build_smooth_face_data() -> torch.Tensor:
    """
    Build a smooth 2D field on 6 faces for visualization.
    Returns: (1, 6, FACE_SIZE, FACE_SIZE)
    """
    faces = torch.zeros(1, 6, FACE_SIZE, FACE_SIZE)
    for f in range(6):
        ii = torch.linspace(0, 2 * np.pi, FACE_SIZE).unsqueeze(1)
        jj = torch.linspace(0, 2 * np.pi, FACE_SIZE).unsqueeze(0)
        faces[0, f] = torch.sin(ii) * torch.cos(jj) + f * 2.0
    return faces


def faces_to_tiles(faces: torch.Tensor) -> torch.Tensor:
    """
    Convert (1, 6, FACE_SIZE, FACE_SIZE) to (1, TOTAL_TILES, TILE_SIZE, TILE_SIZE).
    Tile ordering matches _tile_id_to_tuple: face-major, then row-major within face.
    """
    tiles = []
    for f in range(6):
        for ti in range(TILES_PER_DIM):
            for tj in range(TILES_PER_DIM):
                tile = faces[
                    :,
                    f,
                    ti * TILE_SIZE : (ti + 1) * TILE_SIZE,
                    tj * TILE_SIZE : (tj + 1) * TILE_SIZE,
                ]
                tiles.append(tile)
    return torch.stack(tiles, dim=1)  # (1, 96, 4, 4)


def padded_tiles_to_padded_faces(
    padded_tiles: torch.Tensor,
) -> torch.Tensor:
    """
    Reassemble (1, TOTAL_TILES, PADDED_TILE, PADDED_TILE) into
    (1, 6, PADDED_FACE, PADDED_FACE) by placing each tile's padded region
    into the face-level padded grid.

    Overlapping halo regions are averaged.
    """
    out = torch.zeros(1, 6, PADDED_FACE, PADDED_FACE)
    weights = torch.zeros(1, 6, PADDED_FACE, PADDED_FACE)

    for tile_id in range(TOTAL_TILES):
        f, ti, tj = _tile_id_to_tuple(tile_id, TILES_PER_DIM)
        row_s = ti * TILE_SIZE
        col_s = tj * TILE_SIZE
        out[
            :, f, row_s : row_s + PADDED_TILE, col_s : col_s + PADDED_TILE
        ] += padded_tiles[:, tile_id]
        weights[:, f, row_s : row_s + PADDED_TILE, col_s : col_s + PADDED_TILE] += 1.0

    # Average overlapping regions
    out = out / weights.clamp(min=1.0)
    return out


# ---- Distributed worker ----


def set_env(rank: int, world_size: int, port: int = 29500):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank % torch.cuda.device_count())
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)


def worker(rank: int, world_size: int, plan, tmpdir: str):
    """Each GPU runs this function."""
    set_env(rank, world_size)
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Build module
    module = DistributedTileKNNHaloPadding(plan, rank=rank).to(device)

    # Build input data
    faces = build_smooth_face_data()  # (1, 6, 16, 16)
    all_tiles = faces_to_tiles(faces)  # (1, 96, 4, 4)

    # Slice this rank's tiles
    tiles_per_rank = plan.tiles_per_rank
    start = rank * tiles_per_rank
    end = start + tiles_per_rank
    local_tiles = all_tiles[:, start:end].to(device)  # (1, tiles_per_rank, 4, 4)

    # Run distributed halo padding
    padded_local = module(local_tiles)  # (1, tiles_per_rank, 8, 8)

    # Gather all padded tiles to rank 0 (NCCL requires GPU tensors)
    padded_local_contig = padded_local.contiguous()
    if rank == 0:
        all_padded_list = [
            torch.empty_like(padded_local_contig) for _ in range(world_size)
        ]
    else:
        all_padded_list = None
    dist.gather(
        padded_local_contig,
        gather_list=all_padded_list,
        dst=0,
    )

    if rank == 0:
        # Concatenate in rank order: (1, TOTAL_TILES, 8, 8)
        all_padded = torch.cat(all_padded_list, dim=1).cpu()
        # Save for main process to visualize
        torch.save(all_padded, os.path.join(tmpdir, "padded_tiles.pt"))

    dist.destroy_process_group()


def run_single_gpu(plan) -> torch.Tensor:
    """Single-GPU path (no distributed)."""
    module = DistributedTileKNNHaloPadding(plan, rank=0)
    faces = build_smooth_face_data()
    all_tiles = faces_to_tiles(faces)
    return module(all_tiles)


# ---- Visualization ----


def plot_padded_faces(padded_faces: torch.Tensor, world_size: int, output: str):
    """Figure 1: 6 padded faces with tile grid lines."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        f"Padded Faces -- Tile Halo Exchange ({world_size} GPU{'s' if world_size > 1 else ''})",
        fontsize=14,
    )

    h = HALO_WIDTH
    for f in range(6):
        ax = axes[f // 3, f % 3]
        face_data = padded_faces[0, f].numpy()
        im = ax.imshow(face_data, origin="lower", cmap="RdBu_r", aspect="equal")
        fig.colorbar(im, ax=ax, shrink=0.7)

        # Draw tile boundaries (white grid lines within the interior)
        for k in range(1, TILES_PER_DIM):
            # Vertical
            ax.axvline(
                x=h + k * TILE_SIZE - 0.5, color="white", linewidth=0.8, alpha=0.7
            )
            # Horizontal
            ax.axhline(
                y=h + k * TILE_SIZE - 0.5, color="white", linewidth=0.8, alpha=0.7
            )

        # Draw interior boundary (dashed rectangle)
        rect = plt.Rectangle(
            (h - 0.5, h - 0.5),
            FACE_SIZE,
            FACE_SIZE,
            linewidth=2,
            edgecolor="black",
            facecolor="none",
            linestyle="--",
        )
        ax.add_patch(rect)

        ax.set_title(f"Face {f}", fontsize=11)
        ax.set_xlabel("j")
        ax.set_ylabel("i")

    plt.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


def plot_tile_details(padded_tiles: torch.Tensor, plan, output: str):
    """Figure 2: 4 representative padded tiles with halo highlighted."""
    tile_to_rank = plan.tile_to_rank
    h = HALO_WIDTH

    # Pick 4 representative tiles
    examples = [
        ("Interior", (0, 1, 1)),
        ("Edge", (0, 0, 1)),
        ("Corner", (0, 0, 0)),
        ("Cross-face\n(face 1 corner)", (1, 0, 0)),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Tile Detail View -- Halo Regions", fontsize=14)

    for idx, (label, (face, ti, tj)) in enumerate(examples):
        tile_id = face * TILES_PER_DIM**2 + ti * TILES_PER_DIM + tj
        ax = axes[idx]
        tile_data = padded_tiles[0, tile_id].numpy()

        ax.imshow(tile_data, origin="lower", cmap="RdBu_r", aspect="equal")

        # Interior boundary
        rect = plt.Rectangle(
            (h - 0.5, h - 0.5),
            TILE_SIZE,
            TILE_SIZE,
            linewidth=2,
            edgecolor="black",
            facecolor="none",
            linestyle="--",
        )
        ax.add_patch(rect)

        owner_rank = tile_to_rank[(face, ti, tj)]
        ax.set_title(
            f"{label}\nFace {face}, ({ti},{tj})\nRank {owner_rank}", fontsize=10
        )

        # Check for NaN in halo
        has_nan = np.isnan(tile_data).any()
        if has_nan:
            ax.text(
                0.5,
                0.02,
                "HAS NaN!",
                transform=ax.transAxes,
                ha="center",
                color="red",
                fontsize=12,
                fontweight="bold",
            )

    plt.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


def plot_error_analysis(
    padded_faces_from_tiles: torch.Tensor,
    reference_padded: torch.Tensor,
    output: str,
):
    """Figure 3: Absolute error vs face-level reference (6 panels)."""
    error = (padded_faces_from_tiles - reference_padded).abs()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Halo Error vs Face-Level Reference", fontsize=14)

    h = HALO_WIDTH
    for f in range(6):
        ax = axes[f // 3, f % 3]
        err = error[0, f].numpy()
        vmax = max(err.max(), 1e-8)
        im = ax.imshow(
            err, origin="lower", cmap="hot", aspect="equal", vmin=0, vmax=vmax
        )
        fig.colorbar(im, ax=ax, shrink=0.7, format="%.1e")

        # Interior boundary
        rect = plt.Rectangle(
            (h - 0.5, h - 0.5),
            FACE_SIZE,
            FACE_SIZE,
            linewidth=2,
            edgecolor="cyan",
            facecolor="none",
            linestyle="--",
        )
        ax.add_patch(rect)

        # Stats
        interior_err = err[h : h + FACE_SIZE, h : h + FACE_SIZE]
        halo_err = err.copy()
        halo_err[h : h + FACE_SIZE, h : h + FACE_SIZE] = np.nan
        ax.set_title(
            f"Face {f}\n"
            f"Interior max: {interior_err.max():.1e}\n"
            f"Halo max: {np.nanmax(halo_err):.1e}",
            fontsize=10,
        )

    plt.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


# ---- Main ----


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-gpus", type=int, default=8, help="Number of GPUs (default: 8)"
    )
    parser.add_argument("--output-dir", default=".", help="Directory for output PNGs")
    parser.add_argument(
        "--grid-tgt",
        default="",
        help="Path to padded target SCRIP grid .nc file (e.g., "
        "data/ne1024halo256pg2_scrip.nc). When provided, uses duo KNN "
        "padding for halo cell coordinates instead of naive face padding.",
    )
    args = parser.parse_args()

    world_size = args.num_gpus
    available_gpus = torch.cuda.device_count()
    if world_size > available_gpus:
        print(
            f"Requested {world_size} GPUs but only {available_gpus} available. "
            f"Using {available_gpus}."
        )
        world_size = available_gpus
    if world_size == 0:
        print("No GPUs available. Running on CPU with world_size=1.")
        world_size = 1

    # Check world_size divides total tiles
    if TOTAL_TILES % world_size != 0:
        # Find nearest valid world size
        valid = [w for w in range(1, world_size + 1) if TOTAL_TILES % w == 0]
        world_size = valid[-1]
        print(f"Adjusted world_size to {world_size} (must divide {TOTAL_TILES})")

    print(f"Grid: ne={NE}, npg={NPG}, face_size={FACE_SIZE}")
    print(
        f"Tiles: tile_size={TILE_SIZE}, tiles_per_dim={TILES_PER_DIM}, total={TOTAL_TILES}"
    )
    print(f"Halo: halo_width={HALO_WIDTH}, padded_tile={PADDED_TILE}")
    print(f"GPUs: world_size={world_size}, tiles_per_rank={TOTAL_TILES // world_size}")
    print()

    # Precompute plan (CPU, rank 0)
    print("Building halo plan...")

    grid_tgt = None
    if args.grid_tgt:
        import xarray as xr

        print(f"Loading target grid: {args.grid_tgt}")
        with xr.open_dataset(args.grid_tgt) as ds:
            grid_tgt = {
                "lon": np.asarray(ds["grid_center_lon"].values, dtype=np.float32),
                "lat": np.asarray(ds["grid_center_lat"].values, dtype=np.float32),
            }
        # Extract the source grid (interior cells) from the target grid
        # so both grids use consistent real cubed-sphere geometry.
        from screamcast.distributed_cubesphere_transforms import (
            _padded_lonlat_from_target_grid,
        )

        padded_lonlat = _padded_lonlat_from_target_grid(
            grid_tgt, FACE_SIZE, HALO_WIDTH, NPG
        )
        # Interior is at [h:h+N, h:h+N] in the padded array
        h = HALO_WIDTH
        interior_lon = padded_lonlat[0, :, h : h + FACE_SIZE, h : h + FACE_SIZE]
        interior_lat = padded_lonlat[1, :, h : h + FACE_SIZE, h : h + FACE_SIZE]
        # Convert back to unstructured via cubesphere element ordering
        grid_src_lon = faces_to_unstructured(
            torch.from_numpy(interior_lon), ne=NE, npg=NPG
        ).numpy()
        grid_src_lat = faces_to_unstructured(
            torch.from_numpy(interior_lat), ne=NE, npg=NPG
        ).numpy()
        grid_src = {"lon": grid_src_lon, "lat": grid_src_lat}
        print("Derived source grid from target grid interior")
    else:
        grid_src = make_cubesphere_lonlat()

    plan = build_distributed_tile_halo_plan(
        grid_src,
        num_elements=NE,
        tile_size=TILE_SIZE,
        halo_width=HALO_WIDTH,
        num_pg_cells=NPG,
        world_size=world_size,
        grid_tgt=grid_tgt,
    )
    print(f"Plan built: {len(plan.send_plans)} source ranks")

    # Run distributed or single-GPU
    if world_size == 1 and available_gpus == 0:
        print("Running single-GPU (CPU) path...")
        all_padded = run_single_gpu(plan)
    elif world_size == 1:
        print("Running single-GPU path...")
        all_padded = run_single_gpu(plan)
    else:
        print(f"Launching {world_size} GPU workers...")
        with tempfile.TemporaryDirectory() as tmpdir:
            torch.multiprocessing.set_start_method("spawn", force=True)
            torch.multiprocessing.spawn(
                worker,
                args=(world_size, plan, tmpdir),
                nprocs=world_size,
                join=True,
            )
            all_padded = torch.load(
                os.path.join(tmpdir, "padded_tiles.pt"), weights_only=True
            )
        print("All workers finished.")

    print(f"Padded tiles shape: {all_padded.shape}")
    nan_count = torch.isnan(all_padded).sum().item()
    print(f"NaN count: {nan_count}")

    # Reassemble into padded faces
    padded_faces_from_tiles = padded_tiles_to_padded_faces(all_padded)

    # Reference: single-GPU KNN padding on full faces, using the same grid_tgt
    # as the distributed plan so the error plot reflects algorithm differences only.
    faces = build_smooth_face_data()  # (1, 6, 16, 16)
    if grid_tgt is None:
        # No SCRIP target grid — build one from naive-padded source lon/lat
        # (consistent with build_distributed_tile_halo_plan fallback)
        lon_faces = unstructured_to_6faces(
            torch.from_numpy(grid_src["lon"].astype(np.float32)), ne=NE, npg=NPG
        )
        lat_faces = unstructured_to_6faces(
            torch.from_numpy(grid_src["lat"].astype(np.float32)), ne=NE, npg=NPG
        )
        lonlat_stacked = torch.stack([lon_faces, lat_faces], dim=0)
        padded_lonlat = create_padded_faces_batched(
            lonlat_stacked, pad_width=HALO_WIDTH
        )
        ref_grid_tgt = {
            "lon": padded_lonlat[0].numpy().reshape(-1),
            "lat": padded_lonlat[1].numpy().reshape(-1),
        }
    else:
        # Use the same SCRIP target grid that the distributed plan used
        ref_grid_tgt = grid_tgt
    data_unstructured = faces_to_unstructured(faces, ne=NE, npg=NPG)
    duo_pad_fn = unstructured_to_padded_faces_knn(
        grid_src,
        ref_grid_tgt,
        num_elements=NE,
        halo_width=HALO_WIDTH,
        num_pg_cells=NPG,
    )
    reference_padded = duo_pad_fn(data_unstructured)

    # Generate visualizations
    print("\nGenerating visualizations...")
    prefix = os.path.join(args.output_dir, "tile_halo_demo")

    plot_padded_faces(padded_faces_from_tiles, world_size, f"{prefix}_faces.png")
    plot_tile_details(all_padded, plan, f"{prefix}_tiles.png")
    plot_error_analysis(
        padded_faces_from_tiles, reference_padded, f"{prefix}_error.png"
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
