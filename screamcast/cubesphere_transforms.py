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
import numpy as np
import torch
from earth2grid import KNNS2Interpolator


def reorder_cubesphere_to_2d_tensor(array: torch.Tensor, *, ne: int, npg: int):
    """
    Reorder cube-sphere grid to 2D meshgrid.
    Args:
        array: tensor with shape (..., ne*ne*npg*npg)
        ne: number of elements per face dimension
        npg: number of physics grid points per element dimension

    Returns:
        tensor with shape (..., ne*npg, ne*npg)
    """
    if ne <= 0 or npg <= 0:
        raise ValueError(f"ne and npg must be positive, got ne={ne}, npg={npg}")
    side = int(ne) * int(npg)
    expected = int(ne) * int(ne) * int(npg) * int(npg)
    if array.size(-1) != expected:
        raise ValueError(
            f"Expected last dim = {expected} (=ne^2*npg^2), got {array.size(-1)}"
        )

    other_dims = array.shape[:-1]
    x = array.reshape((*other_dims, ne, ne, npg, npg))
    nd = len(other_dims)
    perm = list(range(nd)) + [nd + 0, nd + 2, nd + 1, nd + 3]
    x = x.permute(perm).reshape((*other_dims, side, side))
    return x


def reorder_2d_tensor_to_cubesphere(array: torch.Tensor, *, ne: int, npg: int):
    """
    Inverse of reorder_cubesphere_to_2d_tensor: convert 2D meshgrid back to cubesphere ordering.

    Args:
        array: tensor with shape (..., ne*npg, ne*npg)
        ne: number of elements per face dimension
        npg: number of physics grid points per element dimension

    Returns:
        tensor with shape (..., ne*ne*npg*npg)
    """
    if ne <= 0 or npg <= 0:
        raise ValueError(f"ne and npg must be positive, got ne={ne}, npg={npg}")
    side = int(ne) * int(npg)
    npts = int(ne) * int(ne) * int(npg) * int(npg)

    if array.shape[-1] != side or array.shape[-2] != side:
        raise ValueError(
            f"Expected last two dims = ({side}, {side}), got {array.shape[-2:]}"
        )

    other_dims = array.shape[:-2]
    # Reshape from (side, side) to (ne, npg, ne, npg)
    x = array.reshape((*other_dims, ne, npg, ne, npg))
    nd = len(other_dims)
    # Permutation [0, 2, 1, 3] is self-inverse (swaps positions 1 and 2)
    perm = list(range(nd)) + [nd + 0, nd + 2, nd + 1, nd + 3]
    x = x.permute(perm).reshape((*other_dims, npts))
    return x


def unstructured_to_6faces(
    data: torch.Tensor, ne: int = 1024, npg: int = 2
) -> torch.Tensor:
    """
    Convert global unstructured cubesphere data to 6-face 2D tensor.

    Args:
        data: Tensor with shape (..., 6 * ne^2 * npg^2)
              e.g., (column,), (channel, column), (N, channel, column)
        ne: number of elements per face dimension
        npg: number of physics grid points per element dimension

    Returns:
        tensor with shape (..., 6, ne*npg, ne*npg)
    """
    npts_per_face = (ne * npg) ** 2
    total_pts = 6 * npts_per_face

    if data.shape[-1] != total_pts:
        raise ValueError(
            f"Expected last dim = {total_pts} (=6*ne^2*npg^2), got {data.shape[-1]}"
        )

    # Get leading dimensions
    leading_dims = data.shape[:-1]

    # Reshape to (..., 6, npts_per_face)
    data_6faces = data.reshape(*leading_dims, 6, npts_per_face)

    # reorder_cubesphere_to_2d_tensor expects (..., ne*ne*npg*npg)
    # It will convert (..., 6, npts_per_face) -> (..., 6, ne*npg, ne*npg)
    faces_2d = reorder_cubesphere_to_2d_tensor(data_6faces, ne=ne, npg=npg)
    return faces_2d  # (..., 6, ne*npg, ne*npg)


def faces_to_unstructured(
    data: torch.Tensor, ne: int = 1024, npg: int = 2
) -> torch.Tensor:
    """
    Inverse of unstructured_to_6faces: convert 6-face 2D tensor back to unstructured.

    Args:
        data: Tensor with shape (..., 6, ne*npg, ne*npg)
        ne: number of elements per face dimension
        npg: number of physics grid points per element dimension

    Returns:
        tensor with shape (..., 6 * ne^2 * npg^2)
    """
    npts_per_face = (ne * npg) ** 2
    total_pts = 6 * npts_per_face
    nside = ne * npg

    if data.shape[-3] != 6 or data.shape[-2] != nside or data.shape[-1] != nside:
        raise ValueError(
            f"Expected last 3 dims = (6, {nside}, {nside}), got {data.shape[-3:]}"
        )

    # Get leading dimensions (everything before the 6, nside, nside)
    leading_dims = data.shape[:-3]

    # Apply inverse reorder to each face: (..., 6, nside, nside) -> (..., 6, npts_per_face)
    data_flat_faces = reorder_2d_tensor_to_cubesphere(data, ne=ne, npg=npg)

    # Reshape to flatten face dimension: (..., 6 * npts_per_face)
    result = data_flat_faces.reshape(*leading_dims, total_pts)
    return result


def get_cubesphere_neighbors():
    """
    connectivity of the cube-sphere grid
    | 5 |
    | 0 | 1 | 2 | 3 |
    | 4 |

    Returns:
      neighbors[face_id][to_dir] = (nbr_id, nbr_edge)
    """
    return {
        0: {
            "top": (5, "bottom"),
            "right": (1, "left"),
            "bottom": (4, "top"),
            "left": (3, "right"),
        },
        1: {
            "top": (5, "right"),
            "right": (2, "left"),
            "bottom": (4, "right"),
            "left": (0, "right"),
        },
        2: {
            "top": (5, "top"),
            "right": (3, "left"),
            "bottom": (4, "bottom"),
            "left": (1, "right"),
        },
        3: {
            "top": (5, "left"),
            "right": (0, "left"),
            "bottom": (4, "left"),
            "left": (2, "right"),
        },
        4: {  # South cap
            "top": (0, "bottom"),
            "right": (1, "bottom"),
            "bottom": (2, "bottom"),
            "left": (3, "bottom"),
        },
        5: {  # North cap
            "top": (2, "top"),
            "right": (1, "top"),
            "bottom": (0, "top"),
            "left": (3, "top"),
        },
    }


def extract_edge_raw_batched(
    face_data: torch.Tensor, edge: str, pad_width: int
) -> torch.Tensor:
    """
    Extract edge strip from batched face data.
    Input:  face_data shape (..., x, y)
    Output: strip shape varies by edge:
      - top/bottom -> (..., pad_width, y)
      - left/right -> (..., x, pad_width)
    """
    # Handle pad_width=0: return empty tensor with correct shape
    if pad_width == 0:
        *leading, x, y = face_data.shape
        if edge in ("top", "bottom"):
            out_shape = tuple(leading) + (0, y)
        else:  # left, right
            out_shape = tuple(leading) + (x, 0)
        return face_data.new_empty(out_shape)

    if edge == "top":
        return face_data[..., -pad_width:, :]
    elif edge == "bottom":
        return face_data[..., :pad_width, :]
    elif edge == "left":
        return face_data[..., :, :pad_width]
    elif edge == "right":
        return face_data[..., :, -pad_width:]
    else:
        raise ValueError(f"Unknown edge: {edge}")


def rotate_edge_to_canonical_batched(
    strip_raw: torch.Tensor, edge: str
) -> torch.Tensor:
    """
    Rotate edge strip to canonical orientation on last two axes.
    Canonical is (pad_width, face_size) on last two dims.
    """
    if edge == "bottom":
        k = 0
    elif edge == "left":
        k = -1  # rot90 CCW
    elif edge == "right":
        k = 1  # rot90 CW
    elif edge == "top":
        k = 2
    else:
        raise ValueError(edge)

    if k == 0:
        return strip_raw
    # torch.rot90 on last two axes
    return torch.rot90(strip_raw, k=k, dims=(-2, -1))


def rotate_canonical_to_halo_batched(
    strip_canon: torch.Tensor, to_dir: str
) -> torch.Tensor:
    """
    Rotate canonical strip to match target halo orientation on last two axes.
    """
    if to_dir == "top":
        k = 0
    elif to_dir == "right":
        k = 1
    elif to_dir == "left":
        k = -1
    elif to_dir == "bottom":
        k = 2
    else:
        raise ValueError(to_dir)

    if k == 0:
        return strip_canon
    return torch.rot90(strip_canon, k=k, dims=(-2, -1))


def fill_corners_batched(padded: torch.Tensor, pad_width: int, face_size: int) -> None:
    """
    Fill the 4 corner regions of padded faces in-place.
    The filling procedure follows Appendix A2 of this paper: https://arxiv.org/abs/2311.06253

    For each corner:
    - v_edge: (pad_width, 1) from adjacent horizontal halo, varies by row
    - h_edge: (1, pad_width) from adjacent vertical halo, varies by column
    - One side of diagonal uses v_edge broadcast, other uses h_edge broadcast
    - On diagonal: average of both
    """
    padded_size = face_size + 2 * pad_width
    device = padded.device
    pw, fs = pad_width, face_size

    # Create index grids (pad_width, pad_width)
    i_idx = torch.arange(pw, device=device).view(pw, 1)
    j_idx = torch.arange(pw, device=device).view(1, pw)

    # Main diagonal masks (for bottom-left and top-right)
    main_below = i_idx > j_idx  # i > j
    main_above = i_idx < j_idx  # i < j

    # Anti-diagonal masks (for bottom-right and top-left)
    anti_below = (i_idx + j_idx) > (pw - 1)  # i + j > pw - 1
    anti_above = (i_idx + j_idx) < (pw - 1)  # i + j < pw - 1

    # Corner specs: (row_slice, col_slice, v_col, h_row, diag_type, v_side)
    # diag_type: 'main' or 'anti'
    # v_side: which side of diagonal uses v_edge ('above' or 'below')
    corners = [
        # bottom-left: main diagonal, v_edge (right) used when j > i (above), h_edge (top) when i > j (below)
        (slice(0, pw), slice(0, pw), pw, pw, "main", "above"),
        # bottom-right: anti-diagonal, v_edge (left) when i+j < pw-1 (above), h_edge (top) when i+j > pw-1 (below)
        (slice(0, pw), slice(pw + fs, padded_size), pw + fs - 1, pw, "anti", "above"),
        # top-left: anti-diagonal, v_edge (right) when i+j > pw-1 (below), h_edge (bottom) when i+j < pw-1 (above)
        (slice(pw + fs, padded_size), slice(0, pw), pw, pw + fs - 1, "anti", "below"),
        # top-right: main diagonal, v_edge (left) when i < j (above), h_edge (bottom) when i > j (below)
        (
            slice(pw + fs, padded_size),
            slice(pw + fs, padded_size),
            pw + fs - 1,
            pw + fs - 1,
            "main",
            "below",
        ),
    ]

    for row_slice, col_slice, v_col, h_row, diag_type, v_side in corners:
        # v_edge: (..., pad_width, 1) - column slice, values vary by row
        v_edge = padded[..., row_slice, v_col : v_col + 1]
        # h_edge: (..., 1, pad_width) - row slice, values vary by column
        h_edge = padded[..., h_row : h_row + 1, col_slice]

        # Broadcast to corner shape (..., pad_width, pad_width)
        corner_shape = padded.shape[:-2] + (pw, pw)
        v_broadcast = v_edge.expand(corner_shape)
        h_broadcast = h_edge.expand(corner_shape)

        # Select appropriate masks based on diagonal type
        if diag_type == "main":
            above_mask, below_mask = main_above, main_below
        else:  # anti
            above_mask, below_mask = anti_above, anti_below

        # Assign v_edge and h_edge to correct sides
        if v_side == "above":
            v_mask, h_mask = above_mask, below_mask
        else:  # v_side == 'below'
            v_mask, h_mask = below_mask, above_mask

        # Fill corner
        corner = torch.where(
            v_mask,
            v_broadcast,
            torch.where(h_mask, h_broadcast, (v_broadcast + h_broadcast) / 2),
        )

        padded[..., row_slice, col_slice] = corner


def create_padded_faces_batched(
    data: torch.Tensor, pad_width: int = 64
) -> torch.Tensor:
    """
    Create padded faces for batched cubesphere data.

    Args:
        data: Input tensor of shape (..., f, x, y) where f=6 faces
              Last 3 dims must be (6, face_size, face_size)
        pad_width: Width of padding/halo region

    Returns:
        padded: Tensor of shape (..., f, padded_size, padded_size)
                where padded_size = face_size + 2 * pad_width
    """
    if pad_width == 0:
        return data

    # Get dimensions
    *leading_dims, num_faces, face_size_x, face_size_y = data.shape
    if num_faces != 6:
        raise ValueError(f"Expected 6 faces, got {num_faces}")
    if face_size_x != face_size_y:
        raise ValueError(f"Face must be square, got {face_size_x}x{face_size_y}")
    face_size = face_size_x
    padded_size = face_size + 2 * pad_width

    # Output shape
    output_shape = tuple(leading_dims) + (6, padded_size, padded_size)
    # Use NaN for float types, 0 for integer types
    fill_value = float("nan") if data.dtype.is_floating_point else 0
    padded = torch.full(output_shape, fill_value, dtype=data.dtype, device=data.device)

    neighbors = get_cubesphere_neighbors()

    for face_id in range(6):
        # Extract this face's data: shape (..., x, y)
        face_data = data[..., face_id, :, :]

        # Place center data
        padded[
            ...,
            face_id,
            pad_width : pad_width + face_size,
            pad_width : pad_width + face_size,
        ] = face_data

        # Fill halos from neighbors
        for to_dir, (nbr_id, nbr_edge) in neighbors[face_id].items():
            # Get neighbor face data
            nbr_data = data[..., nbr_id, :, :]

            # 1) Extract raw edge strip
            strip_raw = extract_edge_raw_batched(nbr_data, nbr_edge, pad_width)

            # 2) Rotate to canonical (pad_width, face_size) on last two dims
            strip_canon = rotate_edge_to_canonical_batched(strip_raw, nbr_edge)

            # 3) Rotate canonical strip to match target halo orientation
            strip_place = rotate_canonical_to_halo_batched(strip_canon, to_dir)

            # 4) Place into padded halo
            if to_dir == "top":
                padded[
                    ...,
                    face_id,
                    padded_size - pad_width :,
                    pad_width : pad_width + face_size,
                ] = strip_place
            elif to_dir == "bottom":
                padded[
                    ..., face_id, :pad_width, pad_width : pad_width + face_size
                ] = strip_place
            elif to_dir == "left":
                padded[
                    ..., face_id, pad_width : pad_width + face_size, :pad_width
                ] = strip_place
            elif to_dir == "right":
                padded[
                    ...,
                    face_id,
                    pad_width : pad_width + face_size,
                    padded_size - pad_width :,
                ] = strip_place

    # Fill corners
    fill_corners_batched(padded, pad_width, face_size)

    return padded


def halo_dst_flat_indices(face_size: int, halo_width: int) -> np.ndarray:
    """
    Return flattened indices into (6, S, S) (flattened as face-major, then i, then j), where S = face_size + 2 * halo_width
    for ALL halo cells (edges + corners), excluding the interior.
    Args:
      face_size: Size of the interior region
      halo_width: Width of the halo to extract
    Returns:
      dst_flat: (Nt,) int64
          Flattened indices into (6, S, S) in face-major, then i, then j.
    """
    S = face_size + 2 * halo_width
    ii, jj = np.meshgrid(np.arange(S), np.arange(S), indexing="ij")
    halo = (
        (ii < halo_width)
        | (ii >= halo_width + face_size)
        | (jj < halo_width)
        | (jj >= halo_width + face_size)
    )

    ii_h = ii[halo].astype(np.int64)  # (Nhalo_face,)
    jj_h = jj[halo].astype(np.int64)

    # repeat for all 6 faces
    dst = [f * (S * S) + ii_h * S + jj_h for f in range(6)]
    return np.concatenate(dst, axis=0).astype(np.int64)  # (Nt,)


def halo_dst_flat_indices_from_gridfile(
    face_size: int,
    face_size_file: int,
    halo_width: int,
    num_faces: int = 6,
    strict: bool = True,
):
    """
    Build flattened indices selecting ONLY the halo ring of width h_use around the
    interior region of size face_size x face_size, embedded inside a file face of size face_size_file x face_size_file.
    Note here face_size_file in a predefined padded cubesphere grid may contains a wider halo than the desired halo width halo_width.

    Args:
      face_size: Size of the interior region
      face_size_file: Size of the file face
      halo_width: Width of the halo to extract
      num_faces: Number of faces in the cubesphere (default 6)
    Returns:
      dst_flat_file : (Nt,) int64
          Flattened indices into (nface, S_file, S_file) in face-major, then i, then j.
      meta : dict
          Useful metadata including halo_width_file, S_file, and slices for the extracted subdomain.
    """
    if face_size_file < face_size:
        raise ValueError(
            f"face_size_file={face_size_file} must be >= face_size={face_size}"
        )

    pad = face_size_file - face_size
    if pad % 2 != 0:
        raise ValueError(
            f"face_size_file - face_size must be even for symmetric halo. "
            f"Got face_size_file={face_size_file}, face_size={face_size}."
        )
    halo_width_file = pad // 2

    if halo_width > halo_width_file:
        raise ValueError(
            f"halo_width={halo_width} cannot exceed file halo width halo_width_file={halo_width_file}"
        )

    # Define the subdomain that corresponds to desired padded domain
    # of size S_use = N + 2*h_use
    i0 = halo_width_file - halo_width
    i1 = halo_width_file + face_size + halo_width  # exclusive
    S_use = i1 - i0
    if S_use != face_size + 2 * halo_width:
        raise RuntimeError("Internal inconsistency computing padded face size S_use.")

    if i0 < 0 or i1 > face_size_file:
        raise ValueError(
            f"Desired halo window [{i0},{i1}) is outside file face "
            f"[0,{face_size_file}). Try smaller halo_width or check sizes."
        )

    # Build indices inside that S_use×S_use window, but expressed in FILE coordinates
    ii, jj = np.meshgrid(np.arange(i0, i1), np.arange(i0, i1), indexing="ij")

    # Interior region (in FILE coords)
    interior = (
        (ii >= halo_width_file)
        & (ii < halo_width_file + face_size)
        & (jj >= halo_width_file)
        & (jj < halo_width_file + face_size)
    )

    # Halo region within the extracted window: everything except interior
    halo = ~interior

    ii_h = ii[halo].astype(np.int64)
    jj_h = jj[halo].astype(np.int64)

    # Flatten into face-major (nface, S_file, S_file)
    dst = [
        f * (face_size_file * face_size_file) + ii_h * face_size_file + jj_h
        for f in range(num_faces)
    ]
    dst_flat_file = np.concatenate(dst, axis=0).astype(np.int64)

    meta = {
        "halo_width_file": int(halo_width_file),
        "S_file": int(face_size_file),
        "h_use": int(halo_width),
        "S_use": int(S_use),
        "file_window_i0": int(i0),
        "file_window_i1": int(i1),
    }
    return dst_flat_file, meta


def _subset_target_grid_to_halo(
    grid_tgt_full: dict,
    face_size: int,
    halo_width: int,
    num_pg_cells: int,
) -> tuple[dict, np.ndarray]:
    """
    Given full target SCRIP grid dict with Nc_tgt = 6*S*S cells,
    subset it to halo-only cells in the assumed (6,S,S) face-major order.

    Returns:
      grid_tgt_halo : dict with lon/lat/lon_b/lat_b for halo cells only
      dst_flat      : flattened destination indices into (6,S,S) for scattering
    """
    lon_length = grid_tgt_full["lon"].shape[0]
    face_size_file = (lon_length / 6) ** 0.5
    if face_size_file % 1 != 0:
        raise ValueError(f"face_size_file={face_size_file} is not an integer")
    face_size_file = int(face_size_file)
    dst_flat = halo_dst_flat_indices_from_gridfile(
        face_size, face_size_file, halo_width, num_faces=6, strict=True
    )[0]

    # Subset target arrays by halo indices
    lon_full = (
        unstructured_to_6faces(
            torch.from_numpy(grid_tgt_full["lon"].astype(np.float32)),
            ne=face_size_file // num_pg_cells,
            npg=num_pg_cells,
        )
        .numpy()
        .reshape(-1)
    )
    lat_full = (
        unstructured_to_6faces(
            torch.from_numpy(grid_tgt_full["lat"].astype(np.float32)),
            ne=face_size_file // num_pg_cells,
            npg=num_pg_cells,
        )
        .numpy()
        .reshape(-1)
    )
    lon = lon_full[dst_flat]
    lat = lat_full[dst_flat]

    grid_tgt_halo = {"lon": lon, "lat": lat}
    return grid_tgt_halo, dst_flat


def unstructured_to_padded_faces_knn(
    grid_src,
    grid_tgt,
    num_elements: int,
    halo_width: int,
    num_pg_cells: int = 2,
):
    """
    Build a padding function that applies KNN halo fill for cubesphere faces.
    grid_src: dict with lon/lat arrays in unit of degrees for the source grid
    grid_tgt: dict with lon/lat arrays in unit of degrees for the target grid
    num_elements: number of elements per face
    num_pg_cells: number of physics grid points per element
    halo_width: width of the halo
    Returns:
      pad(data_src) -> padded faces with shape (..., 6, N+2h, N+2h)
    """

    def pad(data_src):  # torch.Tensor (..., 6*face_size*face_size)

        face_size = num_elements * num_pg_cells
        # 1) src to faces (your function)
        data_faces = unstructured_to_6faces(
            data_src, ne=num_elements, npg=num_pg_cells
        )  # (...,6,face_size,face_size)
        leading_dims = data_faces.shape[:-3]
        padded_size = face_size + 2 * halo_width
        output_shape = tuple(leading_dims) + (6, padded_size, padded_size)

        # Use NaN for float types, 0 for integer types
        fill_value = float("nan") if data_faces.dtype.is_floating_point else 0
        padded = torch.full(
            output_shape, fill_value, dtype=data_faces.dtype, device=data_faces.device
        )
        padded[
            ...,
            halo_width : halo_width + face_size,
            halo_width : halo_width + face_size,
        ] = data_faces
        grid_tgt_halo, _ = _subset_target_grid_to_halo(
            grid_tgt, face_size, halo_width, num_pg_cells
        )

        # expect lon/lat in unit of degrees for input to KNNS2Interpolator
        regrid = KNNS2Interpolator(
            torch.from_numpy(grid_src["lon"].astype(np.float32)),
            torch.from_numpy(grid_src["lat"].astype(np.float32)),
            torch.from_numpy(grid_tgt_halo["lon"].astype(np.float32)),
            torch.from_numpy(grid_tgt_halo["lat"].astype(np.float32)),
            k=4,
            eps=1e-7,
        )
        data_halo = regrid(data_src)

        # Merge halo and data_faces to (..., 6, S, S)
        halo_dst_flat = torch.as_tensor(
            halo_dst_flat_indices(face_size, halo_width),
            device=padded.device,
            dtype=torch.long,
        )
        padded_flat = padded.view(-1, 6 * padded_size * padded_size)

        if not torch.is_tensor(data_halo):
            data_halo = torch.as_tensor(data_halo, device=padded.device)
        data_halo = data_halo.to(device=padded.device, dtype=padded.dtype)
        data_halo_flat = data_halo.reshape(padded_flat.shape[0], -1)

        halo_count = halo_dst_flat.numel()
        if data_halo_flat.shape[1] != halo_count:
            raise ValueError(
                "Unexpected halo interpolation size: got ",
                data_halo_flat.shape[1],
                "expected ",
                halo_count,
            )

        padded_flat[:, halo_dst_flat] = data_halo_flat
        return padded_flat.view(output_shape)

    return pad


if __name__ == "__main__":
    # Example usage of create padded lat/lon fields for visualization
    import matplotlib.pyplot as plt
    import xarray as xr
    from matplotlib.patches import Rectangle

    ds0 = xr.open_dataset("../data/latlon_ne1024pg2.nc")
    lon_1d = torch.from_numpy(ds0["lon"].values)
    lat_1d = torch.from_numpy(ds0["lat"].values)

    # Stack 1D tensors first: (2, column)
    stacked_1d = torch.stack([lon_1d, lat_1d], dim=0)
    print(f"Stacked 1D shape: {stacked_1d.shape}")  # (2, 25165824)

    # Convert to 6 faces in one call: (2, 6, face_size, face_size)
    stacked_faces = unstructured_to_6faces(stacked_1d, ne=1024, npg=2).unsqueeze(0)
    print(f"Stacked input shape: {stacked_faces.shape}")  # (1, 2, 6, 2048, 2048)

    # Use batched function to create padded faces on multichannel tensor
    pad_width = 1024
    padded_stacked = create_padded_faces_batched(
        stacked_faces, pad_width=pad_width
    ).squeeze(0)
    print(f"Padded output shape: {padded_stacked.shape}")  # (2, 6, 4096, 4096)

    # Extract individual channels for plotting
    padded_lon = padded_stacked[0]  # (6, padded_size, padded_size)
    padded_lat = padded_stacked[1]  # (6, padded_size, padded_size)

    face_size = 1024 * 2  # ne * npg

    # Create one big figure: 6 rows (faces) x 2 columns (lon, lat)
    fig, axes = plt.subplots(6, 2, figsize=(24, 48))

    for face_id in range(6):
        # Longitude (left column) - convert to numpy for plotting
        im0 = axes[face_id, 0].imshow(
            padded_lon[face_id].numpy(), origin="lower", cmap="viridis"
        )
        axes[face_id, 0].set_title(f"Padded Longitude (Face {face_id})")
        plt.colorbar(im0, ax=axes[face_id, 0], label="Longitude")

        # Latitude (middle column)
        im1 = axes[face_id, 1].imshow(
            padded_lat[face_id].numpy(), origin="lower", cmap="plasma"
        )
        axes[face_id, 1].set_title(f"Padded Latitude (Face {face_id})")
        plt.colorbar(im1, ax=axes[face_id, 1], label="Latitude")

        # Add boundary boxes to all columns
        for col in range(2):
            rect_center = Rectangle(
                (pad_width - 0.5, pad_width - 0.5),
                face_size,
                face_size,
                linewidth=2,
                edgecolor="white",
                facecolor="none",
            )
            axes[face_id, col].add_patch(rect_center)

    plt.tight_layout()
    plt.savefig("test_padded_all_faces.png", dpi=150)
    plt.show()
