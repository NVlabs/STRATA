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
import zarr


def clone_zarr_group_metadata(src_group, dest_path):
    dest_group = zarr.open_group(dest_path, mode="w")

    def copy_structure(src, dest):
        for name, value in src.arrays():
            dest.create(
                name, shape=value.shape, chunks=value.chunks, dtype=value.dtype
            ).attrs.update(
                value.attrs
            )  # Copy array attributes
        for name, value in src.groups():
            new_group = dest.create_group(name)
            new_group.attrs.update(value.attrs)  # Copy group attributes
            copy_structure(value, new_group)  # Recurse into subgroup

    copy_structure(src_group, dest_group)
    return dest_group
