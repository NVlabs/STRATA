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
import io
import logging

import lmdb
import torch
from torch.utils.data import Dataset


class LMDBCacheDataset(Dataset):
    def __init__(self, dataset, db_path, map_size=1e9):
        self.dataset = dataset
        self.db_path = db_path
        self.map_size = map_size
        logging.info("Opening ldmb file at %s", db_path)
        self.env = lmdb.open(db_path, map_size=int(map_size), subdir=False, lock=False)

    def __reduce__(self):
        return (LMDBCacheDataset, (self.dataset, self.db_path, self.map_size))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(str(idx).encode())

        if byteflow is not None:
            return torch.load(io.BytesIO(byteflow))  # Load from cache

        # Compute and store if not in cache
        data = self.dataset[idx]
        with self.env.begin(write=True) as txn:
            buffer = io.BytesIO()
            torch.save(data, buffer)
            logging.info(f"Writing {idx} to from cache")
            txn.put(str(idx).encode(), buffer.getvalue())

        return data
