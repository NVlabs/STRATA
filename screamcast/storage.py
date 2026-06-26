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
import configparser
import os

DEFAULT_PATH = os.path.expanduser("~/.config/rclone/rclone.conf")


def get_s3fs_options_clone_config(remote_name, config_path=DEFAULT_PATH):
    # Parse the rclone config file
    config = configparser.ConfigParser()
    config.read(config_path)

    # Ensure the remote exists in the config
    if remote_name not in config:
        raise ValueError(f"Remote '{remote_name}' not found in rclone config.")

    # Extract credentials from the config
    remote_config = config[remote_name]

    if remote_config.get("type") != "s3":
        raise ValueError(f"Remote '{remote_name}' is not an S3 remote.")

    access_key = remote_config.get("access_key_id")
    secret_key = remote_config.get("secret_access_key")
    endpoint_url = remote_config.get("endpoint", None)  # Optional endpoint

    if not access_key or not secret_key:
        raise ValueError(
            f"Access key or secret key missing for remote '{remote_name}'."
        )

    # Instantiate and return the S3FileSystem object
    return dict(
        key=access_key,
        secret=secret_key,
        client_kwargs={"endpoint_url": endpoint_url} if endpoint_url else None,
    )
