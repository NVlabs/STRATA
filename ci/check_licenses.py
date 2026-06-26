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
import argparse
import os
import subprocess
import sys

license_header = """# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# limitations under the License."""

parser = argparse.ArgumentParser()
parser.add_argument("--fix", action="store_true")
args = parser.parse_args()

expected = "SPDX-License-Identifier: Apache-2.0"
exts = (
    ".sh",
    ".py",
    ".yaml",
    ".yml",
)

files = subprocess.check_output(["git", "ls-files", "."])  # noqa: S603, S607

failed = []
for line in files.splitlines():
    file = line.decode().strip()
    _, ext = os.path.splitext(file)
    if ext not in exts:
        continue

    with open(file) as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line.startswith("#!"):
                break

        f.seek(0)
        header = f.read(offset)
        buf = f.read(len(license_header))
        if buf != license_header:
            f.seek(offset)
            body = f.read()
            failed.append((file, header, body))

if failed:
    print("Missing license headers found for files:")
    print("----------------------------------------")
    for file, header, body in failed:
        if not args.fix:
            print(file)
            continue

        print(f"Fixing {file}")
        with open(file, "w") as f:
            f.write(header)
            f.write(license_header + "\n")
            f.write(body)

    print("Run python ci/check_licenses.py --fix to fix the license headers")
    sys.exit(1)
