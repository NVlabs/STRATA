# Contribution Guide

Strata is a research project but we are open to contributions to bug fixes.

## Pull Requests

Patches can be submitted using Github's fork and PR approach.

### Licensing Information

All source code files should start with this header:

```text
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
```
This can be done with the following command:
```
python3 ci/check_licenses.py --fix
```

### Signing Your Work

We require that all contributors "sign-off" on each of their commits. This
certifies that the contribution is your original work, or you have rights to
submit it under the same license, or a compatible license.

To sign off on a commit you simply use the `--signoff` (or `-s`) option when
committing your changes:

  ```bash
  git commit -s -m "Add cool feature."
  ```

This will append the following to your commit message:

  ```text
  Signed-off-by: Your Name <your@email.com>
  ```

By doing this you are agreeing to the full DCO:

  ```text
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129

    Everyone is permitted to copy and distribute verbatim copies of this license
    document, but changing it is not allowed.


    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I have the right to
    submit it under the open source license indicated in the file; or

    (b) The contribution is based upon previous work that, to the best of my knowledge,
    is covered under an appropriate open source license and I have the right under that
    license to submit that work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am permitted to submit under a
    different license), as indicated in the file; or

    (c) The contribution was provided directly to me by some other person who certified
    (a), (b) or (c) and I have not modified it.

    (d) I understand and agree that this project and the contribution are public and
    that a record of the contribution (including all personal information I submit with
    it, including my sign-off) is maintained indefinitely and may be redistributed
    consistent with this project or the open source license(s) involved.
  ```

### Pre-commit

For Strata development, [pre-commit](https://pre-commit.com/) is required.
This will run checks before you can commit code, to ensure that the commits are
clean and follow the style and formatting standards of this repo.

To install `pre-commit` follow the below steps inside the repository folder:

```bash
pip install pre-commit
pre-commit install
```

Once the above commands are executed, the pre-commit hooks will be activated and all
the commits will be checked for appropriate formatting.
