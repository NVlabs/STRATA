# Security Policy: Strata

NVIDIA is dedicated to the security and trust of our software products and
services, including all source code repositories managed through our
organization.

If you need to report a security issue, please use the appropriate contact
points outlined below. **Please do not report security vulnerabilities through
GitHub issues or pull requests.**

## Reporting a Vulnerability

To report a potential security vulnerability in Strata:

* **Web (preferred):** [NVIDIA Vulnerability Disclosure Program](https://www.nvidia.com/en-us/security/)
* **E-Mail:** [psirt@nvidia.com](mailto:psirt@nvidia.com)
  - We encourage you to use the following PGP key for secure email communication:
    [NVIDIA public PGP Key](https://www.nvidia.com/en-us/security/pgp-key)
* **GitHub:** Use this repository's **Security** tab > **Report a vulnerability**
  to submit a report privately.

If a vulnerability is reported through public channels (issues, pull requests, or
discussions), maintainers may limit public discussion and redirect the reporter
to the private channels above.

### What to Include

- Project name and version or branch affected
- Type of vulnerability (e.g., deserialization, code execution, information disclosure)
- Step-by-step instructions to reproduce
- Proof-of-concept code (if available)
- Potential impact assessment

NVIDIA's Product Security Incident Response Team (PSIRT) will acknowledge receipt,
validate the issue and assess severity, develop and test a fix, and publish a
security bulletin as appropriate. While NVIDIA does not currently operate a public
bug bounty program, externally reported issues are acknowledged under our
coordinated vulnerability disclosure policy.

## Security Architecture & Context

Strata (Python package `screamcast`) is a deep-learning weather emulator for the
SCREAM global atmosphere model. It trains and runs transformer-based neural
networks to emulate atmospheric physics on cubed-sphere and HEALPix grids and to
produce multi-day global forecasts.

This software operates at the **Library / CLI research-SDK** level. It is executed
directly by a researcher (via `train.py`, the `scripts/` entry points, and a
marimo notebook); it is **not** a long-running network service and exposes no
inbound API, listener, or authentication surface. Its primary security
responsibility is the safe handling of the artifacts it ingests on a trusted host:
model checkpoints, scientific datasets, and experiment configuration.

**Repository Exposure Classification:** Public.
Basis: distributed as an open-source project on public GitHub; this document is
written to public-safe detail.

**Service Exposure Classification:** External / Regulated (medium confidence).
Basis: externally distributed open-source research release (Apache-2.0); it is not
a customer-facing or commercially-supported service and handles only public
scientific data with operator-supplied credentials, but external distribution
places it above the internal tiers.

**Security boundaries.** Trusted inputs are the local execution environment and the
checkpoints, datasets, and Python configs the operator chooses to run. Untrusted
inputs are any of those obtained from third parties. Data enters through local
files and object storage (zarr/netCDF/`.pth`), CLI arguments, and environment
variables, and exits as written output datasets/checkpoints and optional
experiment telemetry.

### Threat Model

The following scenarios represent the primary security concerns for this project,
including auxiliary/support code:

1. **Malicious model checkpoint deserialization:** checkpoint loading in
   `screamcast/model_registry.py` and the rollout/eval scripts uses
   `torch.load` / `load_state_dict`, which deserialize Python pickle data. Loading
   an untrusted `.pth` checkpoint can execute arbitrary code with the operator's
   privileges.
2. **Arbitrary code execution via experiment configs:** `train.py` selects
   experiments defined as executable Python in `train_configs.py`. Running an
   attacker-supplied or tampered config executes arbitrary code.
3. **Untrusted dataset / config parsing:** the data pipeline
   (`screamcast/dali_ext_src.py`) and inference code open zarr/netCDF/`np.load`
   inputs and YAML. Crafted input files could trigger parser or deserialization
   flaws, or cause resource exhaustion.
4. **Supply-chain compromise of dependencies:** `nvidia-physicsnemo` is installed
   from a pinned GitHub source archive and `natten` from a custom wheel index. A
   compromised or relocated upstream could inject code at install time.
5. **Credential exposure via configuration or telemetry:** experiment-logging and
   object-store credentials are supplied through the environment and a local
   rclone config (`screamcast/storage.py`). Misconfiguration — committing a real
   `.env`, logging the environment, or over-broad object-store credentials — could
   leak credentials or data.
6. **Information disclosure via experiment telemetry:** training and rollout can
   push metrics and artifacts to an external experiment-tracking service; running
   against non-public data without disabling telemetry could disclose sensitive
   information.

### Critical Security Assumptions

- Assumes model checkpoints and datasets loaded by the operator are trusted; the
  project does not sandbox `torch.load` / pickle deserialization.
- Assumes the experiment configs executed by `train.py` are authored or reviewed
  by the operator, since configs are arbitrary Python.
- Assumes the execution host (workstation or HPC node) is trusted and
  access-controlled; the software performs no authentication or authorization of
  its own.
- Assumes credentials are provided by a trusted operator via the environment and
  local rclone config, and that the object store enforces its own access control;
  the project does not manage secret storage.
- Assumes dependencies are obtained from trusted sources; no integrity
  verification of packages is performed at install time.
- Assumes transport security (TLS) for object-store and telemetry traffic is
  handled by the underlying client libraries and infrastructure.
