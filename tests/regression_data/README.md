# Regression / golden reference data

Holds the **local, git-ignored** golden tensors for the characterization tests in
`tests/test_characterization_inference.py`, which guard that the OSRB cleanup does
not change production-model (DiT3D / PixelDiT) inference.

The `*.pt` files are binaries and are **not** committed (see the no-binary rule in
`AGENTS.md`); they are regenerated locally at the start of a cleanup session.

Regenerate on a GPU node **before** the first deletion:

```bash
source ~/Documents/python_envs/screamcast_env/bin/activate
SCREAMCAST_REGEN_GOLDEN=1 pytest tests/test_characterization_inference.py -q
```

After each cleanup step, run the tests **without** the env var; a changed output
fails the test and means a reachable code path was altered.

For a permanent committable guard, replace the binary reference with a text
fingerprint via `pytest-regtest` (already a dev dependency).
