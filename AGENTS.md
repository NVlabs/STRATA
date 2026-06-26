# Repo Notes

- Any command that imports `earth2studio` should be run outside the sandbox. In the sandboxed environment, `earth2studio` can fail during import because its CUDA/RMM initialization path is blocked by sandbox restrictions.
- This applies to scripts, tests, and one-off Python commands that directly or indirectly import `earth2studio`.
- `agent_context/` contains local reference material for AI agents. It is not vendored product code and should not be committed. Prefer treating it as read-only context unless explicitly asked to update or replace it.

## Commiting code and git

- Run `make lint` before every commit. It invokes `pre-commit run -a` (black + ruff + added-large-files).
- If black reformats a file, the run exits non-zero and lists "files were modified by this hook". Stage the reformatted version and re-run — the second run should pass.
- You MUST NOT check in images or other binary files to the code base

## Attribute Access

- Do not use `getattr` or `hasattr` in typical code. Access attributes directly; use `isinstance` checks when branching on type.

## API And Type Discipline

- Prefer narrow, explicit contracts over defensive generality.
- For internal computational code, do not add support for both `torch` and `numpy` in the same function unless explicitly requested.
- If a function is intended for `torch.Tensor` inputs, implement it for `torch.Tensor` inputs only. Do not add fallback paths or implicit conversions.
- Type conversion belongs at the boundary or caller layer, not inside internal helper functions.
- Shape adaptation belongs at the caller layer unless shape transformation is the function's explicit purpose.
- Do not generalize code to handle arbitrary shapes unless there is a demonstrated requirement from real callers.
- Prefer simple code with clear preconditions over speculative compatibility code.
- Add validation only when it protects correctness or produces a materially clearer error for real contract violations.

## Distributed code

- Operators that hold non-trainable tensor state (lat/lon tables, basis matrices, quadrature weights, etc.) should subclass `torch.nn.Module` and expose that state via `register_buffer(..., persistent=False)`. This gives correct `.to(device)` / `.cuda()` / `.half()` semantics without a hand-rolled `to()` method. See `DistributedTileKNNHaloPadding_AllGather` and `DistributedSHTHighpass`.
- `TileTopology` owns all knowledge of the face/tile/rank mapping. Prefer its methods (`gather_tiles_to_faces`, `faces_to_local_tiles`, `crop`, `pad_coords`, `crop_coords`) to bespoke loops over `(face, ti, tj)` in `local_tiles`.
- When possible, distribute by all-reducing small derived quantities (coefficients, statistics) instead of gathering full fields.
- Tests under `tests/distributed/` run under `torchrun --nproc_per_node 8` via `make test-distributed`. If the same test should also exercise the math on plain `pytest`, use a fixture that detects whether the process group is already initialized and mocks `torch.distributed.all_reduce` when it isn't — see the `dist_ctx` fixture in `tests/distributed/test_sht_omega_filter.py`.

## Model coords

- `ScreamcastModel.input_coords()["variable"]` and `model.output_coords(in_coords)["variable"]` can differ in length and order. When applying an operation to a model *output* tensor, derive channel indices from `out_vars`, not `in_vars`.

## Testing

- Do not write "reference implementations" in tests that restate the same math as the code under test. A test that matmul's the same matrix in the same order is a tautology — it verifies that two copies of the code agree, not that the code is correct. If the first implementation has a bug, the reference will too.
- Prefer invariants and ground-truth constructions the code under test never sees:
  - Closed-form or externally-evaluated expected outputs (e.g. evaluate a known spherical harmonic with `scipy.special.sph_harm_y` rather than via the module's own basis builder).
  - Algebraic invariants: projections are idempotent, linear operators are linear, adjoints satisfy `<Ax, y> = <x, A*y>`, quadrature weights sum to a known value, etc.
  - Boundary / degenerate inputs: constants, delta functions, single-harmonic fields, all-zero inputs.
- A useful heuristic: "would a buggy implementation fail this test even if my reference code were copied verbatim from the implementation?"
- See `tests/distributed/test_sht_omega_filter.py` for a worked example — the previous version compared against a reference that reproduced the operator's math and was replaced with projection invariants.

## Output Provenance

- For new or modified NetCDF, Zarr, or HDF output workflows, record provenance in a `history` attribute.
- The `history` attribute should append a new line on each write or resume rather than overwriting prior entries.
- Each appended entry should include an absolute timestamp and the exact command used to produce or resume the output.
