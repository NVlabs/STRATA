# PixelDiT inference optimization benchmarks

Cirrus-126 (AI agent), prepared 2026-10-26 from the October 19–20 measurements.

These opt-in kernels and benchmark tools accelerate the released STRATA bilinear
PixelDiT on one fixed workload: a GB300, batch eight, 128×128 horizontal tile,
24 vertical levels and BF16 inference. The benchmark harness installs each
composition reversibly. Existing production files and defaults are unchanged.

## Historical result

![PixelDiT optimization progress](https://gist.githubusercontent.com/nbren12/14a3e92ac7fc85e9160ee934b7ea6afc/raw/4ee42971aa736e6c8ede0aa5342a760ae11c4117/pixeldit-speedup-summary-20261026.svg)

[Public figure](https://gist.github.com/nbren12/14a3e92ac7fc85e9160ee934b7ea6afc) · [Plot data](pixeldit-speedup-summary.csv)

| Scope | Released baseline | Final confirmed composition | Latency reduction |
| --- | ---: | ---: | ---: |
| Full network | 457.37 ms | 306.36 ms | 33.0% (1.49× speedup) |
| Pixel decoder | 190.44 ms | 124.61 ms | 34.6% (1.53× speedup) |

The [summary table](pixeldit-speedup-summary.csv) contains 25 accepted composition
snapshots, including the final attention-axis and pixel residual-layout changes.
Rows come from separate historical runs with different neighboring controls;
compiler recalibration is recorded separately. Missing decoder measurements are
`nan`. The table summarizes progression rather than additive isolated speedups.
Historical revision identifiers identify original executed sources; they are not
commits on this publication branch.

The [final independent confirmation](pixeldit-selected-confirmation.json) retains
800 CUDA-event and host-wall samples per timing phase, five warmups, neighboring
before/after controls, numerical results and statistical acceptance. Candidate
and control output differences were zero. Against the pinned canonical native
schedule, full-network relative RMS difference was 0.03228% overall and 0.85861%
for the worst channel, below the 0.5% overall / 1% per-channel limits. Warmed
forward timings exclude compilation, model loading, data I/O and profiling.
No fresh GPU performance measurements were run for this publication.

Generate the summary figure without GPU dependencies:

```bash
python scripts/plot_pixeldit_speedups.py \
  --data docs/benchmarks/pixeldit-speedup-summary.csv \
  --output /tmp/pixeldit-speedups.png
```

## Code map

- `screamcast/decoder_benchmark.py`: experiment composition, canonical reference
  checks, timing phases and acceptance gates.
- `screamcast/benchmark_statistics.py`: block-bootstrap latency acceptance.
- `screamcast/conditioning_*`: shared conditioning preparation and convolution.
- `screamcast/semantic_*` and `screamcast/column_*`: backbone attention, QKV,
  gates, normalization and layouts.
- `screamcast/gelu_table.py`: exact native-generated BF16 GELU lookup.
- `screamcast/pixel_*`: pixel attention, modulation and residual layout.
- `tests/test_*`: FakeTensor contracts, parameter/storage preservation,
  install/restore behavior and acceptance statistics.

The selected composition is `pixel-state-row-layout`; other registered cases
include research alternatives and incomplete candidates. In particular,
`pixel-packed-output-projection` did not complete its independent full-network
confirmation and is not selected.

## Reproduction and limits

The original environment used Linux aarch64, Python 3.12, PyTorch
`2.13.0a0+9186a08b2c.nv26.07`, CUDA 13.3, NATTEN 0.21.5 and PhysicsNeMo 2.2.0.
The released `nvidia/strata` checkpoint was snapshot
`068cd027aff47454088302da7d5d4e7b69212dc7`, SHA-256
`a66bc75c1f04ddb43375d4280d71951c3d0874b043ad854b5791bc4d6f316a45`.
The input was the 2020-10-13 00:00 UTC face-zero top-left 128×128 weather tile,
repeated eight times. Native depthwise channel chunk size was 512.

The original input fixtures, raw profiler traces and compiler caches were
removed after the campaign. Regenerate a fixture using
`scripts/benchmark_pixeldit_forward.py --stages --save-decoder-inputs ...`, with
the required checkpoint, data environment, output directory, FLOP estimate and
source revision (see `--help`). Run comparisons with
`scripts/benchmark_pixeldit_decoder.py`, specifying input fixture, checkpoint,
result ledger and actual executed revision identifiers. The decoder CLI defaults
compare the selected composition against itself; choose `--experiment` and
`--baseline-experiment` explicitly for a new comparison.

Fixed-shape contracts deliberately reject unsupported layouts/dtypes. Training,
backward passes, other shapes/devices, full forecast rollouts and forecast skill
need separate validation before production adoption. Fresh publication checks
are described in the pull request, including any environment-blocked tests.
