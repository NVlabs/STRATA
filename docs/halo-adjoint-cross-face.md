# Cube-face edge artifacts from the halo adjoint

What `--halo-adjoint` does to the outermost pixels of every cube face, why that
is harmless at tile seams but not at face edges, and what
`--no-halo-adjoint-cross-face` changes. Doubles as a porting note for branches
that have diverged from `main`.

All measurements come from **one initialisation** (t256/halo16 cross-face
rollout of `pixeldit_..._t256_nosemackpt_crossface_sdecadal_4stepft`,
2020-10-11T00Z, 288 steps). §8 explains what that does and does not support —
read it before quoting any number here.

---

## 1. TL;DR

`--halo-adjoint` mixes each tile's neighbours into the outermost ~5 px of every
tile. At *intra-face* tile seams that is benign. Across a *cube-face* edge it
injects a grid-scale-smoothed estimate, because cross-face halo targets do not
coincide with the neighbour face's grid and must be KNN-interpolated.

At one model step the outermost pixels of every face are measurably
over-smoothed and their error runs above the local background. Over a 48 h
rollout, extreme errors concentrate on the seams — at 6 h, extreme surface
pressure errors are 107x enriched in the outermost 2 px, against a hard ceiling
of 256x.

`--no-halo-adjoint-cross-face` zeroes the adjoint's window on targets that leave
the tile's own face. Default behaviour is unchanged, so the A/B is one flag.

```bash
# baseline (unchanged default)
python scripts/ace/run_screamcast_nudged.py ... --halo-adjoint --halo-width 16
# fixed
python scripts/ace/run_screamcast_nudged.py ... --halo-adjoint --halo-width 16 \
    --no-halo-adjoint-cross-face
```

> `--halo-width` **defaults to 32**, but every number in this document is for
> `--halo-width 16`. Pass it explicitly to reproduce. (The output filenames of
> the reference runs say `halo32`; that is a stale naming convention from an
> earlier run, not the halo width used.)

---

## 2. What the halo adjoint actually does

With `--halo-adjoint`, the rollout replaces the plain interior crop
(`topology.crop`) with `NormalizedHaloAdjoint`, computing

```
D_w^-1 A^T (w . y)          D_w = A^T w
```

where `A` is the KNN halo-padding operator and `w` is a Kaiser-Bessel-derived
window over the padded tile (`kaiser_bessel_derived(tile + 2*halo, 10*pi)`).

Every interior point becomes a weighted average of **its own tile's
prediction** and **the neighbouring tiles' predictions at the halo positions
that cover it**. For tile=256 / halo=16, the neighbour's share at `j` px inside
a tile edge is

| j | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| neighbour share | **42.9%** | 29.8% | 19.2% | 11.8% | 6.9% | 3.9% | 2.1% | 1.15% | 0.6% |

So the blend is confined to the outermost ~5–7 px of each tile, but there the
neighbour's contribution is large.

---

## 3. Why this creates cube-face artifacts

The halo target grid is built by `from_padded_face_grid` from a **padded face
array** (`ne1024halo256pg2_scrip.nc`), in which the real face occupies an
interior block. Targets outside that block are the **gnomonic extension** of
the face. That is the whole problem. Measured against the actual SCRIP grid:

| halo type | offset to nearest real grid point | KNN k=4 IDW weights | effective sources |
|---|---|---|---|
| intra-face tile seam | **0.000 px** | ≈ (1, 0, 0, 0) | 1.00 — a copy |
| cross-face edge | median **0.35 px** (1.3 km), p99 0.61 | (0.478, 0.250, 0.131, 0.129) | **3.0** |
| cross-face corner | 0.62 px (2.3 km) | (0.25, 0.25, 0.25, 0.25) | **4.0** |

- **Intra-face**: the neighbour tile's halo targets *are* the same physical grid
  points, so `A` copies them (self-weight ≥ 0.9996 with `k=4, eps=1e-7`). The
  adjoint averages two estimates of the same quantity.
- **Cross-face**: the extension points sit a fraction of a cell off the
  neighbour face's grid, so every halo value is a 4-point inverse-distance
  blend — a low-pass filter at the grid scale. At cube corners the weights
  degenerate to a flat ¼ each: maximal smoothing.

The adjoint then pushes that smoothed value back onto **real** grid points with
up to 43% weight.

### The training angle

Training builds its cross-face samples the same way: the loader duo-pads each
face with a **KNN halo fill** (`GlobalCrossFaceSrc`, see the "Duo-style padding
(KNN halo fill)" path in `screamcast/dali_ext_src.py`) and tiles across it, and
`multistep_loss_edge_crop` is 0 for this checkpoint, so the loss covers the
extension strip.

So the **target** at an extension point is itself an interpolated, smoothed
field. The model is correctly trained to be smooth there. What training never
contains is the **projection back** onto real grid points — that exists only at
inference, and it is what this flag removes.

Two caveats on this argument, both real:

- The exact training halo width is not quoted here on purpose: the knob is
  named differently on the pre-physicsnemo lineage this checkpoint was trained
  from (`cross_face_halo_px`) than on `main` (`duo_padding_halo_px`, derived
  from the tile/face sizes). The mechanism is the same on both; the width is
  not, and it is not load-bearing for the argument.
- The input-side interpolation is in-distribution **along face edges**, but not
  at cube corners: the cross-face loader sets `skip_corner_tiles`, so the model
  never saw a corner-halo sample. Corner behaviour therefore has a second,
  independent explanation that this document does not separate.

---

## 4. Evidence

### 4a. One model step (10 min) — the attributable regime

Prediction's grid-scale roughness (`mean|1-px difference|` of pred ÷ truth)
in the outermost pixel, normalised by the same run's face-interior value.
<1 = over-smoothed:

| | ω | p_s | U | V | precip | qv | T2m | θ |
|---|---|---|---|---|---|---|---|---|
| face edge (x=0) | **−9.6%** | −10.5% | −4.3% | −4.0% | −4.3% | −3.2% | −2.7% | −1.8% |
| intra-face tile seam | −0.2% | **−4.5%** | +0.0% | +0.1% | −1.4% | −0.2% | −0.5% | −0.1% |

The second row is the internal control: tile seams are tile boundaries too, but
not face boundaries. For six of eight variables it is ≤0.5%. It is **not** a
clean control for p_s (−4.5%, about 43% of the face-edge effect) or precip
(−1.4%). The own-face run's intra-face values were not measured.

Ring RMSE at the outermost pixel, relative to the local background 24–64 px in:
ω 1.64, U 1.43, qv 1.35, V 1.34, T2m 1.32, θ 1.24 — but p_s **1.07** and precip
**1.05**, i.e. two of the eight barely move on this metric.

### 4b. Outlier enrichment, 1–12 h — the A/B that survives the noise floor

Enrichment = P(outlier | in the outermost 2 px) ÷ P(outlier), for
`|error| > 5σ_interior`. **The hard ceiling is 256x**, because that band is
0.39% of a face.

p_s, outermost 2 px:

| lead | 1 h | 6 h | 12 h | 24 h | 36 h | 48 h |
|---|---|---|---|---|---|---|
| baseline | 27.9x | **107x** | 89.5x | 39.8x | 8.7x | 2.7x |
| own-face | **1.6x** | **1.3x** | 0.0x | 0.0x | 0.0x | 0.0x |

At 1 h, other variables (baseline → own-face): T2m 2.0 → 1.0, U 2.2 → 1.5,
qv 1.7 → 1.4, θ 1.1 → 0.7. The p_s outlier *fraction* also drops (6 h:
0.0148% → 0.0039%).

This is the strongest evidence in the document: it sits in the lead-time range
where the runs have not yet chaotically diverged, and it aggregates many
independent structures rather than one.

### 4c. 48 h — illustrative, not evidential

In the baseline, T2m has 4308 pixels outside [180, 340] K (reaching 35 K), 99.0%
of them within 64 px of a face edge (12.1% by area), localised to a 5° box at
(35–40°N, 135–140°E) — against the cube corner where faces 1/2/5 meet. In the
own-face run the count is 11, in the Southern Ocean, 350–580 px into a face
interior. U outside ±110 m/s goes 5018 → 468.

**Do not read significance into this.** Each run contains essentially *one*
blow-up structure, so the effective sample is n≈1: a single blob landing near a
seam has probability ≈0.12 under a uniform null, and the after-run's "0% near a
seam" is computed over 11 pixels (P ≈ 0.24 even treating them as independent,
≈0.88 as one structure). It is consistent with §4a–4b, and it is what the
failure looks like, but on its own it proves nothing.

### 4d. Control

Face-interior RMSE moves ≤0.26% out to 6 h, and the face-interior roughness
value the §4a ratios are normalised by is unchanged between runs (p_s 1.3509 vs
1.3514; U 0.9883 vs 0.9884) — so the normalisation is not hiding a global
change, and the effect really is confined to the seams.

---

## 5. The change

Zero the adjoint's window on padded targets that leave the tile's own face.
`D_w = A^T w` is rebuilt from the same window, so the normalization stays exact.

- `screamcast/distributed_halo.py`: new `own_face_target_mask(topology)`, and
  `NormalizedHaloAdjoint(..., own_face_only=True)` which applies it. The
  adjoint also accepts a per-tile `(tiles_per_rank, padded, padded)` window.
- `scripts/ace/run_screamcast_nudged.py`: `--halo-adjoint-cross-face /
  --no-halo-adjoint-cross-face`, default True (current behaviour), recorded in
  the output zarr attrs.

Read the functions rather than a code excerpt here — an inlined copy would rot.

---

## 6. Porting to a diverged branch

Verify these invariants rather than pattern-matching the diff:

1. **The mask must mirror whatever `_select_padded_tile` does on your branch.**
   That function defines the halo target grid;
   `own_face_target_mask` reproduces its row/column arithmetic in face-relative
   coordinates. `test_own_face_target_mask_matches_select_padded_tile` pins the
   two together by deriving the expected mask from `_select_padded_tile`
   applied to a face indicator — port that test, not just the function.
2. **Do not reintroduce a `pad_width_data` argument.** The padded-face offset
   cancels out of both bounds, so the mask depends only on the tile's position
   within its own face. (An earlier revision of this document claimed the
   opposite; it was wrong.) The one place the offset matters is unrelated:
   `_select_padded_tile` requires `pad_width_data >= halo_width`, or it forms a
   negative slice start and wraps.
3. **Window flatten order must match `forward()`'s reshape** — tile-major C
   order over `(tiles_per_rank, P, P)`. Do not transpose it.
4. **Build the mask from the halo exchange's own topology.** Passing
   `own_face_only=True` does this; a hand-built mask from a different rank's
   topology has the same shape and would be accepted silently.

### No-GPU self-check

```python
from screamcast.distributed_halo import TileTopology, own_face_target_mask
topo = TileTopology(world_size=1, rank=0, face_size=2048, tile_size=256, halo_width=16)
m = own_face_target_mask(topo)
assert m.shape == (384, 288, 288)
assert abs((m == 0).float().mean().item() - 0.02758) < 1e-4   # 2.76% masked
assert m[topo.local_tiles.index((0, 3, 3))].min() == 1.0      # interior tile untouched
assert int((m[topo.local_tiles.index((0, 0, 3))] == 0).sum()) == 16 * 288
assert int((m[topo.local_tiles.index((0, 0, 0))] == 0).sum()) == 288**2 - 272**2
```

---

## 7. Reproducing the measurements

The diagnostics are **not in this change** — seven standalone scripts under
`scripts/plotting/`, pure additions with no coupling to shipped code, held on
`zeyuan/halo-adjoint-own-face` and proposed separately:
`plot_face_seam_bias.py` (writes the `seam_stats.npz` the others read, and
gates itself on a step-0 RMSE against truth), `plot_halo_adjoint_footprint.py`,
`plot_seam_ab_compare.py`, `plot_seam_outlier_enrichment.py`,
`plot_global_seam_check.py`, `plot_face_snapshots.py`,
`check_cross_seam_continuity.py`.

Consequence worth stating plainly: **a reviewer of this change alone cannot
re-derive the numbers above.** "Roughness" throughout means
`mean|1-px difference|` of prediction ÷ truth per column — a first-difference
ratio, not a true high-pass.

What we measured on this one init — step-1 roughness at the face edge,
normalised by each run's own face interior (1.0 = matches the interior):

| | ω | p_s | U | V | T2m | qv | θ | precip |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.904 | 0.895 | 0.957 | 0.960 | 0.973 | 0.968 | 0.982 | 0.957 |
| own-face | **0.969** | **0.967** | **0.978** | **0.982** | **1.004** | 0.980 | 0.992 | 0.980 |

All eight move toward 1.0; max overshoot 1.004.

---

## 8. Trade-offs, regressions and caveats

### What you give up

Dropping the cross-face overlap-add removes the only thing coupling the two
sides of a face seam in the *output*. They stay coupled through the input halo,
so they cannot drift apart, but a mild discontinuity can appear. Measured with
`check_cross_seam_continuity.py` (across-seam jump ÷ within-face 1-px
difference; **SCREAM truth is the reference, not 1.0**):

| lead | | U | V | ω | qv | θ | p_s | T2m |
|---|---|---|---|---|---|---|---|---|
| step 1 | baseline | 0.958 | 0.965 | 0.977 | 0.971 | 0.963 | 0.980 | 1.052 |
| | own-face | 1.055 | 1.059 | 1.080 | 1.065 | 1.055 | 1.198 | 1.154 |
| | *truth* | *1.018* | *1.030* | *1.052* | *1.033* | *1.039* | *1.140* | *1.157* |
| 12 h | baseline | 0.892 | 0.904 | 0.906 | 0.935 | 0.836 | 0.838 | 0.695 |
| | own-face | 1.211 | 1.204 | 1.135 | 1.085 | 1.093 | 1.220 | 1.264 |
| | *truth* | *1.026* | *1.012* | *1.078* | *1.024* | *1.043* | *1.139* | *1.120* |

At step 1 own-face is closer to truth on all 7. At 12 h it runs **4.8–19.0%
above** truth where the baseline was **8.7–37.9% below** — comparable
magnitude, opposite sign, and the direction that does not blow up. (12 h is
already partly contaminated; see below.)

If that matters, `own_face_target_mask(...).clamp(min=alpha)` would give
cross-face targets a reduced but nonzero weight. **Untested** — no value of
`alpha` has been run.

### Where it is worse

At step 1, p_s and precip edge ring-RMSE tick up (1.068 → 1.122, 1.048 →
1.067), and qv/p_s corner RMSE tick up (3.06 → 3.33, 1.33 → 1.89). Corner
statistics sample only 8 unique sky locations and are noisy.

`scripts/ace/run_screamcast_era5.py` builds the same adjoint with the same KBD
window and did **not** get this flag, so ERA5 rollouts retain the artifact with
no opt-out.

### The measurement caveat

The two rollouts are **not a bitwise-controlled A/B**. They differ by ~2.6e-4
relative even in deep tile interiors, where the adjoint provably cannot act
(no halo target maps there), so the model forward itself differed between the
two jobs — GPU non-determinism. That floor is fully chaotic by 24–48 h.

Consequently: §4a and §4b are in the attributable regime; §4c is not, and
individual 24–48 h numbers should not be attributed to the flag. Locking the
long-lead behaviour down would need the same config run twice for a noise
baseline, plus more inits. **This is one init.**

### On the tests

`test_masking_preserves_the_normalization` shows the masked operator still
reproduces a constant field, i.e. `D_w` is rebuilt from the window `forward`
applies, so no DC offset is introduced. It does **not** bound the change for
non-constant fields — changing those is the point of the flag.
`test_masked_adjoint_ignores_off_face_predictions` is what covers that.

Finally, "never projected back" is exact for masked targets, but on-face
targets in the outermost row still carry small `k=4` weights on the neighbouring
face, leaving ~3e-4 of residual cross-face mixing at production resolution.
