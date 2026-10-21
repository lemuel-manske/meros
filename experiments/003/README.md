# Experiment 003 — Head-spot graphs with affine RANSAC

## Goal

Test whether local geometric configurations of pigmentation spots on the head of *Epinephelus itajara* can produce candidate correspondences, subsequently verified with affine RANSAC.

This is an **exploratory prototype**, not a validated identification system. The method does not prove that head spots persist across years or independent encounters.

## Dataset

The current metadata identifies:

| Video | Individual | Track | Reference frame | Notes |
|---|---|---|---|---|
| h01 | 11 | 0 | 340 | quality 3; annotated `frontal` |
| h02 | 12 | 0 | 1375 | quality 3; annotated `frontal`; `mirrored: true` |

Although described as left-half head footage, the metadata marks both as `frontal`. Confirm the visible *anatomical* side before interpreting results. The `mirrored` flag is recorded, not automatically re-applied to pixels.

Initial comparisons:

- h01 reference vs h01 offset: same-video **positive control**
- h02 reference vs h02 offset: same-video **positive control**
- h01 reference vs h02 reference: different-individual **negative control**

No cross-video same-individual positive pair is currently available from these two videos.

## Method

1. Decode the annotated reference frame and one frame offset by `--delta`.
2. Optionally crop a manually supplied head region of interest (ROI).
3. Apply CLAHE and adaptive thresholding; filter dark connected components by area and shape.
4. Encode each spot using sorted normalized distances to its six nearest neighbors.
5. Generate tentative correspondences with a descriptor ratio test and mutual nearest-neighbor verification.
6. Verify tentative correspondences with OpenCV `estimateAffine2D` and RANSAC.
7. Save spot visualizations, correspondence visualizations, `metrics.csv`, and `run.json`.

The local descriptors are invariant to translation, in-plane rotation, and uniform scaling, **not** to affine deformation or head curvature. RANSAC is a geometric verification stage, not a solution for arbitrary 3D changes.

## Run

From the repository root after `make init` and `make data-pull`:

```bash
make run EXPERIMENT=003
```

To restrict detection to the head:

```bash
python -m meros.experiments.experiment_003 \
  --roi-h01 X Y WIDTH HEIGHT \
  --roi-h02 X Y WIDTH HEIGHT \
  --delta 30
```

Coordinates are pixel coordinates in decoded source frames, not masked crops. Raw videos must be present under `data/media/videos/`.

Each run writes a new directory under `results/003/`. By default no ROI is applied, which may yield non-head background detections: **inspect the diagnostic images before interpreting metrics**.

## Metrics and limitations

Reported diagnostics: detected spots per image, tentative matches, RANSAC attempted, inlier count and ratio, median inlier residual (pixels), and inlier coverage relative to detections. Coverage is **not** spatial head coverage, and these statistics are **not** identity probabilities.

Do not interpret a minimal three-point affine consensus as individual recognition. Within-video positive controls share recording conditions, so they cannot establish independent-encounter reidentification.

## Next steps

- Visually validate dot localization and tune detection on the two head images.
- Add explicit head masks / anatomical landmarks and ensure consistent left-side semantics.
- Test robustness to pose, partial visibility, and lighting changes.
- Add independently recorded positive pairs and more negative individuals.
- Compare graph-based proposals with Exp-002 SIFT + affine RANSAC under matched evaluation conditions.
