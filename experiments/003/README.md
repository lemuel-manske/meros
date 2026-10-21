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

1. Read precomputed BGRA masked crops for the annotated reference frame and one frame offset by `--delta`. The upstream pipeline extracts frames, applies the `mirrored` flag, tracks fish, and produces masked crops.
2. Use the fish alpha channel to exclude background pixels, with a small erosion to suppress silhouette edges. Optional watermark rectangles are excluded in **original full-frame coordinates**. No manual head ROI is used.
3. Apply CLAHE and adaptive thresholding; filter dark connected components by area and shape.
4. Encode each spot using sorted normalized distances to its six nearest neighbors.
5. Generate tentative correspondences with a descriptor ratio test and mutual nearest-neighbor verification.
6. Verify tentative correspondences with OpenCV `estimateAffine2D` and RANSAC.
7. Save spot visualizations, correspondence visualizations, `metrics.csv`, and `run.json`.

The local descriptors are invariant to translation, in-plane rotation, and uniform scaling, **not** to affine deformation or head curvature. RANSAC is a geometric verification stage, not a solution for arbitrary 3D changes.

## Run

From the repository root after preparing the project's masked crops (same default `Project.open()` storage as Exp-002):

```bash
make run EXPERIMENT=003
```

To exclude watermark regions overlapping a fish:

```bash
python -m meros.experiments.experiment_003 \
  --exclude-h01 FRAME_X FRAME_Y WIDTH HEIGHT \
  --exclude-h02 FRAME_X FRAME_Y WIDTH HEIGHT \
  --delta 30
```

Watermark exclusion coordinates use the original frame and are translated using track bounding-box metadata. Prepared masked crops must be present under `data/media/masked_crops/`. Generate them with the existing processing pipeline before running Exp-003. The experiment does not decode videos or execute SAM2. It reads selections and videos through `project.metadata` and crops through `project.media`. There are no `--data` or `--output` arguments; each run creates a timestamped directory under `results/003/`, following Exp-002.

Each run writes a new directory under `results/003/`. Spot detection covers the entire segmented fish, potentially including body spots rather than only head spots. **Inspect the diagnostic images before interpreting metrics**.

## Metrics and limitations

Reported diagnostics: detected spots per image, tentative matches, RANSAC attempted, inlier count and ratio, median inlier residual (pixels), and inlier coverage relative to detections. Coverage is **not** spatial head coverage, and these statistics are **not** identity probabilities.

Do not interpret a minimal three-point affine consensus as individual recognition. Within-video positive controls share recording conditions, so they cannot establish independent-encounter reidentification.

## Next steps

- Visually validate dot localization and tune detection on the two head images.
- If head-only identification becomes necessary, investigate an automatic head mask or anatomical landmarks; ensure consistent left-side semantics.
- Test robustness to pose, partial visibility, and lighting changes.
- Add independently recorded positive pairs and more negative individuals.
- Compare graph-based proposals with Exp-002 SIFT + affine RANSAC under matched evaluation conditions.
