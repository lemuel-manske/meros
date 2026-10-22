# Experiment 003 — Spatial spot matching with forced registration

## Question

After bringing two segmented fish into a comparable coordinate system, how many
detected dark components can be paired spatially? In particular, how many pairs
appear **between different individuals** if registration is forced instead of
skipping the comparison when SIFT fails?

This is a diagnostic of spatial agreement, **not** an individual-ID classifier.

## Inputs and comparisons

The experiment reads BGRA masked crops from the existing pipeline. Eligible
selections have frontal annotations with quality >= 3 and matching track IDs.
For each selection it compares its reference to an offset frame and compares
reference frames across selections (including different individuals).

The existing optional hard-set frame pairs in the Python file can override the
reference/offset frames for a video; these are exploratory frame choices and
must be considered when interpreting same-video results.

## Method

1. Detect up to 320 dark connected components inside an eroded fish alpha mask,
   using grayscale CLAHE, adaptive thresholding, and component shape filters.
   The resulting centroids are *candidate* spots, not verified pigmentation.
2. Attempt masked SIFT matching followed by partial-affine RANSAC. Accept this
   registration when it has at least 12 geometric inliers.
3. **When SIFT fails**, force a coarse registration using the fish mask bounding
   boxes: translate mask centers to coincide and apply a uniform scale based on
   their longest bounding-box dimension. This does not use SIFT texture or
   detected spot correspondences. It does not account for pose, orientation,
   anatomical landmarks, or 3D shape.
4. Transform source spot centers into target coordinates and use Hungarian
   one-to-one minimum-cost assignment. Source points may remain unmatched;
   retain only target assignments within 8 pixels.
5. Save spot overlays, one comparison PNG per pair, `metrics.csv`, and
   `run.json`.

The previous six-neighbor descriptor matcher (A) and conditional spatial
matcher (B) were removed. This single method always attempts an association
when both masks have valid pixels, recording the registration source in
`alignment_method` (`sift` or `mask_bbox`).

## Metrics and interpretation

- `alignment_method`: distinguishes SIFT registration from forced mask
  bounding-box registration.
- `alignment_inliers`: SIFT RANSAC inliers, even if mask fallback is used.
- `spatial_matches`: number of uniquely assigned spots within 8 pixels.
- `spatial_source_recall`, `spatial_target_recall`: paired fraction of
  detected points in each image.
- `spatial_repeatability`: paired fraction relative to the smaller point set.
- `spatial_median_error_px`: median distance among accepted pairs.

These measures are **conditional on the registration method**. In particular,
a forced negative h01/h02 comparison can produce many spatial pairs simply
because two dense point clouds were brought into overlapping regions.
A green line means an assigned pair passed the distance cutoff, not that the
same biological spot was identified. Compare positive and negative pairs under
comparable registrations before drawing conclusions.

Previous results (before this change) were: h01 positive, 140/320 spatial
pairs with SIFT; h02 positive, 112/320 with SIFT. The h01/h02 negative was
**not evaluated** by the previous spatial method because SIFT alignment
failed. Those numbers are historical context, not results from the new fallback.

The detector still covers the entire masked fish, not just the head, and does
not explicitly exclude watermarks.

## Run

Prepare masked crops with the existing pipeline, then:

```bash
make run EXPERIMENT=003
```

Results are written to a new timestamped directory under `results/003/`.
Inspect `alignment_method` alongside match counts; do not compare fallback
scores directly with SIFT-aligned scores as though they used the same geometry.
