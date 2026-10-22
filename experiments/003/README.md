# Experiment 003 — Appearance-first matching with geometric verification

## Question and scope

> Can visual features from pigmentation-rich regions, verified by geometric consistency, distinguish observations of the same *Epinephelus itajara* from different individuals?

This is a SIFT-based exploratory baseline, not a validated identity classifier or a reproduction of the U-Net/HesAffNet/HardNet/Fisher-vector method.

## Data and selection criteria

The experiment uses the existing `Project` metadata API and BGRA masked crops.

Eligible observations:

- Belong to a track in `individuals.json`.
- Overlap an annotation with `viewpoint="frontal"` and `quality >= 3`.
- Match the annotation's `track_id`.

Each eligible annotation yields a reference frame and, when available, an offset frame normally 30 frames away. Optional hard-set frame pairs in the Python file override this selection for specified videos; currently h01 uses frames 170 and 510.

## Comparisons and labels

The experiment compares reference/offset frames within each eligible selection and reference frames across selections.

`same_individual` comes from the metadata, not from the matcher. Reference/offset frames from a single video do **not** constitute an independent-encounter identification test.

## Representations and processing

1. Read the masked BGRA crop and erode its alpha mask to avoid silhouette edges.
2. Apply grayscale CLAHE and adaptive thresholding; filter connected components by size and shape, retaining up to 320 candidate dark-region centers.
3. Restrict **SIFT keypoint detection** to 16-pixel-radius neighborhoods around those centers, intersected with the alpha mask.
4. Compute visual SIFT descriptors, perform bidirectional nearest-neighbor ratio tests (0.7), and retain mutual correspondences.
5. Apply partial-affine RANSAC with a 4-pixel reprojection threshold to those appearance-selected matches.
6. Save detected-spot overlays, match figures, per-pair metrics, and run provenance.

SIFT descriptor support can extend beyond the keypoint-detection mask. Dark components are candidates, not independently validated pigmentation spots; the detector covers the whole segmented fish rather than a head-only anatomical ROI.

## Matching method

Appearance determines tentative correspondences **before** geometric verification. Unlike the previous spatial matcher, this implementation does not pre-align images by SIFT or mask bounding boxes and does not match spots solely because their centers are close.

Mutual descriptor matches are passed to partial-affine RANSAC when at least four are available. Green correspondence lines are RANSAC inliers; red lines are rejected candidates. A failed fit or insufficient matches yields zero inliers.

The exploratory `identity_score` is defined as:

```text
ransac_inliers / min(source_features, target_features)
```

It is a normalized diagnostic score, not a calibrated identity probability.

## Measurements and outputs

Each comparison records source/target feature counts, forward/reverse ratio-test matches, mutual matches, whether RANSAC ran, inlier count and ratio, feature coverage, `identity_score`, and median inlier reprojection residual in pixels.

`make run EXPERIMENT=003` writes a new timestamped directory under `results/003/`, containing `metrics.csv`, `run.json`, spot overlays, and correspondence PNGs. Results from earlier versions of experiment 003 are not directly comparable.

## Observations and results

Earlier spatial-assignment runs paired many points even across h01 and h02 after forced silhouette alignment, illustrating that centroid proximity is insufficient as an identity signal.

The appearance-first approach instead tests whether visual descriptors produce mutual matches that also agree geometrically. Its scores must be examined on positive and negative pairs rather than interpreted from the number of green lines alone. No general identification accuracy is established by the current small sample.

## Running the experiment

Prepare masked crops through the existing pipeline, then run:

```bash
make run EXPERIMENT=003
```

## Interpretation and limitations

SIFT may respond to common textures, anatomical structures, lighting, or watermarks rather than individual pigmentation alone. Same-video positives share acquisition conditions; the current sample is too small to establish re-identification performance across encounters or unseen individuals.

## Next steps

1. Inspect whether RANSAC inliers lie on real, stable pigmentation regions rather than background artifacts or watermarks.
2. Compare the score on additional positive and negative pairs with similar viewpoints.
3. Validate on independent encounters before tuning parameters or claiming identification.

## References

- https://doi.org/10.1049/cvi2.12337
