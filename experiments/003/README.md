# Experiment 003 — Appearance-first matching with geometric verification

## Goal

Test whether pigmentation-region **visual descriptors** and geometric consistency
can distinguish two observations of the same *Epinephelus itajara* from different
individuals. This is a small exploratory baseline, not a validated ID model
or an implementation of the paper's U-Net/HesAffNet/HardNet/Fisher-vector pipeline.

## Method

1. Select frontal track observations (annotation quality >= 3), reference frames
   and offset frames. The optional hard-set frame pairs in the Python source
   override the usual 30-frame offset for selected videos.
2. Read masked BGRA crops. Apply CLAHE and adaptive thresholding to detect up
   to 320 candidate dark-component centroids inside the eroded fish mask.
3. Restrict SIFT feature **detection** to disks of radius 16 pixels around these
   candidate centroids, intersected with the fish alpha mask. The SIFT
   descriptors describe actual local image appearance, not just centroid
   positions; their support patches can extend beyond the detection mask.
4. Match SIFT descriptors by nearest-neighbor ratio (0.7) **in both directions**,
   retaining mutually preferred pairs.
5. Apply partial-affine RANSAC (4 px reprojection threshold) to these visual
   correspondences. Green lines show inliers; red lines show rejected candidates.
   No preliminary SIFT registration or mask-bounding-box alignment is used
   to *force* spatial matches.
6. Record a simple `identity_score = inliers / min(source_features, target_features)`
   along with raw match and inlier counts. A missing/failed geometric fit scores
   zero. This score is a provisional diagnostic, **not a probability**.

## Outputs

`make run EXPERIMENT=003` writes a timestamped directory under `results/003/`,
containing spot overlays, visual correspondences, `metrics.csv`, and `run.json`.
Compare `identity_score`, `inliers`, `mutual_matches`, and `median_residual_px`
between same-individual and different-individual pairs. Counts from previous
versions of experiment 003 are not directly comparable to these features.

## Interpretation

Previous experiments found many accidental centroid-proximity matches in
h01/h02 after forced silhouette alignment, so proximity alone does not
establish identity. The current implementation instead **selects candidate
correspondences from visual appearance before verifying geometry**.

SIFT can still respond to common texture, shared acquisition conditions,
or watermarks. The spot detector covers the entire segmented fish, not an
anatomically localized head region. Same-video positives are not independent
encounters, and the available small set does not establish population-level
accuracy. Measure positive/negative score separation and expand to independent
encounters before tuning thresholds or claiming identification.
