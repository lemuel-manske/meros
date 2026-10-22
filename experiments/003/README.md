# Experiment 003 — Spot geometry with affine RANSAC

## Question and scope

> Can local configurations of pigmentation spots on segmented fish produce candidate correspondences that survive affine geometric verification?


Explore whether geometric configurations of pigmentation spots on segmented *Epinephelus itajara* can support candidate correspondences verified with affine RANSAC.

This experiment is not a validated identity classifier. In its present form, spot detection covers the entire segmented fish, not only the head.

## Data and selection criteria

Exp-003 uses the existing `Project` metadata API. It no longer hard-codes video IDs or individual IDs.

Eligible observations must:

- Belong to a track selection in `individuals.json`.
- Have an overlapping annotation with `viewpoint="frontal"`.
- Have annotation `quality >= 3`.
- Match the annotated `track_id`.

Every eligible annotation segment creates a reference sample. The selection's reference frame is used when it lies inside the qualifying segment; otherwise, the segment midpoint is used. An offset frame (30 frames later, or 30 earlier when needed) is included only if it belongs to that segment.

An individual selection may contribute multiple qualifying annotation segments. Each segment gets a distinct sample ID.

## Comparisons and labels

The experiment compares:

1. Reference and offset frames of each eligible annotation segment (same-video positive controls).
2. Reference frames of every pair of eligible segments, including segments belonging to the same or different individuals.

The `same_individual` column is derived from `individuals.json`, never inferred by the matcher. Same-video comparisons are not independent-encounter evidence.

## Representations and processing

1. Load already prepared BGRA masked crops with `project.media.read_masked_crop()`.
2. Restrict spot detection to the segmented fish via its alpha mask; erode the mask to suppress silhouette edges.
3. Apply CLAHE and adaptive thresholding.
4. Detect spot centroids and filter connected components by size and shape.
5. Encode each candidate spot using its six nearest-neighbor distances, normalized by the farthest of those neighbors.
6. Normalize spot positions by each sample's detected-point extent, reject candidate pairs more than 0.22 apart in normalized position, and generate mutual descriptor matches with a ratio test.
7. Independently estimate an initial alignment using masked SIFT features and partial-affine RANSAC (at least 12 inliers). If registration fails, use the previous normalized-position gate.
8. Verify spot correspondences with affine RANSAC.
9. As an ablation, assign transformed spot positions to target spots with one-to-one minimum-cost matching and an 8-pixel cutoff; only attempt this when SIFT registration succeeds.
10. Write overlays, both kinds of match figures, `metrics.csv`, and `run.json`.

**Limitations:** These descriptors capture sorted neighbor distances rather than explicit graph connectivity. They are not affine-invariant and can be confused by distinct constellations. Watermarks overlapping fish pixels are not explicitly removed; visually inspect detections.

## Matching method

The detector extracts candidate spot centers from thresholded masked crops.
For each detected spot, its descriptor comprises the sorted distances to its
six nearest neighboring spots, normalized by the distance to the sixth neighbor.

Tentative cross-image correspondences are restricted to spatially nearby candidates
(normalized spot-coordinate distance at most 0.22) before applying descriptor
nearest-neighbor matching, a ratio test, and a mutual-nearest-neighbor check.
The detector's maximum spot count is increased from 180 to 320; this is still a
cap rather than a guarantee that all detected components are true pigmentation spots.

Affine RANSAC then verifies geometric consistency. The experiment requires
at least eight tentative matches before attempting RANSAC, uses a 5-pixel
reprojection threshold, and rejects transformations with a nonpositive
linear-part determinant. The descriptors do not encode neighbor angles or
explicit graph edges.

## Measurements and outputs

Each comparison records:

- Detected spots in source and target crops.
- Number of coarse-registration SIFT inliers and whether it was applied.
- Post-registration spatial match count, per-image matched fractions, repeatability relative to the smaller detection set, and median aligned-coordinate distance. These are undefined when registration fails, not zero-evidence of mismatch.
- Number of tentative descriptor matches.
- Whether RANSAC was attempted, and the number and fraction of inliers.
- Median accepted-match reprojection error, in pixels, when defined.
- The number of inliers divided by detected source and target spots (coverage
  **of detections**, not geographic coverage of the fish).

Each timestamped `results/003/` run writes a `metrics.csv` table, `run.json`
with the source commit and selected metadata, spot-overlay PNGs, and
correspondence PNGs, and separate `--spatial.png` comparisons. An undefined inlier ratio or median residual is represented
as a missing value, not as zero.

## Observations and results

Latest inspected run (two same-video positive pairs and one cross-individual negative):

| Pair | SIFT alignment inliers | Descriptor/RANSAC inliers (A) | Spatial pairs (B) | Spatial repeatability | Median spatial error |
| --- | ---: | ---: | ---: | ---: | ---: |
| h01 reference/offset | 572 | 5 | 140 | 43.75% | 3.71 px |
| h02 reference/offset | 283 | 6 | 112 | 35.00% | 3.50 px |
| h01/h02 reference | 0 | 4 | Not evaluated | Undefined | Undefined |

The negative pair did not pass SIFT registration. Its `spatial_matches=0`
means method B was skipped, **not** that the spot pattern rejected a different
individual. The positive pairs are nearby frames from the same videos, not
independent encounters.

### What the A/B comparison shows

- **Method A:** six sorted nearest-neighbor distances per spot, spatial gating,
  descriptor ratio/mutual matching, then affine RANSAC. It yielded only 5–6
  geometric inliers on the two positive pairs.
- **Method B:** masked SIFT features and partial-affine RANSAC align the images;
  the resulting transform moves source spot centers into target coordinates.
  One-to-one minimum-cost (Hungarian) assignment then pairs points within 8 px.
  It found 112–140 spatial associations in the positive pairs.
- **What explains the visually clean result:** SIFT + RANSAC performs the main
  registration. Hungarian assignment resolves competing nearby points without
  reusing targets; the cutoff leaves other points unmatched. In `--spatial.png`,
  all drawn lines are green **by design**: they are accepted spatial pairs, not
  independently verified spot identities. Method A figures distinguish RANSAC
  inliers (green) from rejected candidate matches (red).
- **What remains unknown:** whether paired detections represent the same physical
  pigmentation marks and whether their geometry can distinguish individuals.
  Dense point sets can generate close accidental pairs after alignment. Better
  spatial counts alone do not demonstrate re-identification.

### Parameter interpretation

The present detector uses CLAHE, adaptive thresholding and component filters,
then retains at most 320 components by area; these are candidate dark regions,
not guaranteed pigmentation spots. The spot detector covers the whole segmented
fish and does not explicitly exclude watermarks.

Parameters such as the 8-pixel spatial cutoff, SIFT ratio threshold, RANSAC
reprojection thresholds and detector filters are heuristic experimental
settings, including values currently embedded directly in function calls.
Increasing a matching tolerance can raise true **and false** associations.
Future comparisons should change one parameter family at a time and record its
values in run provenance; raw match counts are not an identification metric.

### Most useful follow-up

Evaluate spot-detection repeatability and spatial matching against negative
controls with a comparable **independent** geometric registration, including
chance matches at similar point density. Then expand to independent encounters
of known individuals. Do not interpret failed SIFT registration as a measured
negative spot-pattern score.

## Running the experiment

Prepare the project media and masked crops using the existing Meros pipeline first.

```bash
make run EXPERIMENT=003
```

Exp-003 takes no custom CLI arguments. It uses `Project.open()` and the shared experiment-output helper, like Exp-002.

Results are written to a unique directory under `results/003/`.

## Interpretation and limitations

Inlier count, inlier ratio, and residual error are geometry diagnostics, not identity probabilities. Reference/offset pairs from the same video share acquisition conditions and do not establish long-term re-identification.

## Next steps

1. Inspect detected components for texture, edges and watermark contamination.
2. Measure accidental spatial associations using registered negative controls.
3. Evaluate parameter sensitivity and collect independent-encounter positives.

