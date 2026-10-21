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

**Why `>= 3` rather than `> 3`?** The highest quality recorded in the current `videos.json` is 3. A strict threshold greater than 3 would select no data.

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
6. Generate mutual descriptor matches with a ratio test.
7. Verify correspondences with affine RANSAC.
8. Write overlays, matched-pair images, `metrics.csv`, and `run.json`.

**Limitations:** These descriptors capture sorted neighbor distances rather than explicit graph connectivity. They are not affine-invariant and can be confused by distinct constellations. Watermarks overlapping fish pixels are not explicitly removed; visually inspect detections.

## Matching method

The detector extracts candidate spot centers from thresholded masked crops.
For each detected spot, its descriptor comprises the sorted distances to its
six nearest neighboring spots, normalized by the distance to the sixth neighbor.

Tentative cross-image correspondences are generated using descriptor nearest
neighbors, a ratio test, and a mutual-nearest-neighbor check.

Affine RANSAC then verifies geometric consistency. The experiment requires
at least eight tentative matches before attempting RANSAC, uses a 5-pixel
reprojection threshold, and rejects transformations with a nonpositive
linear-part determinant. The descriptors do not encode neighbor angles or
explicit graph edges.

## Measurements and outputs

Each comparison records:

- Detected spots in source and target crops.
- Number of tentative descriptor matches.
- Whether RANSAC was attempted, and the number and fraction of inliers.
- Median accepted-match reprojection error, in pixels, when defined.
- The number of inliers divided by detected source and target spots (coverage
  **of detections**, not geographic coverage of the fish).

Each timestamped `results/003/` run writes a `metrics.csv` table, `run.json`
with the source commit and selected metadata, spot-overlay PNGs, and
correspondence PNGs. An undefined inlier ratio or median residual is represented
as a missing value, not as zero.

## Observations and results

No verified performance measurements from the current prepared masked-crop
dataset are reported here yet. Inspection of the spot overlays and matches
must precede claims about identification accuracy.

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

The next priority is checking whether detections actually correspond to pigmentation spots, and whether whole-fish matching should be narrowed to the head automatically in a future experiment.
