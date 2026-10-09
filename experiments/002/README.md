# Experiment 002

### Question

> With a comparable body pose, can the method distinguish the same individual from different individuals?

This experiment compares three representations of each track:

```text
reference
composite
enhanced composite
```

### Representations

**Reference**

The selected reference masked crop for each track.

This is a single observation of the individual.

**Composite**

Several accepted aligned crops are combined using a per-pixel median.

The intention is to reduce temporary artifacts such as reflections, noise, and small illumination changes while preserving stable body patterns.

**Enhanced composite**

The composite is processed with CLAHE-based local contrast enhancement before SIFT feature extraction.

The objective is to make local body patterns more visible to the feature detector.

### Viewing a run

Run `make run EXPERIMENT=002`, then open `results/002/<run>/comparisons.png`.
The Matplotlib figure is generated automatically alongside `metrics.csv`.

Each row is one observation pair. Three panels show mutual matches, affine RANSAC
inliers, and inlier ratio, with the three representations in adjacent columns.
Values are printed in each cell, including zeros. Color scales are shared across
representations within each metric; inlier ratio uses a fixed 0–100% scale.

Pair labels indicate same or different annotated individuals; an asterisk marks
same-video pairs. The selection key below the figure maps short labels to the exact
inputs. `N/A` means no mutual matches. Zero inliers can include pairs where RANSAC
was not attempted; the CSV retains that status and all matching diagnostics.

These statistics are not identity probabilities. Same-video pairs may share encounter
conditions and are not independent cross-encounter evidence.

### Matching statistics

For each pair, the experiment records:

```text
source keypoints
target keypoints
forward ratio-test matches
backward ratio-test matches
mutual matches
affine RANSAC inliers
```

`mutual matches` are feature correspondences that pass the ratio test in both directions.

`inliers` are mutual matches that are also geometrically consistent with the affine transformation estimated by RANSAC.

### Reported observations

Earlier runs reported the following enhanced-composite comparisons. The track numbering in those reports differs from the current selection manifest; consult Git history for the exact earlier inputs. These values have not been reproduced against the current manifest.

```text
same individual

65_5/1 vs 65_6/1
mutual matches = 12
inliers = 4

65_5/2 vs 65_6/2
mutual matches = 15
inliers = 7
```

For comparisons between different individuals:

```text
different individuals

mutual matches = 3–6
inliers = 0
```

The reference crops and non-enhanced composites generally produced fewer usable matches and no affine inliers.

The enhanced representation therefore currently shows the strongest separation between same-individual and different-individual pairs.

### Interpretation

These results are preliminary.

The positive pairs currently have relatively similar body poses. Therefore the experiment is not testing pose-independent identification.

The current question is narrower:

> When body pose is reasonably comparable, does the visual pattern contain enough information for SIFT-based matching to distinguish individuals?

With the current small dataset, the answer appears promising: same-individual enhanced composites produced geometrically consistent matches, while different-individual comparisons did not.

However, the current sample size is too small for conclusions about general performance.

### Next steps

The next experiments should increase the number of known individuals and explicitly record viewpoint or body pose.

This will allow comparisons such as:

```text
same individual + similar viewpoint
same individual + different viewpoint
different individual + similar viewpoint
different individual + different viewpoint
```

This is important because the current matcher may be responding to a combination of identity and pose compatibility.

Future evaluation should also keep the raw statistics separately instead of relying on a single score:

```text
mutual matches
affine inliers
inlier ratio
```

With a larger dataset, these values can be evaluated to determine which signal best separates same-individual from different-individual pairs.
