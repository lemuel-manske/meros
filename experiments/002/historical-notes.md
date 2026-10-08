# lab

This directory contains small experiments used to validate assumptions in the image-processing and individual-identification pipeline.

The goal is not to treat these scripts as final evaluation code. Each experiment isolates one question about the current method and helps decide what should be tested next.

> Moved experiments are archived in Git version control, and kept as reference on this file.

## 001

### Question

> [!NOTE]
> Moved to: [experiment 002](#002)

> Can the current SIFT-based matcher distinguish the same individual from different individuals using track composites?

The pipeline first produces one composite image per track. Each composite combines several aligned masked crops using a per-pixel median.

The experiment then compares composites from different videos using:

1. SIFT feature detection and description.
2. Bidirectional ratio-test matching.
3. Affine RANSAC to verify geometric consistency.

The first score tested was:

```text
score = affine_inliers / mutual_matches
```

### Initial result

The first runs produced very few SIFT correspondences:

```text
same individual:
matches = 4
inliers = 0

different individual:
matches = 1–4
inliers = 0
```

Because RANSAC is only attempted when at least 8 mutual matches are available, most comparisons never reached geometric verification.

This showed that the main limitation was not yet the RANSAC threshold or the final score. The images were producing too few reliable feature correspondences.

This led to experiment `002`.

---

## 002

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

### Current observations

For the two known same-individual comparisons, the enhanced composites produced:

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
