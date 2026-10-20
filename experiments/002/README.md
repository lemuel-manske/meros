# Experiment 002 — Body-pattern matching with SIFT and affine RANSAC

## Question and scope

> With a comparable body pose, can the method distinguish the same individual from different individuals?

This experiment compares four representations of each track:

```text
reference
composite
enhanced composite (CLAHE)
Mantiuk composite
```

## Representations and processing

**Reference**

The selected reference masked crop for each track.

This is a single observation of the individual.

**Composite**

Several accepted aligned crops are combined using a per-pixel median.

The intention is to reduce temporary artifacts such as reflections, noise, and small illumination changes while preserving stable body patterns.

**Enhanced composite**

The composite is processed with CLAHE-based local contrast enhancement before SIFT feature extraction.

The objective is to make local body patterns more visible to the feature detector.

## Matching method and measurements

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

## Data and selection criteria

The experiment uses the individual track selections defined in `individuals.json`
through the `Project` metadata API. The existing processing pipeline generates
masked reference crops, aligned composites, and enhanced composites.

Same-individual labels come from the annotated individual manifest. Comparison
pairs may share video or encounter conditions; see the limitations below.

## Running the experiment

From the repository root, with the project dependencies and source media prepared:

```sh
make run EXPERIMENT=002
```

The experiment uses `Project.open()` and runs the preparation pipeline. It writes
a unique timestamped directory under `results/002/`.

Each run includes `metrics.csv` and `run.json`, a matching-statistics
`plot.png`, an `enhancements.png` comparison image, and saved input
representations. The run provenance records the source Git commit.

## Observations and results

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

## Interpretation and limitations

These results are preliminary.

The positive pairs currently have relatively similar body poses. Therefore the experiment is not testing pose-independent identification.

The current question is narrower:

> When body pose is reasonably comparable, does the visual pattern contain enough information for SIFT-based matching to distinguish individuals?

With the current small dataset, the answer appears promising: same-individual enhanced composites produced geometrically consistent matches, while different-individual comparisons did not.

However, the current sample size is too small for conclusions about general performance.

The selected observations demonstrate that SIFT with affine RANSAC can recover geometrically consistent correspondences under favorable capture conditions and comparable viewpoints. The dataset does not establish that these correspondences persist across independently recorded encounters.

## Next steps

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

Definitely, we need a broader dataset.


## Mantiuk comparison

The run also applies OpenCV's Mantiuk tone mapper to the original composite and
compares it with the reference, original composite, and CLAHE representation using
the same SIFT matching and affine RANSAC settings. The CLAHE pipeline output is
preserved; the Mantiuk representation is generated and saved inside each run.

Open `enhancements.png` for original/CLAHE/Mantiuk images side by side and
`plot.png` for matching statistics. Inlier ratio is 100 × inliers / mutual matches;
it is undefined when there are no mutual matches.

This is an adaptation of Mantiuk tone mapping, not an exact reproduction of ALFRE-ID.
The authors use `pfstmo_mantiuk06` on images before segmentation. Here we use
`cv.createTonemapMantiuk(gamma=1.0, scale=0.7, saturation=1.0)` on composites.
Input sRGB values are decoded before processing. Normalized black pixels are restored
to zero to handle OpenCV's zero-luminance division. The output is clipped to [0, 1]
before applying display gamma 2.2, preventing roundoff below zero from producing NaNs.
These are initial parameters, not paper-reported settings.

Transparent padding is filled with nearest valid fish colors during processing,
then the original alpha mask is restored. Empty masks and constant-luminance inputs
are left unchanged, since they contain no spatial contrast to process.

Compare visible body markings and genuine matching correspondences, rather than
assuming a more attractive image or more keypoints means better identification.
Tone mapping adjusts existing contrast; it does not recover detail absent from the capture.

## References

- https://doi.org/10.1007/s11263-024-02071-1
- https://github.com/kwadraterry/NORPPA/blob/main/tonemapping/tonemapping.py
- https://docs.opencv.org/4.x/d6/df5/group__photo__hdr.html
