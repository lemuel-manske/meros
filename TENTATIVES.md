# Approaches tried so far

This is a record of exploratory work and its main lessons. The observations
below describe the examples examined so far; they do not establish identification
accuracy on independent encounters.

## Dataset and unresolved constraints

- The current dataset is very small: six individuals, only two with more than
  three images. It is also heavily imbalanced.
- Some photographs were taken years apart. This allows us to inspect changes
  across encounters, but there are too few examples to establish long-term
  stability of the identifying patterns.
- The intended capture setup is still unclear, including whether cameras will
  be fixed or operated by divers. We therefore do not yet know the image quality,
  viewing distances, or range of poses the method must handle in deployment.

## Fish segmentation and image enhancement

We used SAM2 to isolate the fish and reduce background interference. We also
explored grayscale CLAHE, multiscale contrast enhancement, and segmentation of
dark and pale head markings.

These experiments helped inspect faint patterns, but making a marking more
visible does not establish that it can be matched reliably across encounters.
The current SIFT implementation uses grayscale CLAHE and restricts descriptor
neighborhoods to the interior of the fish mask.

## Spot detection and anatomical matching

We compared the spatial arrangement of head spots using manually annotated eyes
and mouths as anatomical references. We tried both manually selected spots and
automatically detected centers, including contrast-based detection and a learned
spot detector.

Manual correspondences gave encouraging results, but some spots were selected
while comparing the photographs. Those results therefore include human matching
knowledge. Automatic detection introduced missed spots and false candidates;
adding more detected centers did not consistently improve identification.

## Learned image and local descriptors

We tried frozen MiewID embeddings on complete photographs and segmented fish,
as well as pretrained local descriptors around head spots, including ResNet18
and DINOv2. We also explored combining local appearance with geometric context.

Whole-image embeddings did not provide a dependable identity signal in the
explored examples. Local descriptors were useful in some comparisons, but
matching ambiguity remained. Training feature weights on one annotated pair
improved its training examples while worsening identification of other fish.
This does not establish that learned representations are unsuitable; the
attempted adaptation did not transfer reliably.

## Geometric alignment

We tested similarity, affine, homography, and thin-plate spline (TPS)
transformations, as well as non-rigid coherent point drift (CPD) registration.
Some experiments used manually verified spot correspondences to inspect the
alignment separately from the problem of finding matches.

We did not obtain dependable alignment of the entire fish across difficult
viewpoint and pose changes. A transformation that fits a few corresponding
points does not necessarily align the surrounding skin correctly. The results
motivated checking consistency within smaller regions rather than requiring
every match to agree with one global transformation.

## SIFT with global and local geometric verification

We extracted SIFT features from segmented, contrast-normalized fish images and
matched their descriptors. We then compared two ways of checking those matches:

- A single global homography estimated with RANSAC.
- AdaLAM filtering based on local affine consistency.

These are separate checks of the candidate matches in the current implementation.
AdaLAM can retain locally consistent matches that the global homography rejects;
it does not produce a complete alignment of the fish.

In the comparisons inspected so far, some same-individual pairs produced many
retained matches, while different-individual pairs produced few. However, some
same-individual pairs also produced no inliers. An absence of matches therefore
cannot, by itself, rule out that two photographs show the same fish. Strong pose
changes and limited shared visible texture remain obstacles.

The approach is appealing because its matching evidence is easy to inspect and
the feature-matching stages are relatively simple. Production suitability and
end-to-end runtime, including segmentation, remain to be evaluated.

The limitation is not that SIFT prevents us from using more complex geometric
models. Feature extraction, matching, and geometric verification are separate
choices. A more flexible model still needs reliable correspondences and may
also make incorrect matches easier to fit.

## Open directions

- Tune SIFT extraction and matching thresholds. SIFT has configurable parameters;
  it is not a learned model to fine-tune. Improvements on these repeatedly used
  photographs would still need independent evaluation.
- Explore HOG descriptors and backprojection-based approaches for identification.
- Explore YOLO for fish detection, a separate task from individual identification.
- Further investigate task-specific learned descriptors. CNN-based features and
  a learned spot detector have already been tried, so CNNs as a whole are not an
  untested direction.
