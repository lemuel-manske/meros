# Sandbox

Standalone scripts for manually inspecting computer-vision methods outside the normal experiment pipeline.

## Evaluating spot detectors

Run `python -m meros.sandbox.evaluate_spots` to compare LoG, Blobs, and
Segmentation against the existing spot-center annotations, without creating
any new manual regions. The evaluator generates a mask automatically from the
saved spot centers, using editable `ROI_RADIUS_PX` and `ROI_CLOSING_PX`.
Only candidates inside that mask (and inside the fish mask) are considered.

The output reports recall of **selected annotated spots**, candidate counts,
and mean localization error at configurable distance tolerances, plus a
Matplotlib visualization of the ROI and detections. Red points are unmatched
detections, **not necessarily false positives**, because annotations were not
exhaustive. Precision and F1 are intentionally not reported. This ROI is
defined by the selected points, not by inferred anatomical eye/mouth landmarks;
therefore it does not claim to identify anatomy automatically. No generated
masks or evaluation results are saved.
