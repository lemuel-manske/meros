# Sandbox

Standalone scripts for manually inspecting computer-vision methods outside the normal experiment pipeline.

## Quantitative spot detector evaluation

Run `python -m meros.sandbox.evaluate_spots` to compare LoG,
SimpleBlobDetector and threshold segmentation against the existing manual
annotations under `data/metadata/spots`. The script discovers annotation keys
through `project.metadata.list_spot_annotations()` and reads points through
`project.metadata.read_spots()`; it never writes annotations.

Edit `TOLERANCES_PX` and `DISPLAY_TOLERANCE_PX` near the top of the script.
Each annotated crop is evaluated at each specified pixel tolerance using
one-to-one, distance-constrained matching (Hungarian assignment), with
per-image TP/FP/FN/F1 and aggregated micro precision, recall, F1, and mean
localization error printed to the terminal. The Matplotlib comparison shows
green matched predictions, red unmatched predictions, and cyan missed manual
spots. No outputs are saved automatically.

The current metadata contains only a small number of annotated crops. Metrics
are descriptive and do not establish detector generalization to new videos.
