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

## Evaluation regions

Run `python -m meros.processing.annotate_spot_regions` to draw polygonal ROIs for saved spot annotations. Use I to draw included regions (around eyes and above mouth), E for excluded regions (eye/eyelid), left click to add vertices, Enter to finish each polygon, Backspace to undo a vertex or last polygon, C to toggle whether all spots inside included areas were annotated, S to save, and Q to skip. Regions are stored via the metadata store in `data/metadata/spot_regions/<video_id>/<track_id>/<frame>.json`.

The spot evaluator skips images without saved inclusion regions. Only annotations and detector candidates inside the included polygons, and outside exclusions, are assessed. For incomplete annotations, recall means coverage of the selected spots; unmatched detections are **not** presumed false positives and precision/F1 are unavailable. When marked complete, precision/F1 are reported for exhaustively annotated regions.
