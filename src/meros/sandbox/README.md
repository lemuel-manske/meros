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

## Positive-unlabeled spot candidate ranking

Run `python -m meros.sandbox.spot_pu_ranker` for an exploratory ranking model.
It combines LoG, Blob and Segmentation candidates, computes multiscale
intensity/texture features, and learns to prioritize candidates near manually
marked spot centers over a sample of distant **unlabeled** candidates.
Unlabeled candidates are *not* confirmed negatives: this is a noisy contrastive
PU heuristic, not calibrated probability estimation.

The script trains using other annotated videos and evaluates on one held-out
video at a time (leave-one-video-out), reporting recall at fixed candidate
budgets and showing an optional visualization. It does not calculate precision
or F1, save a trained model, change experiments, or write annotations. Tune
configuration constants in the script directly. The existing six annotated
images remain a small exploratory dataset; more independent encounters will
be necessary to estimate generalization.

## Scale and density diagnostics

Run `python -m meros.sandbox.spot_diagnostics` before training a new
detector. It reads all existing spot metadata through `project.metadata`,
skips empty annotations such as `h04`, and reports per-image annotated
nearest-neighbor spacing (pixels), annotated points per 10,000 fish-mask
pixels, grayscale variation, and detector recall at budgets scaled to the
number of manually selected spots. Macro recall gives each image equal
weight. Adjust constants at the top of the script and optionally show plots.

**Interpretation:** spacing between annotated centers is *not* spot diameter,
and the density of incomplete labels is *not* true spot density. Budgets
computed using reference labels are diagnostic only, not a deployable
inference policy; detector output order is not a learned quality score.
These statistics should guide a later independently evaluated multiscale
detector, not be used to rescale a held-out image using its labels.

## Adaptive LoG comparison

Run `python -m src.meros.sandbox.compare_log` to compare original and adaptive
LoG at the same fixed candidate budgets (100, 250, 500, 1000) with each image
weighted equally. This avoids using the number of annotations to choose an
inference budget. The sandbox comparison displays a side-by-side preview and
prints per-video recall and macro recall. The original LoG remains unchanged.

The experimental implementation is in `spot_log_adaptive.py`. Edit
`PREPROCESS` (`local`, `clahe`, or `none`), `SIGMAS`, response/contrast
thresholds, and nonmaximum-suppression settings directly in that file.
The detector can return positions, estimated scales and response scores.
`python -m src.meros.sandbox.spot_log_adaptive` shows the standalone preview.

**Caveats:** these are partial positive annotations, so recall is only for
the selected spots; precision cannot be determined. Results may be sensitive to
normalization parameters, and comparisons on the same small development set
should not be mistaken for held-out generalization. No model is trained and
no experiment or metadata files are modified.
