# Sandbox

Standalone scripts for manually inspecting computer-vision methods outside the normal experiment pipeline.

## Spot centers on frontal grouper crops

Run these as Python modules from the project environment:

- `python -m meros.sandbox.spot_log`: multi-scale Laplacian of Gaussian (LoG).
- `python -m meros.sandbox.spot_blobs`: OpenCV SimpleBlobDetector.
- `python -m meros.sandbox.spot_segmentation`: adaptive thresholding and connected-component centroids.

Each script has editable constants at the top. No CLI arguments are required.
Each draws a side-by-side pyplot window (original / detected centers) for annotated frontal crops with quality >= 3. Nothing is saved automatically.

The detectors return an `(N, 2)` NumPy array of spot-center coordinates. They do not extract SIFT descriptors, perform matching, or run RANSAC. These are heuristic baselines, not trained semantic segmentation, and detections still need manual review.

`spot_detection.py` contains common image-loading and plotting helpers.

The historical `meros.ml.spot_sift` module is retained for compatibility with `experiment_003`; these new sandbox scripts do not use it.
