# Processing

Modules expose a preparation step and reusable image functions. Preparation runs in a fixed order and uses each step's completion check to reuse valid outputs. Rebuilding a step rebuilds the steps after it. Alignment stores transforms; composite construction applies them in memory. The matcher returns comparison statistics.


## Left-eye annotations

Run `python -m meros.processing.annotate_eyes` to annotate the visible left eye
on subject photographs that have at least one saved spot center. The current
dataset is assumed to show the **left side** only.

Left-click and drag a rectangle around the *outer visible ocular outline*,
then release. The rectangle becomes a yellow axis-aligned ellipse. Press
Enter or S to save; R, Delete, or Backspace to clear and redraw; Escape or Q
to skip without overwriting an existing annotation. Closing the window skips.
Existing eyes are displayed when an image is reopened.

Annotations are stored separately from spots by the metadata adapter at
`data/metadata/subjects/<subject_id>/eyes/<image_id>.json`, with fields
`subject_id`, `image_id`, `side: left`,
`coordinate_system: subject_image_pixels`, and `eye` containing
`center`, `radius_x`, `radius_y`, and `angle_degrees` (currently zero).
The ellipse describes the visible outline, not the pupil, and is a proposed
scale reference rather than a verified anatomical calibration.

Edit `SUBJECT_ID`, `IMAGE_ID`, and `ONLY_WITH_SPOTS` at the top of
`annotate_eyes.py` to select photos. Missing and zero-point spot annotations
are excluded by default; the latter are treated as skipped, not negative
examples. Photos and existing spot JSON files are never modified.
