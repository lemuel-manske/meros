# Processing

Modules expose a preparation step and reusable image functions. Preparation runs in a fixed order and uses each step's completion check to reuse valid outputs. Rebuilding a step rebuilds the steps after it. Alignment stores transforms; composite construction applies them in memory. The matcher returns comparison statistics.


## Subject spot annotations

Run `python -m meros.processing.annotate_spots` (from an installed editable
package) to mark spot centers on all numbered JPG images under
`data/media/subjects/<subject_id>/<image_id>.jpg`. Edit `SUBJECT_ID` and
`IMAGE_ID` in the script to restrict the images shown. Existing annotations
are loaded on reopening an image, so work can be resumed.

Controls: left click adds a spot, right click removes the nearest spot,
Backspace/Ctrl+Z undoes the last point, Enter/S saves, and Escape/Q skips
without changing the saved annotation. Closing the figure also skips.

Coordinates are original-image pixels and are persisted by the metadata
adapter in `data/metadata/subjects/<subject_id>/spots/<image_id>.json`.
The JSON contains the subject/image identifiers, image path,
`coordinate_system: subject_image_pixels`, and the `points` list. The
script doesn't touch video metadata, doesn't require SAM2, and doesn't modify
subject JPGs.
