# Processing

Modules expose a preparation step and reusable image functions. Preparation runs in a fixed order and uses each step's completion check to reuse valid outputs. Rebuilding a step rebuilds the steps after it. Alignment stores transforms; composite construction applies them in memory. The matcher returns comparison statistics.

## Manual head spot annotations

Run `python -m meros.processing.annotate_spots` with a graphical Matplotlib backend.
The script selects the midpoint frame of frontal annotations with quality >= 3 and
loads its existing masked crop. Edit `VIDEO_ID`, `TRACK_ID`, and `MIN_QUALITY`
at the top of the script to restrict which images you annotate.

Left click adds a spot center; right click removes the closest center.
Backspace/Ctrl+Z undoes the last point; Enter or S saves and advances;
Esc or Q skips without saving. An existing annotation is loaded for editing.

Each image is saved independently in
`data/metadata/spots/<video_id>/<track_id>/<frame>.json`.
The JSON includes the video ID, track ID, selected frame, source crop path,
coordinate system (`masked_crop_pixels`), and `points` as `[[x, y], ...]`.
Coordinates are relative to the masked crop, **not** the original video frame.
The annotation script does not modify the video/track metadata files or experiments.
