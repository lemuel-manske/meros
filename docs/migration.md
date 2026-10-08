# Migration from the experiment 002 baseline

The preserved baseline is `386bdcec3b00ea348c523f1d3904c41210f175ba`. Source manifests, annotation files, raw-media pointer, and historical notes are retained. Historical README track numbering is not silently reassigned to the current selection manifest.

1. Install the package and pull the baseline media with DVC.
2. Run `make migrate`. It checks legacy alignment/reference metadata and BGRA files, verifies SHA-256 checksums, copies composites into `data/media/alignment/selections/<selection_id>/`, and writes a receipt. It never deletes originals or replaces different destination contents. Missing/invalid inputs fail before copying starts.
3. Run `make evaluate` to record a new metrics/evidence run. The migration copies existing representations; it does not certify their historical parameter provenance. The baseline pointer/notes and exact copied-image hashes are retained for inspection.
4. Rebuild with `make run` when the checkpoint and CUDA environment are ready. Local stage-state names remain, but source/configuration fingerprints change; legacy track metadata lacks a completed-frame count and is intentionally recomputed. New alignment files include reference-frame identity. This is a new run, not a replacement of historical evidence.
5. Compare the independently recorded runs before updating/pushing DVC pointers. Record unexplained differences rather than relabeling old numbers.

Python imports now use `meros`, and `cmd`/`engine` moved to `processing`/`pipeline`. `IndividualTrack`/`IndividualId` became `TrackSelection`/`TrackSelectionKey`. The matcher is `processing.match_images.match_images`; drawing is opt-in. The old `python -m meros.lab.002` entry point delegates to the supported experiment CLI. The duplicate match stage and plain rectangular crop exporter were removed.

Video records may omit `frame_count`; frame completeness then checks the source video's declared count. Explicit counts can be supplied if the codec's frame-count metadata is inaccurate. Frame IDs are zero-based; selection interval endpoints are inclusive. JSON observation keys remain strings at the storage boundary. Images use BGR/BGRA ordering; alpha carries foreground support. Mirroring is applied during extraction, and seeds refer to extracted-frame coordinates.

Selection IDs contain video, track, interval, and reference. Video/track identifiers must use letters, digits, underscore, or hyphen and must not contain the reserved `--` delimiter. Existing IDs comply. Legacy alignment status `candidate` means automatically accepted and is translated on read; this does not introduce human approval.

The old DVC pipeline-state pointer is archived as text, since fingerprints are local cache state. Old backup video inventory and annotation CSVs remain until their provenance can be reconciled. They were not guessed away. No original DVC media was deleted, reorganized, pulled, or republished during this refactor.
