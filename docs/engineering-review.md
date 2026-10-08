# Engineering review: preserve experiment 002

Reviewed `main` at `386bdcec3b00ea348c523f1d3904c41210f175ba` on 2026-10-08.
Scope: tracked Python, metadata, documentation, setup, and DVC pointers. No production code or data was changed. Media, checkpoint, and installed SAM2 were unavailable; numerical experiment results were not rerun.

## Recommendation

Keep the processing methods that support 002. First preserve its evidence and repair reproducibility, then remove unused interfaces and establish the new names and storage layout. The largest problem is that fish identity, a video track, and a selected interval are conflated. This already causes output collisions; it is more consequential than folder spelling.

002's dependency path is frames → tracks → masked crops → alignments → composites → enhanced composites, followed by comparisons of reference/composite/enhanced representations. `match_composites()` is essential even though the separate `matches` pipeline stage is not invoked by 002.

## Findings, in priority order

### 1. Blocker: checked-in video metadata cannot load

`domain/storage.py:Video` requires `frame_count`. All four records in `data/metadata/videos/videos.json` omit it. `LocalFsMetadataStore.read_videos()` directly constructs `Video(**video_metadata)`, so these records raise `TypeError`. 002 calls the pipeline first and encounters this before comparing images.

Confirmed by executing the extracted `Video` dataclass against each JSON record. Recover actual counts from the preserved videos or complete frame inventory; do not guess from the last tracked observation, since a fish can disappear before a video ends. Validate the manifest before executing stages.

### 2. High: rerunning tracking can erase observations

`extract_tracks.propagate_tracks()` processes only frames without visualizations. `run()` creates fresh track objects and unconditionally writes the returned metadata. If all visualizations exist, propagation returns fresh tracks with empty `frames`, overwriting the saved observations. Partial reruns similarly lose existing observations and compress the video timeline, while prompt indices remain original frame indices.

Confirmed the all-visualizations-present branch using the extracted function with small stubs. A visualization is a diagnostic output, not a tracking checkpoint. Recompute over the full sequence into a new output location, validate, and replace atomically. Supporting true resume would require predictor state and consistent indexing; it is unnecessary for this cleanup.

### 3. High: artifact paths do not identify their inputs

Aligned crops are keyed by video/track/frame, omitting selection interval and reference frame. Composites add the reference frame but still omit the interval. Pair visualizations omit both intervals and reference frames.

The current manifest contains two selections of `65_7/0`, with references 170 and 210. Their disjoint intervals avoid an aligned-crop collision today, but comparisons against other selections overwrite the same match image. There are 21 logical pairs but only 16 pair path keys, before representation variants. Five negative-pair keys collide.

Give every selection a stable `selection_id`; include both selection IDs, representation, and run ID in result paths. Alignment outputs must be selection-scoped too. Test simultaneous selections of the same track with different intervals/references.

### 4. High: the historical result and current experiment inputs differ

`lab/README.md` reports positive comparisons `65_5/1` vs `65_6/1` (12 mutual, 4 inliers) and `/2` vs `/2` (15 mutual, 7 inliers). Current tracked metadata uses track IDs 0 and 1 for those videos; no track 2 exists. The current identity manifest has four individuals, seven selections, three positive pairs, and 18 negative pairs. One positive compares two intervals of the same video and track.

Preserve the README numbers as historical reported observations. Do not silently relabel them or assert that the current manifest reproduces them. Locate their originating revision/run and record exact selections, parameters, versions, and media identity. Keep same-video pairs labeled separately from cross-video evidence. Consecutive observations are not independent encounters.

002 prints statistics but does not save a result table; it discards the returned match canvas. Preserve all three representations and record each pair's labels, IDs, references, keypoint counts, forward/backward/mutual matches, inliers, and whether RANSAC was attempted. Zero inliers below eight mutual matches means verification was skipped, not that an estimated transform failed.

### 5. High: cache invalidation misses behavior-changing inputs

`frames_stage_fingerprint()` omits `mirrored`, although frame extraction flips `101_11`. Tracking fingerprints omit seed boxes and automatic/manual mode. Segmentation fingerprints omit checkpoint contents, external revision, and relevant execution configuration. Enhancement fingerprint values duplicate implementation constants; changing `CLIP_LIMIT` or `STRENGTH` leaves the declared fingerprint unchanged. Matching fingerprints omit pair-selection flags. Algorithm changes depend on manually maintained version strings.

Use one explicit configuration per stage, shared by execution and fingerprinting. Preserve the 002 values before moving them. Use media content identity for archived runs; mtime-based fingerprints are local optimization and change across machines. Include implementation/configuration identity in run provenance.

### 6. Medium: completion checks can accept incomplete outputs

Frames are considered complete if any JPEG exists. Tracking checks visualizations rather than track metadata. Alignment checks recorded candidate files but not coverage of every expected non-reference frame. Empty collections can pass several `all()`-style checks.

Validate expected frame IDs/counts, nonempty selected data, tracking records, complete alignment rows, and readable image shape/channel contracts. Treat diagnostics as optional. Enhancement currently requires its side-by-side comparison image to consider a stage complete, unnecessarily coupling computation to previews.

### 7. Medium: annotations are valuable but inconsistent

No current processing code reads the annotation CSVs. `MetadataPaths.annotations()` expects a track subdirectory that does not match their actual video-level locations. Six rows in `101_11/metadata.csv` have four fields under a five-field header. Annotation track IDs also differ from the current track inventory.

Archive these as original human annotations, with their originating tracking revision. Do not guess missing track IDs or decrement all IDs automatically. Validate their labels before using them in active evaluation. Viewpoint and quality information remains relevant to interpreting 002.

### 8. Medium: setup does not describe a reproducible experiment

There is no root README or package manifest. `make run` stops at masked crops; it does not run 002. All imports use `src.meros`, tying invocation and data paths to the repository working directory. `requirements.txt` mixes direct libraries with a large environment freeze and an unusual editable SAM2 URL containing `../../../external/sam2`. A clean install was not tested. The checkpoint and CUDA choice are hardcoded and checkpoint acquisition is undocumented.

Declare direct runtime dependencies and a tested setup workflow in `pyproject.toml`; retire the old freeze rather than retaining another dependency file. Likely groups: analysis (NumPy, OpenCV, SciPy), segmentation (Torch and pinned SAM2 with its requirements), and data tooling (DVC with the configured backend). If a lockfile is introduced later, its transitive packages should be resolved by the packaging tool rather than selected from application imports. Python must support the existing `type` statement syntax (3.12+).

## Keep, remove, or consolidate

| Area | Decision | Reason / prerequisite |
|---|---|---|
| `lab/002.py`, its three-representation comparison, `lab/README.md` | Keep and make reproducible | Preserve question, results, limitations, and exact historical provenance. |
| Frames, tracking, masked crops, alignment, median composite, enhancement | Keep | All support 002; fix execution issues before rerunning. |
| `match_composites()` and `MatchStats` | Keep; move into reusable processing code | 002 directly depends on them; name should describe image matching rather than only composites. |
| Pair loops in `lab/002.py` and `match_composites.iter_track_pairs()` | Consolidate | One enumerator with explicit labels and same-video policy; preserve baseline pair order. |
| `extract_crops.py`, `MediaPaths.crop`, crop read/write protocol and adapter methods | Remove together after reference check | Plain rectangular crops are not used by 002 or another tracked caller. Remove derived crop media only after an inventory and baseline backup. |
| `TrackAnnotation` and `MetadataPaths.annotations()` | Remove unused interfaces or implement a validated reader | Keep the source CSVs as archived evidence even if interfaces go. |
| `estimate_alignment()` convenience wrapper | Optional removal | No tracked callers; retain if it becomes the public/test entry point. The feature-based implementation is required. |
| Match-image stage and comparison/overlay/visualization outputs | Optional diagnostics | Preserve useful existing review images; move writing out of the required computational path. |
| `videos.bkp.json` | Consolidate into a full video inventory | Contains 17 video records; active manifest has four. Its extra records are dataset knowledge, not proven disposable. |
| Alignment JSONs absent from the current selection manifest | Archive, then remove from active derived metadata | Six of 13 files are outside current selections; they may document historical 002 inputs. |
| `SPEC.md` | Keep as research context | Broader scientific goals do not imply these features must be implemented now. |
| DVC media pointer and SAM2 submodule | Keep | Data and model-code provenance; their contents are not disposable based on Git alone. |
| DVC pipeline-state pointer | Preserve baseline; retire from active versioned data | Pipeline fingerprints are disposable local state, distinct from research outputs. |

Do not add another framework or second pipeline scheduler during cleanup. The current small stage graph is adequate once inputs and outputs are explicit. Preserve protocols that support storage boundaries; improve them to include filesystem methods actually used by segmentation, or isolate those methods in a dedicated adapter instead of suppressing type errors.

## Canonical vocabulary

| Term | Meaning | Existing names to change |
|---|---|---|
| `individual_id` | Biological identity across observations | Keep this field; never use it for an interval key. |
| `track_id` | Local fish trajectory within one video | Keep opaque strings; no global renumbering. |
| `TrackSelection` / `selection_id` | Chosen inclusive interval of a track | `IndividualTrack`, `IndividualId`, ambiguous `.id`. |
| `reference_frame` | Frame defining selection alignment coordinates | `ref_frame_idx` variants. |
| `frame_idx` | Zero-based frame index | `frame_num` variants; normalize JSON keys at the storage boundary. |
| `reference`, `composite`, `enhanced_composite` | Three compared representations | `original`, `enhanced`, `median` filename terminology. |
| `accepted` | Alignment satisfying automatic geometry checks | Use `accepted` in the current schema. |
| `match_images` | Pairwise feature matching | Current `match_composites` also processes reference crops. |
| `select_track_seeds` | Choose initial prompts for SAM2 | `select_subjects`; automatic mode uses human-defined boxes, not an automatic detector. |

Keep `median` and `CLAHE` as method/configuration fields. Alignment and identification matching share mechanics but use different preprocessing and thresholds (ratios 0.75 vs 0.9; minimum inliers 12 vs RANSAC minimum matches 8). Do not merge their configurations or change numerical behavior while standardizing names.

## Proposed structure

Use `src/meros/domain/` for records and contracts, `storage/` for paths and serialization, `processing/` for image algorithms, `pipeline/` for stage orchestration, and `experiments/experiment_002.py` for pair evaluation. Start with normal `meros` package imports and explicit application construction rather than wildcard exports and global stores.

Keep `data/raw/videos/` immutable; keep authoritative inventory, track seeds, identity labels, selections, and curated annotations in `data/metadata/`. Store generated tracking/masks/alignments/composites under `data/derived/`, with selection-scoped paths. Write experiment results and optional diagnostic figures under `results/002/<run_id>/`.

Separate identity labels from the experiment selection manifest: changing a label should not require recalculating image alignment. Keep the four-video subset in the 002 manifest, not as the only inventory. Store coordinate conventions, frame inclusivity, mirrored-frame conventions, and BGRA/alpha handling in a short schema document.

Use the new schema directly. Earlier layouts and pointers are available in Git history. Raw media and reproducibility outputs require durable storage; scratch caches and optional previews have different retention rules.

## Refactoring order and acceptance criteria

1. **Capture baseline evidence.** Record source commit, DVC hashes, checkpoint hash, external revision, versions, all selections/configuration, and reported historical results. Verify baseline media is recoverable. An immutable tag/snapshot can be created as part of implementation; none was created in this review.
2. **Repair reproduction separately.** Resolve missing video counts, unsafe tracking reruns, incomplete output checks, and path collisions. Compare fresh results with preserved artifacts; any numerical difference must be explained. A defect fix that changes images is a new run, not a silent replacement of historical 002.
3. **Persist experiment output.** Export structured metrics and provenance for all representations. Under the current manifest and policies, expect 21 pairs × 3 representations = 63 rows, including the separately labeled same-video positive. This row count is not a performance claim.
4. **Remove proven unused code.** Delete the plain-crop exporter and associated interfaces; consolidate pair enumeration; retire unused annotation interfaces while archiving source information. Check references and smoke-test the experiment path.
5. **Establish the current names and directories.** Use stable selection IDs and one schema. Rebuild derived artifacts in the current layout; do not maintain readers or tools for earlier layouts.
6. **Document one supported workflow.** Provide install/data/model setup and one command for 002, including how to evaluate existing artifacts without rebuilding SAM2 outputs. Add focused regression checks for reruns, selection collisions, manifest validity, cache invalidation, alpha-aware composites, and result completeness.

Completion means historical evidence remains recoverable, current 002 executes from a clean environment with available media/model, no selected inputs are ambiguous, pair results cannot overwrite one another, and cleanup preserves expected representation outputs and match statistics within explicitly recorded reproducibility limits.

## Validation performed for this review

Read all tracked application Python and metadata/configuration. Inspected recent history and reference usage. Confirmed all four video-constructor failures, reproduced the tracking empty-return branch with isolated stubs, checked every current selection against its saved observations/alignment reference (all seven matched), audited CSV field counts and track references, enumerated pair counts, and detected five pair-path collisions. No GPU execution, DVC pull, dependency installation, or result reproduction was attempted.
