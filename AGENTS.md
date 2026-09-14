# Meros — Agent Guidance and Pipeline

## Scope

Follow the author's style: explicit structure, blank lines between logical steps, and concise explanations. These rules come from the author's edits to `build_review.py` and `fn.py`.

Preserve local edits and behavior during style changes, including validation, defaults, sampling, output fields, and errors. Avoid unrelated formatting changes. These rules apply to project code, not external SAM2 code.

## Terminology

Use one meaning per term in code, paths, schemas, and documentation. Use **review**, not manifest.

| Term | Meaning | Identifier or artifact |
|---|---|---|
| Video | Source recording. | `video_id`, `data/media/videos/` |
| Frame | Full image at one video position. | `frame_idx`, `data/media/frames/` |
| Observation | *One* fish's localization in a frame: bounding box and mask area. | Entry in a track's `frames` mapping |
| Track | Associated observations of one candidate fish within a video. | `(video_id, track_id)`, `data/metadata/tracks/` |
| Crop | Image region extracted around an observation. | `crop_path`, `data/media/crops/` |
| Individual | Biological fish across videos and tracks. | `individual_id` |
| Annotation | Human label. | `data/metadata/annotations/`, review label columns |
| Visualization | Full frame with masks and track labels. | `data/media/visualizations/` |
| Review | Selected crop references and fields for human annotation. | `review.csv` |
| Dataset | Curated observations and accepted annotations. | `dataset.csv` |

A track contains observations and crops; several tracks can belong to one individual. Track IDs are local to a video; individual IDs span the dataset. A crop is not a visualization.

SAM2's external API uses `obj_id`; project code uses `track_id` and the JSON key `tracks`. Use `individual_id`, formerly `who_id`, for biological identity.

## Code organization

Order modules as imports, types, constants, functions, then the entry-point guard.

Keep experiment algorithms and orchestration in their command. Extract helpers for established reuse or a concrete domain responsibility:

- `artifacts.py`: paths;
- `metadata.py`: metadata and annotation access;
- `fn.py`: general transformations and helpers;
- `individuals.py`: known individual groups and track selections;
- `masked_crop.py`: masked crop access;
- `alignment.py`: alignment metadata types and persistence.

Place code by responsibility, not merely because an alignment command uses it. Keep `estimate_alignment` in the command; individual definitions belong to `individuals.py`. Prefer domain operations such as `read_masked_crop` and `persist_overlay` over generic names such as `read_crop` and `save_image`. Persistence helpers accept domain IDs and resolve paths through `artifacts.py`.

Name records for their content (`AlignmentMetadata`) and give status unions a named type (`AlignmentStatus`). Keep ordinary local collections inferred. Separate reading, estimation, record updates, persistence, and counters with blank lines. Name derived return values when that makes the result clearer.

Possible future reuse alone does not justify abstraction. Use artifact helpers such as `path_to_crop(video_id, frame_idx, track_id)` instead of composing storage paths from constants. Add missing path helpers there.

Commands run with defaults and configured videos, without arguments unless requested. Keep useful settings as function defaults. Put orchestration under `if __name__ == "__main__":`; add a `main()` wrapper only when an integration needs it.

Print useful results or necessary diagnostics, not routine per-frame or per-video progress. A review count and output path suffice. Shared processing functions should usually stay quiet.

## Python style

### Imports and signatures

Separate direct standard-library imports, standard-library `from` imports, and project imports:

```python
import csv

from pathlib import Path
from typing import TypedDict

from src.artifacts import get_vids
from src.metadata import (
    get_track_annotations,
    get_track_metadata,
)
```

For multiple imports from one project module, use parentheses, one name per line, and trailing commas. Single-name imports stay on one line.

Multiline definitions use one parameter per line, a final trailing comma, and the return annotation on the closing line:

```python
def select_frames(
    rows: list[ReviewRow],
    samples: int,
    min_frame_gap: int,
) -> list[ReviewRow]:
```

Short definitions and calls can stay compact. Do not expand every call.

### Types and expressions

Annotate parameters and returns. Use `TypedDict` for records and built-in generics such as `list[ReviewRow]`. Prefer plain local dictionaries (`groups = {}`, `row = {...}`); keep useful local annotations such as `spaced: list[ReviewRow] = []`.

Use four-space indentation, double quotes, f-strings, `snake_case` functions/variables, `PascalCase` types, and uppercase constants. Keep trailing commas in multiline imports, parameters, and dictionaries.

Comprehensions, generators, short sorting lambdas, early returns, `continue`, `Path`, and adjacent f-strings are welcome. No fixed line-length limit or formatter is prescribed; keep expressions readable.

### Blank lines

Separate logical steps, including inside blocks:

- initialization from processing;
- separate setup steps;
- consecutive guards and the final return;
- multiline assignments from their use;
- path creation from checks;
- validation from state updates;
- successful processing from `except`;
- command setup, processing, and final output.

Separate a state update from its early exit:

```python
if annotation is None:
    unannotated += 1

    continue
```

Keep a guard next to its immediate `raise` or `return`. Keep related counter initializations together, as well as writer creation, `writeheader()`, and `writerows()`.

If argument parsing is requested, separate parser setup, argument groups, and parsing; adjacent short argument definitions can stay grouped.

Use two blank lines between top-level functions and before the entry point. Preserve nearby spacing; the author's single blank before a type after imports is not a universal rule.

### Comments and docstrings

Comments explain reasons or constraints, not obvious operations. Use lowercase prose, including acronyms such as `id` and `csv`, with periods. Keep normal capitalization in identifiers, messages, and help text.

Standalone comment blocks may have blank lines around them. A short comment can directly precede the operation it explains.

Use docstrings selectively. Avoid long command preambles or algorithm narration. Short helper docstrings use separate quote lines, a capitalized sentence, and a period, with no blank before the first statement:

```python
def positive_int(value: str) -> int:
    """
    Convert a string to a positive integer.
    """
    number = int(value)
```

Check dependencies on `__doc__` before removing documentation. Keep extended usage instructions here.

## Pipeline

Configure filenames in `src/artifacts.py`. Run from the repository root with the virtual environment:

```sh
.venv/bin/python -m src.cmd.extract_frames
.venv/bin/python -m src.cmd.extract_tracks
.venv/bin/python -m src.cmd.extract_crops
.venv/bin/python -m src.cmd.build_review
```

### Fish selection

In `extract_tracks`, use the frame slider, press `a`, draw a box, and confirm with Enter or Space. Repeat for each distinct fish; press `q` to propagate. Canceling a box adds nothing.

Select each fish once on its earliest usable frame, including later entrants. Tracks begin at that frame; earlier observations are excluded. IDs follow selection order. Selecting a fish again creates another track. When reprocessing annotated videos, preserve selection order or update the annotations.

### Storage and labels

Store images and videos under `data/media/`, ignored by Git and reserved for DVC. Store track metadata and annotations under `data/metadata/`, tracked by Git. Keep dated human reviews in `reviews/`. Do not add `.gitkeep` files; output commands create directories as needed. DVC is not configured yet.

- Crops: `data/media/crops/<video_id>/<track_id>/<frame_idx>.jpg`, retaining source frame indices.
- Track metadata: `data/metadata/tracks/<video_id>/metadata.json`, containing each track's initial frame, box, and propagated observations.
- Optional viewpoint/quality annotations: `data/metadata/annotations/<video_id>/metadata.csv`.

```csv
track_id,start_frame,end_frame,viewpoint,quality
1,0,100,left,3
2,50,150,right,2
```

Intervals are inclusive. Overlap is allowed across tracks, not within one track. Legacy files without `track_id` belong to track `1`.

### Human review

`build_review` creates `review.csv`: up to three samples per track and viewpoint, at least 15 source frames apart within each group. Unannotated frames form a separate group with blank viewpoint and quality.

Review decisions and `individual_id` start blank. Use `Y` (yes), `N` (no), or `U` (uncertain) for decisions. Assign individual identity from biological evidence, not track IDs.

Existing `review.csv` is protected from overwriting. Commands do not replace the curated `dataset.csv`.

### Masked crops

Run `.venv/bin/python -m src.cmd.extract_masked_crops` after extracting tracks. It replays saved SAM2 prompts without interactive selection because track metadata does not store masks. Results go to `data/media/masked_crops/<video_id>/<track_id>/<frame_idx>.png`: tightly bounded BGRA crops with transparent background and original foreground colors and resolution.

Only saved observations are exported; empty masks are skipped. Masks are regenerated and may differ from the original run. Existing crops, reviews, and track metadata stay unchanged. Matching output PNGs are overwritten on reruns. This isolates the fish; it does not extract its skin pattern or identify it. Downstream consumers must handle the alpha channel explicitly.

### Alignment experiment

Run `.venv/bin/python -m src.cmd.align_masked_crops`. Defaults align `65_5/1` frames 55–63 to frame 59, and `65_6/1` frames 82–98 to frame 90, independently.

`src/individuals.py` holds `INDIVIDUALS`: known individual IDs grouped with named track/frame selections from human reviews. `src/alignment.py` holds metadata types and persistence. The command owns estimation and orchestration.

SIFT matches inside the masks estimate an affine transform with RANSAC. Local contrast enhancement is used for matching and inspection; aligned crops retain source colors. Sparse or spatially concentrated matches are rejected. A single affine transform cannot account for all body deformation.

Results: `data/media/alignment/<video_id>/<track_id>/`, with aligned PNGs and overlays. The reference is the original masked crop identified by `reference_frame` in the metadata. In overlays, reference intensity is magenta and aligned intensity is green; agreement appears gray. Color differences can also reflect illumination. Only overlapping mask interiors are shown.

Transforms and diagnostics: `data/metadata/alignment/<video_id>/<track_id>.json`. Pixel errors measure fitted inliers, not whole-body accuracy. Candidates require human inspection before fusion. Reruns overwrite matching outputs. This command does not combine frames or compare identities.

### Median composites

Run `.venv/bin/python -m src.cmd.build_composites` after inspecting alignment. It combines each configured track independently, using the original reference and candidate aligned crops listed in metadata. Changed selections require rerunning alignment.

Output: `data/media/alignment/<video_id>/<track_id>/median.png`. Each color channel uses the median of fully opaque pixels; partially transparent warp edges are excluded. Uncovered pixels stay transparent. Coverage is the union of valid pixels, so some regions may have only one contributing frame. No color normalization or sharpening is applied. Reruns overwrite composites; source crops remain unchanged.

### Composite contrast

Run `.venv/bin/python -m src.cmd.enhance_composites` after building composites. It applies CLAHE to LAB lightness (clip limit 1.5, 8×8 tiles), blended at 50% strength. Alpha is preserved; nearest foreground lightness fills the background during processing to reduce mask-edge effects.

Outputs alongside `median.png`: `median_contrast.png` and `median_comparison.png` (original left, enhanced right). Originals remain unchanged. Reruns overwrite these variants. This enhances contrast, not spatial resolution.

## Before finishing

Check local style, terminology, logical spacing, selective typing/documentation, and justified helper placement. Preserve existing work and behavior unless the task explicitly changes it.
