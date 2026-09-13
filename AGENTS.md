# Meros — Agent Guidance and Pipeline

## Scope

Follow the author's style: explicit structure, blank lines between logical steps, and concise explanations. These rules come from the author's edits to `build_review.py` and `fn.py`.

Preserve local edits and behavior during style changes, including validation, defaults, sampling, output fields, and errors. Avoid unrelated formatting changes. These rules apply to project code, not external SAM2 code.

## Terminology

Use one meaning per term in code, paths, schemas, and documentation. Use **review**, not manifest.

| Term | Meaning | Identifier or artifact |
|---|---|---|
| Video | Source recording. | `video_id`, `data/videos/` |
| Frame | Full image at one video position. | `frame_idx`, `data/frames/` |
| Observation | *One* fish's localization in a frame: bounding box and mask area. | Entry in a track's `frames` mapping |
| Track | Associated observations of one candidate fish within a video. | `(video_id, track_id)`, `data/tracks/` |
| Crop | Image region extracted around an observation. | `crop_path`, `data/crops/` |
| Individual | Biological fish across videos and tracks. | `individual_id` |
| Annotation | Human label. | `data/annotations/`, review label columns |
| Visualization | Full frame with masks and track labels. | `data/visualizations/` |
| Review | Selected crop references and fields for human annotation. | `review.csv` |
| Dataset | Curated observations and accepted annotations. | `dataset.csv` |

A track contains observations and crops; several tracks can belong to one individual. Track IDs are local to a video; individual IDs span the dataset. A crop is not a visualization.

SAM2's external API uses `obj_id`; project code uses `track_id` and the JSON key `tracks`. Use `individual_id`, formerly `who_id`, for biological identity.

## Code organization

Order modules as imports, types, constants, functions, then the entry-point guard.

Keep command-specific logic in its command, including review building, sampling, and writing. Extract helpers only for established reuse or a concrete shared responsibility:

- `artifacts.py`: paths;
- `metadata.py`: metadata and annotation access;
- `fn.py`: general helpers, such as `positive_int`.

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

- Crops: `data/crops/<video_id>/<track_id>/<frame_idx>.jpg`, retaining source frame indices.
- Track metadata: `data/tracks/<video_id>/metadata.json`, containing each track's initial frame, box, and propagated observations.
- Optional viewpoint/quality annotations: `data/annotations/<video_id>/metadata.csv`.

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

## Before finishing

Check local style, terminology, logical spacing, selective typing/documentation, and justified helper placement. Preserve existing work and behavior unless the task explicitly changes it.
