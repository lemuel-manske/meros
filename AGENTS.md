# Meros — Code Style and Agent Guidance

## Purpose

Code contributed to Meros should read like the surrounding project code and reflect the author's editing preferences.

This document is based on the comparison between the staged version of `src/cmd/build_review_manifest.py` and the author's working-tree edits, including the new `src/fn.py`. The edited code provides the clearest evidence of the preferred style. Existing code provides additional context.

The central preference is **explicit structure, visual separation between logical steps, and limited explanatory overhead**.

## Scope of changes

Preserve the behavior of existing code when making style changes. A formatting preference does not justify changing sampling logic, validation, output fields, defaults, or error handling.

When working in a file with local edits, preserve the author's changes and follow their style. Do not automatically reformat unrelated code or replace local conventions with a formatter's defaults.

These guidelines describe project-owned Python code. They do not require restyling the external SAM2 dependency.

## Imports and module organization

### Import groups

Separate direct standard-library imports, standard-library `from` imports, and project imports with blank lines.

For example:

```python
import argparse
import csv

from pathlib import Path
from typing import TypedDict

from src.artifacts import get_vids
from src.consts import TRACKS_FOLDER
from src.fn import positive_int
from src.metadata import (
    get_track_metadata,
    get_vis_metadata,
)
```

Use parenthesized imports with one name per line and a trailing comma when importing several names from the same project module. Keep single-name imports on one line.

### Module layout

Keep imports at the beginning of command modules. Follow them with relevant data structures, constants, processing functions, and the script entry point.

The edited command removes its opening module docstring, including usage instructions and implementation notes. Do not add a long documentation preamble to every command. Put extended usage explanations in project documentation when needed.

### Shared helpers

Place small general-purpose helpers in an appropriate shared module. The author moved `positive_int` from the command into `src/fn.py` and imported it through `from src.fn import positive_int`.

Keep command-specific processing in the command module. Functions such as `build_review_rows`, its sampling logic, and review-manifest writing belong in `src/cmd/build_review_manifest.py` because they serve that command. A command can contain data structures and substantial processing functions; it does not need to be a thin wrapper.

Extract a helper into a regular module under `src/` only when there is a concrete shared responsibility or established reuse. Artifact paths belong in `artifacts.py`, metadata access belongs in `metadata.py`, and general-purpose helpers can belong in `fn.py`. Possible future reuse alone does not justify a new abstraction or module.

Keep useful configuration in function parameters with defaults rather than exposing command-line options before they are needed.

## Function definitions and type annotations

### Function signatures

For multiline function definitions, place each parameter on its own line and include a trailing comma after the final parameter. Keep the return annotation on the closing-parenthesis line.

```python
def select_frames(
    rows: list[ReviewRow],
    samples: int,
    min_frame_gap: int,
) -> list[ReviewRow]:
```

Short definitions can remain on one line, as in `def positive_int(value: str) -> int:`.

This preference is specific to definitions and suitable multiline constructs. The edited code retains compact calls with several arguments on a line, including `parser.add_argument(...)` and the call to `build_review_rows(...)`. Do not expand every call mechanically.

### Type information

Keep parameter and return annotations and use `TypedDict` for named record structures. Use built-in generic notation such as `list[ReviewRow]`.

Local annotations should be selective. The author changed:

```python
groups: dict[str, list[ReviewRow]] = {}
row: ReviewRow = {
```

to plain assignments:

```python
groups = {}
row = {
```

The author retained `spaced: list[ReviewRow] = []`. Therefore, do not interpret these edits as a prohibition on local annotations. Prefer plain assignments for straightforward local dictionaries; retain annotations where they help explain a collection or satisfy a meaningful typing requirement.

## Whitespace and visual structure

Blank lines are used to separate **individual logical steps**, including steps inside functions and control-flow blocks.

### Setup and processing

Separate initialization from the loop or operation that follows it:

```python
spaced: list[ReviewRow] = []

for row in sorted(rows, key=lambda row: row["frame_idx"]):
    ...
```

Separate distinct setup steps as well. In the edited code, loading and sorting intervals, initializing `previous_end`, and entering the loop are three visually separate steps.

### Validation and early exits

Separate consecutive guard clauses with a blank line. Separate the final guard from the function's concluding return.

When an early exit follows a state update, put a blank line before the exit:

```python
if annotation is None:
    unannotated += 1

    continue
```

The same structure appears in the helper:

```python
number = int(value)

if number < 1:
    raise argparse.ArgumentTypeError("must be a positive integer")

return number
```

Keep a guard and its immediate `raise` or `return` together when there is no preceding operation inside the branch.

### Intermediate results

Use a blank line after a multiline assignment before inspecting or consuming its result. This applies to the annotation lookup, the row dictionary, and the selected-frame comprehension.

Likewise, separate path construction from an existence check, and separate validation from updates to loop state.

### Related operations

Keep tightly related statements together. The author retained adjacent counter initializations:

```python
missing_crops = 0
unannotated = 0
```

The writer setup and write operations also remain together:

```python
writer = csv.DictWriter(output, fieldnames=FIELDS)
writer.writeheader()
writer.writerows(rows)
```

Use blank lines according to meaning rather than inserting one after every statement.

### Script blocks

Separate parser creation, argument definitions, argument parsing, processing, and the final status message.

The multiline `--video-id` argument definition is separated from the following group of short argument definitions. The short definitions remain adjacent.

The edited code also includes a blank line before `except`, separating the successful write block from error handling.

Retain the usual two blank lines between top-level functions and before the script entry point. The edited import block has one blank line before `ReviewRow`; preserve nearby spacing without treating that single occurrence as a universal top-level spacing rule.

## Comments and documentation

### Comments

Use lowercase prose in ordinary code comments. The author changed sentence-initial capitals and the acronyms `ID` and `CSV` to lowercase in comments, while keeping periods.

```python
# the current crop extractor saves only object 1 in a per-video folder.
# refuse multi-object metadata because interval labels have no track id.
```

Keep comments that explain a constraint or a reason for a decision. Avoid narrating operations that are already clear from the code.

A standalone explanatory comment block can have a blank line before and after it. A short comment about the immediately following operation can stay attached to that operation:

```python
# exclusive creation protects any human review work in an existing csv.
with args.output.open("x", newline="", encoding="utf-8") as output:
    ...
```

The lowercase preference applies to comments. Preserve normal capitalization in names, error messages, help text, and printed output.

### Docstrings

Do not add a docstring to every function by default. The author removed both processing-function docstrings, including the longer explanation of the sampling algorithm.

Short useful helper docstrings are welcome. The extracted `positive_int` helper gained a concise description with opening and closing triple quotes on separate lines:

```python
def positive_int(value: str) -> int:
    """
    Convert a string to a positive integer.
    """
    number = int(value)
```

Use this multiline form when adding a docstring, even for a single sentence. Begin the sentence with a capital letter and finish it with a period. There is no extra blank line between this docstring and the first statement.

Removing documentation can affect behavior when code uses `__doc__`, as the argument parser currently does. Consider that dependency before changing documentation; do not infer that removing CLI help content is a general style requirement.

## Command entry points

Commands should run with a single invocation such as `python -m src.cmd.build_review_manifest`. Use defaults and the configured videos; do not add argument parsers or command-line options unless requested.

For simple command scripts, put orchestration directly under:

```python
if __name__ == "__main__":
    ...
```

The author removed the `main()` wrapper and moved its body into this guard. Follow that structure for similar commands. Keep command-specific functions above the guard in the same file, and import only helpers with a concrete reason to live in shared modules.

Do not introduce an entry-point wrapper solely as boilerplate. A callable entry point can still be appropriate when an actual integration requires one.

### Artifact paths

Use helpers from `src.artifacts` to locate files. For example, use `path_to_track(video_id, frame_idx)` instead of constructing a path from `TRACKS_FOLDER`. Add a suitable path helper when an artifact has no existing helper. Commands should not need to know the storage layout.

### Console output

Print only information useful to the person running the command. A final message with the result count and output path is sufficient for the review manifest. Avoid routine per-frame or per-video progress messages, and keep reusable processing functions quiet unless a diagnostic is necessary.

## Expressions and conventions retained in the edit

The author's changes preserve several existing conventions:

- four spaces for indentation;
- double-quoted strings and f-strings;
- `snake_case` functions and variables;
- `PascalCase` record types and uppercase module constants;
- trailing commas in multiline imports, parameter lists, and dictionaries;
- comprehensions, generator expressions, and short sorting lambdas;
- early returns and `continue` statements;
- `Path` objects for path composition;
- adjacent f-strings for multiline status messages.

These retained choices provide context, but they do not establish an exact line-length limit or a requirement to use a particular formatter. Preserve readable expressions, including the longer conditions and error messages retained in the edited command.

## Review before completing a change

Before presenting code, check that:

1. Function definitions, import groups, and script entry points follow the local structure.
2. Blank lines separate setup, validation, state changes, and final results.
3. Closely related operations remain grouped.
4. Comments explain reasons in lowercase prose.
5. Docstrings are short, selective, and formatted like the shared helper.
6. Type annotations describe interfaces without unnecessarily repeating local dictionary structure.
7. Shared helpers live in an appropriate existing module.
8. Style edits preserve behavior and the author's existing work.

The objective is code that follows the author's demonstrated choices at both the structural and line-by-line level. Apply these conventions with the same selectivity shown in the edits.
