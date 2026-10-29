# Meros

Computer-vision research for identifying individual Goliath Groupers (*Epinephelus itajara*) with Instituto Meros do Brasil.

## Setup

Run commands from the repository root. Python 3.12+ and Make are required.

The Makefile is the shared command entry point for local work, this README, and GitHub Actions. `make help` lists the targets. Commands use `.venv/bin/python` when available; override with `PYTHON=...` for another environment.

```sh
make init

make data-pull
```

DVC access requires credentials for the configured Backblaze S3 remote. No credentials are stored here. Evaluation of existing artifacts needs NumPy, OpenCV, and SciPy; it does not import Torch or SAM2.

For rebuilding tracking/masked crops, initialize the pinned SAM2 submodule, install Torch/Torchvision for your CUDA environment, and install SAM2:

```sh
make install-sam2
```

Obtain the official SAM2.1 Hiera Large checkpoint using the pinned submodule's documented checkpoint download instructions. Place it at `external/sam2/checkpoints/sam2.1_hiera_large.pt`. The predictor uses CUDA and the saved human-provided seed boxes.

Tracking reuses each video's boxes from `data/metadata/videos/bboxes.json`. When a video has no boxes (or the file is absent), OpenCV opens its first extracted frame for manual fish selection. Press `a`, drag a box around the fish, and press Enter or Space to confirm it. Confirmed boxes appear green with their fish IDs. Press `a` to add another fish and `q` to save and continue. Esc or closing the window cancels without saving. Fish IDs follow selection order: `0`, `1`, and so on.

Selected boxes are saved immediately to `bboxes.json`, preserving other videos' entries, so subsequent runs reuse them even if inference is interrupted. Finishing without selecting a fish stops the run without saving boxes. Manual selection requires a desktop display and the GUI-enabled `opencv-python` dependency.

`pyproject.toml` declares the runtime dependencies and optional development/data groups. Run provenance records the source commit. The full historical GPU environment has not been recreated.


## Development conventions

These rules apply across the Meros codebase.

- **Follow existing patterns.** Prefer the repository's current architecture, naming,
  and dependencies over introducing new frameworks or abstractions for a small change.
- **Separate code blocks with blank lines.** Keep a blank line between logical
  steps, branches, loops, and groups of statements whenever it improves readability.
  **When in doubt, always add a blank line.**
- **Keep experimental work lightweight.** Exploratory scripts should expose
  adjustable settings as module-level constants rather than add CLI flags
  unless a command-line interface is explicitly needed.
- **Keep filesystem access behind adapters.** Processing and application code
  should use `Project` and its media/metadata stores instead of constructing
  persistence paths directly. Adapter-specific conventions belong in that
  module's own rules.
- **Keep documentation durable.** Do not update READMEs for every experiment,
  prototype, parameter change, or temporary workflow. Update documentation
  when an architectural convention, public interface, or established workflow
  is intended to remain. A module README must contain rules relevant to that
  module; global rules belong here.
- **Avoid unnecessary validation and scaffolding.** Do not add speculative
  validation layers, extra abstractions, or unrelated cleanup when implementing
  a focused change.
- **Keep experimental changes isolated.** Do not alter established pipelines,
  unrelated scripts, or existing data unless the task calls for it. Before
  editing a shared branch, read its current files and preserve concurrent changes.
- **Tests are not required as permanent repository artifacts for now.** Temporary
  tests may be used during development, but do not add or retain test files
  unless that policy changes.
- **Run `make check` after code changes.** Report accurately when checks
  could not be executed; do not claim validation from code inspection alone.

## Validation

> `make test`, `make typecheck`, `make lint`, `make format-check`, and `make cli-check` can also run separately. `make format` applies the formatting rules.

```sh
make check
```
