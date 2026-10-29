# Meros

Computer-vision research for identifying individual Goliath Groupers (*Epinephelus itajara*) with Instituto Meros do Brasil.

## Subject photographs

The primary data model is a **subject** (individual) with numbered head images:

```text
data/media/subjects/
├── athila/
│   ├── 0.jpg
│   └── 1.jpg
└── maria/
    ├── 0.jpg
    └── 1.jpg
```

The directory name is the subject identifier. The basename of each numbered
JPG is its image identifier. Lists are sorted numerically.

Use `Project.open().media.list_subjects()` to find subjects and their photos;
`list_subject_images("athila")` for one individual; and
`read_subject_image("athila", 0)` to read OpenCV BGR pixels.
The application accesses files through the adapter, not directly.

These photographs live under `data/media` (DVC managed). Run
`make data-pull` if they have been uploaded to the DVC remote. Spot-center
annotations for subjects are not yet created; they will be extracted next.

## Setup

Requires Python 3.12+, Make, and packages declared in `pyproject.toml`.
From the repository root:

```sh
make init
make data-pull
make subjects
make check
```

`make subjects` lists locally available individuals/images. The Makefile
uses `.venv/bin/python` when available, or set `PYTHON=...`.

## Architecture

- `src/meros/domain/`: entities including `Subject` and `SubjectImage`, plus retained video types.
- `src/meros/adapters/`: filesystem access for photographs, legacy frames, tracking, and metadata.
- `src/meros/project.py`: application dependency wiring.
- `src/meros/subjects.py`: inspect the subject-photo dataset.
- `src/meros/processing/`: optional video extraction/tracking/masked crops.
- `src/meros/pipeline/`: video preparation orchestration.
- `src/meros/external/`: optional SAM2 predictor integration.

The former `sandbox/` and `experiments/` packages have been removed.
Subject images are the primary input; the video preparation pipeline is
retained for generating further media if needed via `make pipeline`.

SAM2 requires its pinned submodule, a compatible Torch/CUDA installation,
and the official SAM2.1 Hiera Large checkpoint. Subject listing does not
require SAM2 or GPU setup.

## Validation

Run `make check` for type checking, linting, formatting checks, and a
CPU-only subject import smoke test. Run `make format` to apply formatting.
