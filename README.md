# Meros

Computer-vision research for identifying individual Goliath Groupers (*Epinephelus itajara*) with Instituto Meros do Brasil. Experiment 002 compares reference masked crops, median composites, and CLAHE-enhanced composites using SIFT and affine RANSAC. Its historical observations are promising but preliminary; they do not demonstrate pose-independent identification.

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

`pyproject.toml` declares the runtime dependencies and optional development/data groups. Run provenance records the source commit. The full historical GPU environment has not been recreated.

## Run experiment 002

Build the required representations and evaluate using the current organization:

```sh
make run
```

`make run` has one objective: prepare the representations and evaluate all pairs. Experiment 002 uses `experiments/002/selections.json` and writes a new directory under `results/002/`. Settings belong in the experiment code.

The shared runner loads `meros.experiments.experiment_<id>`. Use `EXPERIMENT=003` for a future `experiment_003.py`; each module provides `main()`.

Each successful run contains `metrics.csv`, `run.json`, and the exact 21 PNG representations used for the current seven selections. The current manifest produces 63 rows: 21 pairs × three representations. Same-video comparisons are explicitly labeled. Zero inliers when `ransac_attempted` is false means geometric verification was skipped.

## Processing and persistence

| Step | Saved output |
|---|---|
| Frame extraction | Frames for SAM2 |
| Tracking | Metadata with completed frame count |
| Masked crops | Reusable BGRA crops |
| Alignment | Transform/statistics metadata |
| Median composite | Selection-scoped composite |
| Enhancement | Enhanced composite |
| Pair evaluation | Metrics, source commit, exact compared representations |

Preparation runs these steps in order. Completion checks reuse valid outputs; rebuilding a step rebuilds all later steps. Alignment transforms are applied in memory during composite construction.

After editing raw videos, mirroring, algorithms, or the model, delete the affected generated outputs before running again. Code and input hashing are absent from this workflow.

Selection manifests live with their experiments and are chosen explicitly when constructing a project. Generated tracking/alignment metadata is a disposable cache excluded from Git; human annotations and video/seed manifests remain versioned.

Raw videos and generated media are stored in the DVC-managed `data/media` tree. After producing new artifacts, explicitly version them with `make data-save` and `make data-push`. Run evidence under `results/` is excluded from Git; preserve it with `make results-save RUN_DIR=results/002/<run-id>` and `make data-push`, then commit the pointer.

## Code and metadata

- `src/meros/domain`: records.
- `src/meros/adapters`: filesystem paths and serialization.
- `src/meros/processing`: image-processing steps and reusable matcher.
- `src/meros/pipeline`: ordered preparation.
- `src/meros/experiments`: experiment evaluation.
- `experiments/002`: the current selection manifest and experiment notes.

An **individual** is a biological identity; a **track** is local to a video; a **TrackSelection** is an inclusive interval plus a reference frame. Derived outputs are keyed by the entire selection. Alignment metadata uses the `accepted` status and includes the reference frame in its path. The current organization is the supported workflow; earlier implementations and layouts are available in Git history.

The broader research objectives remain in `SPEC.md`. Experiment findings and limitations are documented in `experiments/002/README.md`.

## Validation

```sh
make check
```

`make test`, `make typecheck`, `make lint`, `make format-check`, and `make cli-check` can also run separately. `make format` applies the formatting rules.

Code follows the spacing used on `main`: separate import groups, setup, calculations, control-flow blocks, and returns with blank lines. Keep Makefile target groups and workflow steps visually separated too. GitHub Actions uses `make install EXTRAS=dev` followed by the same `make check` target.

`make typecheck` runs Pyright over production code and tests, including required parameter annotations. SAM2 is isolated behind a typed predictor contract; processing uses ordinary NumPy masks without conditional typing imports.

Tests use temporary project roots, real encoded videos, PNG/JPEG files, metadata serialization, SIFT, transforms, aggregation, enhancement, and pair evaluation. A readable recorded predictor replaces only GPU segmentation through its public API; tests do not patch globals. Every stage's completion check is exercised with valid, missing, corrupt, and inconsistent outputs. Numerical comparisons load the pre-refactor algorithms from Git history.

Real media/GPU reproduction must be performed separately; these checks do not validate biological identification performance.
