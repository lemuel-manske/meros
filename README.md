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

Obtain the official SAM2.1 Hiera Large checkpoint using the pinned submodule's documented checkpoint download instructions. Place it at `external/sam2/checkpoints/sam2.1_hiera_large.pt`. The current wrapper uses CUDA. Manual seed selection requires a desktop OpenCV session; the default uses saved human-provided seed boxes.

The historical full environment freeze is retained at `experiments/002/requirements-baseline.txt`; it is evidence, not the fresh-install recipe. New run manifests record actual library versions and source/image hashes. This PR does not claim the complete historical GPU environment was recreated.

## Run experiment 002

To reuse old composites, first copy them into selection-scoped paths with validation and a checksum receipt. Originals are retained:

```sh
make migrate

make evaluate
```

To rebuild all required representations and evaluate:

```sh
make run
```

Pass CLI options through `ARGS`, for example `make evaluate ARGS="--manifest path/to/selections.json --output results/002/my-run"`. The default is the frozen `experiments/002/selections.json`. `--cross-video-only` excludes same-video comparisons. `--output results/002/my-run` sets an explicit destination; existing run directories cannot be overwritten.

`make diagnostics` saves optional previews and match drawings. `make diagnostics ARGS="--force"` also recreates upstream previews when preparation is already cached. Deleting previews does not invalidate computed outputs.

Each successful run contains `metrics.csv`, `run.json`, and the exact 21 PNG representations used for the current seven selections. The current manifest produces 63 rows: 21 pairs × three representations. Same-video comparisons are explicitly labeled. Zero inliers when `ransac_attempted` is false means geometric verification was skipped.

## Processing and persistence

| Step | Required saved output | Optional output |
|---|---|---|
| Frame extraction | Frame cache for SAM2 | None |
| Tracking | Metadata with completed frame count | Tracking previews |
| Masked crops | Reusable BGRA crop cache | None |
| Alignment | Transform/statistics metadata | Aligned PNGs and overlays |
| Median composite | Selection-scoped composite | None |
| Enhancement | Enhanced composite | Side-by-side preview |
| Pair evaluation | Metrics, provenance, exact compared representations | Match drawings |

Aligned crops are reconstructed from transforms in memory during composite construction. They do not need to be saved. Original videos remain the source data. Cache fingerprints include source contents, mirroring, seeds, checkpoint contents, implementation identity, and processing settings; changing code conservatively invalidates preparation.

Media remains inside the existing DVC-managed `data/media` tree to avoid a blind migration of unavailable media. After producing new artifacts, explicitly version them with `make data-save` and `make data-push`. Run evidence under `results/` is excluded from Git; preserve it with `make results-save RUN_DIR=results/002/<run-id>` and `make data-push`, then commit the pointer. Pipeline fingerprints under `data/.pipeline` are local disposable state.

## Code and metadata

- `src/meros/domain`: records and storage contracts.
- `src/meros/adapters`: filesystem paths and serialization.
- `src/meros/processing`: image-processing steps and reusable matcher.
- `src/meros/pipeline`: stage orchestration and invalidation.
- `src/meros/experiments`: evaluation and reversible legacy migration.
- `experiments/002`: preserved selections, historical notes, and baseline provenance.

An **individual** is a biological identity; a **track** is local to a video; a **TrackSelection** is an inclusive interval plus a reference frame. Derived outputs are keyed by the entire selection. The existing identity JSON format remains readable, and historical alignment `candidate` statuses are translated to `accepted` when read. Source annotation CSVs are retained unchanged because their old track mapping is unresolved.

The broader research objectives remain in `SPEC.md`. See `docs/engineering-review.md` for the review of the pre-refactor revision and `docs/migration.md` for compatibility details.

## Validation

```sh
make check
```

`make test`, `make lint`, `make format-check`, and `make cli-check` can also run separately. `make format` applies the formatting rules. GitHub Actions uses `make install EXTRAS=dev` followed by the same `make check` target.

Tests cover tracking reruns, incomplete caches, selection isolation, optional diagnostics, transparency-aware aggregation, and structured evaluation. Real media/GPU reproduction must be performed separately; synthetic checks do not validate biological identification performance.
