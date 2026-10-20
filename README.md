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

## Validation

> `make test`, `make typecheck`, `make lint`, `make format-check`, and `make cli-check` can also run separately. `make format` applies the formatting rules.

```sh
make check
```
