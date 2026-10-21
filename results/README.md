# Experiment results

This directory contains **generated artifacts** from Meros research experiments.

Experiment definitions, methodology, observations, and interpretation belong in
[`experiments/`](../experiments/). Use
[`experiments/TEMPLATE.md`](../experiments/TEMPLATE.md) when documenting a new experiment.

## Directory layout

Each run writes to a unique timestamped directory:

```text
results/
├── README.md
├── 002/
│   └── YYYYMMDDTHHMMSSZ-<id>/
│       ├── metrics.csv
│       ├── run.json
│       ├── plot.png
│       ├── enhancements.png
│       └── representations/
└── 003/
    └── YYYYMMDDTHHMMSSZ-<id>/
        ├── metrics.csv
        ├── run.json
        └── *--spots.png / *--vs--*.png
```

Actual outputs are determined by each experiment's implementation. Consult
[`experiments/002/README.md`](../experiments/002/README.md) or
[`experiments/003/README.md`](../experiments/003/README.md) for details.

## Run an experiment

From the repository root:

```sh
make run EXPERIMENT=002

make run EXPERIMENT=003
```

Both experiments use `Project.open()` to access the project metadata and media.
They generate distinct output directories automatically. Exp-002 also runs the
processing pipeline; Exp-003 expects the required masked crops to have been prepared.

## Interpret results

`metrics.csv` contains pairwise matching diagnostics. `run.json` records
provenance and experiment-specific parameters. Image files are intended for
visual inspection of the representations, detected spots, and correspondences.

A high match count or inlier ratio **is not** a probability of individual identity.
The data and matching conditions must be considered before drawing conclusions.

## Retention and reproducibility

Generated results are ignored by Git by default. This README is an exception.

To preserve a selected run through the repository's DVC workflow:

```sh
make results-save RUN_DIR=results/003/<run-directory>
```

Review the generated DVC metadata and store the associated data using the
project's configured DVC remote. Do not commit bulk experiment images directly
to Git. Keep the source commit and selection metadata needed to reproduce a run.

The template in `experiments/TEMPLATE.md` is for **experiment documentation**,
not for automatically generated run folders.
