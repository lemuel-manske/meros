# Experiment results

This directory contains **generated artifacts** from Meros research experiments.

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

## Run an experiment

From the repository root:

```sh
make run EXPERIMENT=002

make run EXPERIMENT=003
```
