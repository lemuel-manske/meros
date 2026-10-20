# Experiment NNN — Short descriptive title

## Question and scope

> State the precise research question.

Describe the target features, intended comparison, and what this experiment
**does not** attempt to establish.

## Data and selection criteria

Document where images, tracks, individuals, and annotations come from.

Specify any viewpoint, quality, frame-selection, identity-labeling, or pose
requirements. Distinguish same-video comparisons from independent encounters.

## Representations and processing

Describe every input representation and all preprocessing stages.

Explain segmentation, masking, alignment, enhancement, or spot detection, as
applicable. Record any important implementation decisions or parameters.

## Matching method

Describe the features/descriptors, candidate matching, geometric verification,
thresholds, and known algorithmic assumptions.

## Measurements and outputs

Define the metrics precisely, including undefined or missing cases.

List generated outputs (e.g. `metrics.csv`, `run.json`, plots, visualizations,
and representations) and explain how to inspect them.

## Running the experiment

Run from the repository root:

```sh
make run EXPERIMENT=NNN
```

State required prepared data, installation/setup requirements, and any
experiment-specific arguments. Runs should use `Project.open()` and write
under `results/NNN/` using the shared output helper.

## Observations and results

Distinguish **historical observations** from **reproduced measurements on the
current manifest**. Include sample sizes, pair labels, provenance, and exact
figures when available. Do not invent results.

## Interpretation and limitations

State which claims are supported, which are not, and how recording conditions,
pose, segmentation, sample size, or missing repeat encounters affect conclusions.

## Next steps

List specific changes or additional comparisons needed to answer the question.

## References

Link papers, implementations, and relevant project materials. Omit only when
no references are used.
