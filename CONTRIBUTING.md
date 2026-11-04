# Contributing to Meros

Meros is a research project. Keep the working codebase small while preserving the experiments that led to it.

This document describes how research work is saved and consolidated.

Read [`AGENTS.md`](AGENTS.md) for coding conventions and [`SPEC.md`](SPEC.md) for the scientific scope.

## Working on experiments

- Start a dedicated branch from the current `main`, such as `experiment/coordinate-verification`. Do not use `main` as an experimental scratchpad.

- Make ordinary Git commits as you investigate. Commit meaningful failed attempts **before** deleting or replacing their code if they need to remain recoverable.
  Do not squash or rewrite the research history before archiving it.

- Keep exploratory scripts and temporary dependencies off `main`.

- Record important conclusions in commit messages or the annotated checkpoint tag. There is no required experiment manifest, index, or per-attempt Markdown document.

## Saving results

A checkpoint can recover only what was actually saved.

- Keep source code, configuration, small result summaries, and other lightweight records in Git.

- Use the existing DVC storage for large or numerous outputs, including diagnostic images, models, and bulk result tables.

- Before archiving a DVC-backed result directory, run:

  ```sh
  make results-save RUN_DIR=results/<campaign>
  git add results/<campaign>.dvc results/.gitignore
  git commit -m "chore(research): track campaign artifacts"
  make data-push
  ```

  Adjust the paths staged with `git add` to the files actually created or changed by DVC. `make results-save` tracks the directory but does **not** upload its content; `make data-push` does.

- Do not rely on ignored local files, CI artifacts, or a Git tag alone to preserve generated outputs. Check that required DVC objects are available remotely before publishing the checkpoint.

## Creating a research checkpoint

After the experimental work is complete:

1. Commit all files needed to understand and recover the campaign. Ensure the working tree is clean.
2. Push the research branch and any required DVC objects.
3. Create and publish an **annotated Git tag** at the final research commit.

```sh
git status --short
git push -u origin HEAD
git tag -a research/<topic>/YYYY-MM-DD -m "Short summary of the campaign and outcome"
git push origin research/<topic>/YYYY-MM-DD
```

Use a short descriptive topic and the checkpoint date; if multiple checkpoints share a date, add a meaningful suffix. **Published research tags are permanent checkpoints by convention:** never move, reuse, delete, or force-update them.

The Git tag preserves the commit and its reachable history, including earlier committed attempts. DVC preserves the content of tracked artifacts **only while the referenced remote objects are retained**. Do not run DVC garbage collection that would remove objects needed by historical checkpoints.

The repository's Git tags are the research checkpoint catalog. No central list of historical commits or experiment files is required in `SPEC.md` or `README.md`. GitHub Releases may be added for human-readable notes, but are optional.

## Consolidating the outcome

After tagging, prepare a separate, focused change against `main`:

- Bring over only code worth maintaining, in the appropriate existing module.

- Remove or leave behind experiments, abandoned implementations, temporary runners, and bulky intermediate records.

- Update durable documentation only if the project's behavior, interface, or scientific understanding changes materially.

- Keep the experimental branch until the tag and DVC recovery have been verified. It may then be deleted without losing the tagged Git history.

Do not merge an entire experimental branch into `main` merely to preserve its history. The checkpoint serves that purpose.

## Recovering an old checkpoint

```sh
git fetch --tags
git worktree add ../meros-research research/<topic>/YYYY-MM-DD
cd ../meros-research
make data-pull
```

`make data-pull` restores the project's tracked media. If the checkpoint has other DVC-tracked results, also run `dvc pull` for those outputs (or `dvc pull` for all tracked DVC data). The checkpoint's dependencies and environment may need to be recreated to rerun its historical code.

The goal is simple: **`main` represents the maintained present, annotated tags preserve past research, and DVC stores the associated large artifacts.**
