.DEFAULT_GOAL := help


# Environment and installation.

VENV ?= .venv

PYTHON ?= $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)

EXTRAS ?= data,dev


# Optional arguments and artifact locations.

ARGS ?=

RUN_DIR ?=


# Local media preview server.

PORT ?= 8080

BIND ?= 127.0.0.1


.PHONY: help init install install-sam2 sam2-submodule
.PHONY: data-pull data-save data-push results-save
.PHONY: run evaluate diagnostics
.PHONY: test lint format format-check cli-check check serve-media


# Setup.

help:
	@printf '%s\n' \
		'Setup:       init, install, install-sam2' \
		'Data:        data-pull, data-save, data-push, results-save RUN_DIR=...' \
		'Experiment:  run, evaluate, diagnostics (optional ARGS="...")' \
		'Checks:      test, lint, format, format-check, cli-check, check' \
		'Preview:     serve-media (optional PORT=... BIND=...)'


init:
	python3 -m venv $(VENV)

	$(VENV)/bin/python -m pip install --upgrade pip

	$(MAKE) install PYTHON=$(VENV)/bin/python


install:
	$(PYTHON) -m pip install -e '.[$(EXTRAS)]'


sam2-submodule:
	git submodule update --init external/sam2


install-sam2: sam2-submodule
	$(PYTHON) -m pip install -e external/sam2


# Data and durable experiment artifacts.

data-pull:
	$(PYTHON) -m dvc pull data/media.dvc


data-save:
	$(PYTHON) -m dvc add data/media


data-push:
	$(PYTHON) -m dvc push


results-save:
	@test -n "$(RUN_DIR)" || { printf '%s\n' 'Set RUN_DIR to a completed experiment run.'; exit 2; }

	$(PYTHON) -m dvc add "$(RUN_DIR)"


# Experiment 002.

run:
	$(PYTHON) -m meros.experiments.experiment_002 $(ARGS)


evaluate:
	$(PYTHON) -m meros.experiments.experiment_002 --evaluate-only $(ARGS)


diagnostics:
	$(PYTHON) -m meros.experiments.experiment_002 --diagnostics $(ARGS)


# Tests, code style, and CPU CLI checks.

test:
	$(PYTHON) -m pytest $(ARGS)


lint:
	$(PYTHON) -m ruff check src tests


format:
	$(PYTHON) -m ruff format src tests


format-check:
	$(PYTHON) -m ruff format --check src tests


cli-check:
	$(PYTHON) -m meros.experiments.experiment_002 --help


check: test lint format-check cli-check


# Local media preview.

serve-media:
	$(PYTHON) -m http.server $(PORT) --bind $(BIND) --directory data/media
