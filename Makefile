.DEFAULT_GOAL := help

VENV ?= .venv
PYTHON ?= $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)
EXTRAS ?= data,dev
PORT ?= 8080
BIND ?= 127.0.0.1

.PHONY: help init install install-sam2 sam2-submodule
.PHONY: data-pull data-save data-push results-save pipeline subjects
.PHONY: typecheck lint format format-check check serve-media

help:
	@printf '%s\n' \
		'Setup: init, install, install-sam2' \
		'Data: data-pull, data-save, data-push, results-save RUN_DIR=...' \
		'Images: subjects' \
		'Video: pipeline (optional processing)' \
		'Checks: typecheck, lint, format, format-check, check' \
		'Preview: serve-media (PORT=... BIND=...)'

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

data-pull:
	$(PYTHON) -m dvc pull data/media.dvc

data-save:
	$(PYTHON) -m dvc add data/media

data-push:
	$(PYTHON) -m dvc push

results-save:
	@test -n "$(RUN_DIR)" || { printf '%s\n' 'Set RUN_DIR to a completed run.'; exit 2; }
	$(PYTHON) -m dvc add "$(RUN_DIR)"

subjects:
	$(PYTHON) -m meros.subjects

pipeline:
	$(PYTHON) -m meros.pipeline.engine

typecheck:
	$(PYTHON) -m pyright

lint:
	$(PYTHON) -m ruff check src

format:
	$(PYTHON) -m ruff format src

format-check:
	$(PYTHON) -m ruff format --check src

check: typecheck lint format-check

serve-media:
	$(PYTHON) -m http.server $(PORT) --bind $(BIND) --directory data/media
