init:
	python3 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -e '.[data,dev]'

run:
	python -m meros.experiments.experiment_002

test:
	python -m pytest

serve-media:
	cd data/media && python -m http.server 8080 --bind 127.0.0.1
