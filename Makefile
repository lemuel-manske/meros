init-venv:
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip
	. .venv/bin/activate && pip install -r requirements.txt

extract-frames:
	python -m src.cmd.extract_frames

extract-tracks:
	python -m src.cmd.extract_tracks

extract-crops:
	python -m src.cmd.extract_crops

extract-masked-crops:
	python -m src.cmd.extract_masked_crops

build-review:
	python -m src.cmd.build_review
