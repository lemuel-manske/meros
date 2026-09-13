init-venv:
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip
	. .venv/bin/activate && pip install -r requirements.txt

extract-frames:
	python -m src.cmd.extract_frames

extract-tracks:
	python -m src.cmd.extract_tracks
