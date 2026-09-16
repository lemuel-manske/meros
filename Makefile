init:
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip
	. .venv/bin/activate && pip install -r requirements.txt

serve-media:
	cd data/media && python -m http.server 8080 --bind 0.0.0.0
