# Processing

Prepare and annotate observations from the project's media and metadata.

- `extract_*.py` modules generate frames, tracks, and masked crops.
- `annotate_*.py` modules provide interactive annotation tools.
- `common.py` provides shared video-processing helpers.

The pipeline controls stage order and completion checks. Persistence belongs
in adapters accessed through `Project`; individual-identity scoring belongs
in `meros.identification` rather than this package.
