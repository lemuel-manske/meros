# Meros

Computer-vision research for identifying individual Goliath Groupers (*Epinephelus itajara*) with Instituto Meros do Brasil.

## Setup

Run commands from the repository root. Python 3.12+ and Make are required.

```sh
make init

make data-pull
```

DVC access requires credentials for the configured Backblaze S3 remote. No credentials are stored here.

## Documentation

- [Project specification and research findings](SPEC.md)
- [Coordinate-only spot verification — 04 November 2026](docs/04_11_2026.md)

The retained E24 coordinate-only signature is exploratory and does not establish
validated automatic individual identification.
