"""Copy verified legacy 002 composites into selection-scoped storage; keep originals."""

import argparse
import hashlib
from pathlib import Path

from meros import media, metadata
from meros.adapters.fs_storage import write_json_atomic


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def migrate(manifest: Path, receipt: Path):
    metadata.individuals_path = manifest

    operations = []

    for individual in metadata.read_individuals():
        for selection in individual.tracks:
            # Legacy alignment metadata must describe this exact interval/reference.
            alignment = metadata.read_alignment(selection.id)

            if alignment.reference_frame != selection.reference_frame:
                raise ValueError("Legacy alignment reference differs")

            folder = Path("data/media/alignment") / selection.video_id / selection.track_id

            for suffix, target in [
                ("median", media.paths.composite),
                ("median_enhanced", media.paths.composite_enhanced),
            ]:
                source = folder / f"{selection.reference_frame}_{suffix}.png"

                destination = target(
                    selection.video_id,
                    selection.track_id,
                    selection.reference_frame,
                    selection_id=selection.selection_id,
                )

                if not source.is_file():
                    raise FileNotFoundError(source)

                # Validate PNG and alpha before admitting it into a run's evidence.
                image = media._read_image(source, -1)

                if image.ndim != 3 or image.shape[2] != 4:
                    raise ValueError(f"Expected BGRA: {source}")

                checksum = digest(source)

                if destination.exists() and digest(destination) != checksum:
                    raise FileExistsError(f"Different artifact already exists: {destination}")

                operations.append((source, destination, checksum))

    rows = []

    for source, destination, checksum in operations:
        destination.parent.mkdir(parents=True, exist_ok=True)

        if not destination.exists():
            # Exclusive creation; a failed copy is removed rather than accepted later.
            try:
                with destination.open("xb") as stream:
                    stream.write(source.read_bytes())

                if digest(destination) != checksum:
                    raise RuntimeError(f"Checksum differs: {destination}")
            except BaseException:
                destination.unlink(missing_ok=True)

                raise

        rows.append({"source": str(source), "destination": str(destination), "sha256": checksum})

    write_json_atomic(
        receipt, {"manifest": str(manifest), "copies": rows, "originals_retained": True}
    )

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--manifest", type=Path, default=Path("experiments/002/selections.json"))

    parser.add_argument("--receipt", type=Path, default=Path("results/002/migration.json"))

    args = parser.parse_args()

    print(
        f"Copied/verified {len(migrate(args.manifest, args.receipt))} artifacts; originals retained"
    )


if __name__ == "__main__":
    main()
