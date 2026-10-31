"""
One-time migration: move Luana's photos and annotations into Rogério's subject.

Run from the repository checkout. Preview is the default; --apply performs
the migration. Run on branch fix/merge-luana-rogerio after pulling DVC media.
The script does not update the DVC pointer; run 'make data-save' afterwards.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


SOURCE = "luana"
DESTINATION = "rogerio"
BRANCH = "fix/merge-luana-rogerio"
ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "data/media/subjects"
METADATA = ROOT / "data/metadata/subjects"
PER_IMAGE = ("eyes", "mouths", "spots", "spot_region_reviews")
CORRESPONDENCES = "spot_correspondences"


def image_files(subject: str) -> dict[int, Path]:
    """
    Discover available JPEG images, rejecting unexpected source file names.
    """
    directory = MEDIA / subject

    if not directory.is_dir():
        raise RuntimeError(f"Missing media directory: {directory}. Run 'make data-pull'.")

    images = {}

    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() == ".jpg" and path.stem.isdecimal():
            image_id = int(path.stem)

            if image_id in images:
                raise RuntimeError(f"Duplicate image ID in {directory}: {image_id}")

            images[image_id] = path
        elif subject == SOURCE:
            raise RuntimeError(f"Unexpected source media entry: {path}")

    return images


def metadata_ids(subject: str) -> set[int]:
    """
    Include existing annotation IDs to prevent collisions with sparse photo IDs.
    """
    subject_root = METADATA / subject
    ids = set()

    for category in PER_IMAGE:
        for path in (subject_root / category).glob("*.json"):
            if not path.stem.isdecimal():
                raise RuntimeError(f"Unexpected metadata filename: {path}")

            ids.add(int(path.stem))

    for path in (subject_root / CORRESPONDENCES).glob("*.json"):
        parts = path.stem.split("-")

        if len(parts) != 2 or not all(part.isdecimal() for part in parts):
            raise RuntimeError(f"Unexpected correspondence filename: {path}")

        ids.update(map(int, parts))

    return ids


def prepare() -> tuple[dict[int, int], list[tuple[Path, Path]], list[tuple[Path, Path, str]]]:
    """
    Validate all source files and destination paths before changing anything.
    """
    source_images = image_files(SOURCE)
    destination_images = image_files(DESTINATION)
    source_root = METADATA / SOURCE
    destination_root = METADATA / DESTINATION

    if not source_images or not source_root.is_dir() or not destination_root.is_dir():
        raise RuntimeError("Source images or subject metadata are missing.")

    source_ids = set(source_images)
    annotated_source_ids = metadata_ids(SOURCE)

    if source_ids != annotated_source_ids:
        raise RuntimeError(
            f"Source media/metadata IDs disagree: {source_ids} vs {annotated_source_ids}"
        )

    destination_ids = set(destination_images) | metadata_ids(DESTINATION)

    if not destination_ids:
        raise RuntimeError("Destination subject has no images or annotations.")

    if not destination_ids.issubset(destination_images):
        raise RuntimeError(
            "Destination has annotations without local JPEGs. Run 'make data-pull'."
        )

    start = max(destination_ids) + 1
    mapping = {
        old_id: start + offset
        for offset, old_id in enumerate(sorted(source_ids))
    }
    media_moves = [
        (source_images[old_id], MEDIA / DESTINATION / f"{new_id}.jpg")
        for old_id, new_id in mapping.items()
    ]

    metadata_moves: list[tuple[Path, Path, str]] = []
    permitted = set(PER_IMAGE) | {CORRESPONDENCES}

    for path in sorted(source_root.rglob("*")):
        if path.is_dir():
            if path.parent != source_root or path.name not in permitted:
                raise RuntimeError(f"Unexpected metadata directory: {path}")

            continue

        relative = path.relative_to(source_root)

        if len(relative.parts) != 2 or relative.parts[0] not in permitted:
            raise RuntimeError(f"Unexpected metadata entry: {path}")

        if path.suffix != ".json":
            raise RuntimeError(f"Unexpected metadata file: {path}")

        category = relative.parts[0]
        data = json.loads(path.read_text(encoding="utf-8"))

        if data.get("subject_id") != SOURCE:
            raise RuntimeError(f"Unexpected subject_id in {path}")

        data["subject_id"] = DESTINATION

        if category == CORRESPONDENCES:
            old_ids = data.get("image_ids")
            expected_name = "-".join(map(str, old_ids)) if isinstance(old_ids, list) else ""

            if expected_name != path.stem or len(old_ids) != 2:
                raise RuntimeError(f"Invalid correspondence image_ids: {path}")

            if not all(old_id in mapping for old_id in old_ids):
                raise RuntimeError(f"Unknown correspondence image_ids: {path}")

            new_ids = [mapping[old_id] for old_id in old_ids]
            data["image_ids"] = new_ids
            new_name = "-".join(map(str, new_ids)) + ".json"
        else:
            if not path.stem.isdecimal():
                raise RuntimeError(f"Unexpected image metadata filename: {path}")

            old_id = int(path.stem)

            if old_id not in mapping or data.get("image_id") != old_id:
                raise RuntimeError(f"Unexpected image_id in {path}")

            new_id = mapping[old_id]
            data["image_id"] = new_id
            new_name = f"{new_id}.json"

            if category == "spots":
                old_image = f"data/media/subjects/{SOURCE}/{old_id}.jpg"

                if data.get("image") != old_image:
                    raise RuntimeError(f"Unexpected image reference in {path}")

                data["image"] = f"data/media/subjects/{DESTINATION}/{new_id}.jpg"

            if category == "spot_region_reviews":
                for region in data.get("regions", []):
                    region_id = region.get("region_id", "")
                    old_prefix = f"{SOURCE}_{old_id}_"

                    if not region_id.startswith(old_prefix):
                        raise RuntimeError(f"Unexpected region_id in {path}: {region_id}")

                    region["region_id"] = f"{DESTINATION}_{new_id}_" + region_id[len(old_prefix):]

        destination_path = destination_root / category / new_name
        content = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
        metadata_moves.append((path, destination_path, content))

    if not metadata_moves:
        raise RuntimeError("No source metadata found.")

    destinations = [new for _, new in media_moves]
    destinations.extend(new for _, new, _ in metadata_moves)

    if len(destinations) != len(set(destinations)):
        raise RuntimeError("Multiple source files would use the same destination.")

    for path in destinations:
        if path.exists():
            raise RuntimeError(f"Destination already exists: {path}")

    return mapping, media_moves, metadata_moves


def check_checkout() -> None:
    """
    Require the dedicated branch and a clean Git worktree for a destructive change.
    """
    branch = subprocess.check_output(
        ["git", "-C", str(ROOT), "branch", "--show-current"], text=True
    ).strip()

    if branch != BRANCH:
        raise RuntimeError(f"Switch to branch {BRANCH!r} before applying (current: {branch!r}).")

    status = subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain"], text=True
    ).strip()

    if status:
        raise RuntimeError("Commit or stash existing Git changes before applying.")


def apply(
    media_moves: list[tuple[Path, Path]],
    metadata_moves: list[tuple[Path, Path, str]],
) -> None:
    """
    Copy all targets first; remove originals only after every copy succeeds.
    """
    created: list[Path] = []

    try:
        for source, destination in media_moves:
            destination.parent.mkdir(parents=True, exist_ok=True)
            created.append(destination)
            shutil.copy2(source, destination)

        for _, destination, content in metadata_moves:
            destination.parent.mkdir(parents=True, exist_ok=True)
            created.append(destination)
            destination.write_text(content, encoding="utf-8")

    except Exception:
        for path in reversed(created):
            path.unlink(missing_ok=True)

        raise

    for source, _ in media_moves:
        source.unlink()

    for source, _, _ in metadata_moves:
        source.unlink()

    for directory in sorted((METADATA / SOURCE).iterdir()):
        directory.rmdir()

    (METADATA / SOURCE).rmdir()
    (MEDIA / SOURCE).rmdir()


def main() -> None:
    """
    Show the image ID mapping; apply only when explicitly requested.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="perform the migration")
    arguments = parser.parse_args()

    mapping, media_moves, metadata_moves = prepare()

    print(f"Migration: {SOURCE} -> {DESTINATION}")
    print("Image IDs:", ", ".join(f"{old} -> {new}" for old, new in mapping.items()))
    print(f"JPEGs: {len(media_moves)}; annotation JSONs: {len(metadata_moves)}")

    for source, destination in media_moves:
        print(f"  {source.relative_to(ROOT)} -> {destination.relative_to(ROOT)}")

    if not arguments.apply:
        print("Preview only. Re-run with --apply to modify files.")
        return

    check_checkout()
    apply(media_moves, metadata_moves)
    print("Local migration completed.")
    print("Next: make data-save && make data-push")
    print("Review Git diff and DVC pointer before committing.")


if __name__ == "__main__":
    main()
