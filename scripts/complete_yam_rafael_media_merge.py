"""
Finish the DVC image move after Yam's annotations have been merged into Rafael.

Run from the repository root with DVC media present. Dry-run by default.
After --apply, run 'make data-save', 'make data-push', then commit data/media.dvc.
"""

from __future__ import annotations

import argparse

from meros.project import Project


IMAGE_IDS = {0: 2, 1: 3, 2: 4}


def run(*, apply: bool) -> None:
    """
    Move the three original Yam images without overwriting existing images.
    """
    project = Project.open()
    movements = []

    for old_id, new_id in IMAGE_IDS.items():
        source = project.media.subject_image_path("yam", old_id)
        target = project.media.subject_image_path("rafael", new_id)
        annotation = project.metadata.paths.subject_spots("rafael", new_id)
        old_annotation = project.metadata.paths.subject_spots("yam", old_id)

        if not annotation.is_file() or old_annotation.exists():
            raise RuntimeError(
                f"Metadata not migrated for rafael/{new_id}; "
                "switch to fix/merge-yam-into-rafael first."
            )

        if source.exists() == target.exists():
            raise RuntimeError(
                f"Expected exactly one image at {source} or {target}. "
                "If neither exists, pull the DVC media first."
            )

        movements.append((source, target))

    extra_images = {
        image.image_id for image in project.media.list_subject_images("yam")
    } - set(IMAGE_IDS)

    if extra_images:
        raise RuntimeError(f"Unexpected Yam images: {sorted(extra_images)}")

    for source, target in movements:
        if source.is_file():
            print(f"{'MOVE' if apply else 'WOULD MOVE'} {source} -> {target}")

            if apply:
                target.parent.mkdir(parents=True, exist_ok=True)
                source.rename(target)
        else:
            print(f"ALREADY MOVED {target}")

    if apply:
        source_dir = movements[0][0].parent

        if source_dir.exists() and not any(source_dir.iterdir()):
            source_dir.rmdir()

        print("Next: make data-save && make data-push")
        print("Then: git add data/media.dvc && git commit -m 'data: merge Yam into Rafael'")
    else:
        print("Run with --apply to move the images.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    run(apply=args.apply)
