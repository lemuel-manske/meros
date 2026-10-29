"""Inspect known individuals and numbered image records."""

from __future__ import annotations

import sys

from meros.project import Project


def main() -> None:
    subjects = Project.open().media.list_subjects()

    if "--check" in sys.argv[1:]:
        return

    if not subjects:
        print("No subject photographs found under data/media/subjects/")
        return

    for subject in subjects:
        print(f"{subject.subject_id}: {len(subject.images)} image(s)")

        for image in subject.images:
            print(f"  {image.image_id}: {image.path}")


if __name__ == "__main__":
    main()
