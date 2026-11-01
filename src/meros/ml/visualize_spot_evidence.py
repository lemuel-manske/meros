"""
Inspect spot correspondences and potential contradictions on complete photographs.

The current visual/geometric matcher is unchanged. Cyan pairs are proposed
matches, yellow spots are annotated but unmatched, and red crosses indicate
spots without a nearby counterpart in an overlapping annotation footprint.
These are not verified negative spots or proof of different identities.

Edit PAIRS to inspect other examples. Requires a desktop Matplotlib backend.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch
from scipy.spatial.distance import cdist

from meros.ml.evaluate_individual_reid import (
    GEOMETRIC_RADIUS_RATIO,
    MIN_SPOTS,
    SpotMatch,
    SpotPhoto,
    create_encoder,
    geometric_visual_match,
    load_photo,
    pair_score,
    project_points,
)
from meros.project import Project


PAIRS = (
    (("rafael", 3), ("rogerio", 1)),
    (("rogerio", 1), ("rogerio", 4)),
    (("athila", 0), ("rogerio", 1)),
    (("rogerio", 8), ("rafael", 2)),
    (("rafael", 4), ("rogerio", 0)),
)

SHOW_PLOTS = True
MAX_CONNECTIONS = 20
MIN_SHARED_SPOTS = 5
MIN_ANNOTATED_OVERLAP = 0.20
ANNOTATED_BORDER_RATIO = 0.03


@dataclass(frozen=True)
class SharedEvidence:
    source_indices: tuple[int, ...]
    target_indices: tuple[int, ...]
    source_unsupported: tuple[int, ...]
    target_unsupported: tuple[int, ...]
    annotated_overlap: float

    @property
    def density_ratio(self) -> float:
        counts = (len(self.source_indices), len(self.target_indices))
        return max(counts) / min(counts)


def shared_evidence(
    source: SpotPhoto,
    target: SpotPhoto,
    matrix: np.ndarray,
) -> SharedEvidence | None:
    """
    Estimate spot support only within intersecting annotation convex hulls.

    This proxy does not establish identical skin visibility or exhaustive
    annotations. The fitted geometric transformation may also be overfit.
    """
    projected = project_points(source.points, matrix)
    source_hull = cv.convexHull(projected.astype(np.float32))
    target_hull = cv.convexHull(target.points.astype(np.float32))
    source_area = cv.contourArea(source_hull)
    target_area = cv.contourArea(target_hull)

    if min(source_area, target_area) <= 0:
        return None

    area, intersection = cv.intersectConvexConvex(source_hull, target_hull)

    if intersection is None or area <= 0:
        return None

    overlap = float(area / min(source_area, target_area))

    if overlap < MIN_ANNOTATED_OVERLAP:
        return None

    border = ANNOTATED_BORDER_RATIO * target.scale

    def interior(points: np.ndarray) -> tuple[int, ...]:
        return tuple(
            i
            for i, point in enumerate(points)
            if cv.pointPolygonTest(
                intersection, (float(point[0]), float(point[1])), True
            ) >= border
        )

    source_ids = interior(projected)
    target_ids = interior(target.points)

    if min(len(source_ids), len(target_ids)) < MIN_SHARED_SPOTS:
        return None

    distances = cdist(projected[list(source_ids)], target.points[list(target_ids)])
    radius = GEOMETRIC_RADIUS_RATIO * target.scale
    source_unsupported = tuple(
        source_ids[i] for i in np.where(np.min(distances, axis=1) > radius)[0]
    )
    target_unsupported = tuple(
        target_ids[j] for j in np.where(np.min(distances, axis=0) > radius)[0]
    )

    return SharedEvidence(
        source_ids, target_ids, source_unsupported, target_unsupported, overlap
    )


def draw_spots(
    axis,
    photo: SpotPhoto,
    matched: set[int],
    unsupported: tuple[int, ...],
) -> None:
    """
    Distinguish matched, unmatched, and potentially unsupported annotations.
    """
    unmatched = [i for i in range(len(photo.points)) if i not in matched]

    if unmatched:
        points = photo.points[unmatched]
        axis.scatter(
            points[:, 0], points[:, 1], s=28, c="#ffcf54",
            edgecolors="black", linewidths=0.35, alpha=0.8, zorder=3,
        )

    if matched:
        points = photo.points[sorted(matched)]
        axis.scatter(
            points[:, 0], points[:, 1], s=30, c="#37f5eb",
            edgecolors="black", linewidths=0.45, zorder=4,
        )

    if unsupported:
        points = photo.points[list(unsupported)]
        axis.scatter(
            points[:, 0], points[:, 1], s=75, marker="x", c="#ff416c",
            linewidths=1.5, zorder=5,
        )

    axis.scatter(
        [photo.eye[0]], [photo.eye[1]], marker="+", s=110,
        c="white", linewidths=1.5, zorder=6,
    )
    axis.scatter(
        [photo.mouth[0]], [photo.mouth[1]], marker="s", s=45,
        facecolors="none", edgecolors="white", linewidths=1.5, zorder=6,
    )


def show_pair(
    first_label: tuple[str, int],
    second_label: tuple[str, int],
    first_image: np.ndarray,
    second_image: np.ndarray,
    source: SpotPhoto,
    target: SpotPhoto,
    match: SpotMatch,
    score: float,
    shared: SharedEvidence | None,
) -> None:
    """
    Show full-size photo panels, matched links, and potential absent evidence.
    """
    figure, axes = plt.subplots(1, 2, figsize=(18, 9))
    first_axis, second_axis = axes

    for axis, label, image in zip(
        axes, (first_label, second_label), (first_image, second_image), strict=True
    ):
        axis.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        axis.set_title(f"{label[0]}/{label[1]} — {image.shape[1]}x{image.shape[0]}")
        axis.set_axis_off()

    source_matched = {i for i, _ in match.pairs}
    target_matched = {j for _, j in match.pairs}

    draw_spots(
        first_axis, source, source_matched,
        shared.source_unsupported if shared is not None else (),
    )
    draw_spots(
        second_axis, target, target_matched,
        shared.target_unsupported if shared is not None else (),
    )

    # Show a readable subset; draw ALL matched points on their photos above.
    pair_indices = sorted(
        range(len(match.pairs)), key=lambda index: match.errors[index]
    )[:MAX_CONNECTIONS]

    for line_number, pair_index in enumerate(pair_indices, start=1):
        i, j = match.pairs[pair_index]
        first_point = source.points[i]
        second_point = target.points[j]
        connection = ConnectionPatch(
            xyA=(float(first_point[0]), float(first_point[1])),
            xyB=(float(second_point[0]), float(second_point[1])),
            coordsA="data",
            coordsB="data",
            axesA=first_axis,
            axesB=second_axis,
            color="#37f5eb",
            alpha=0.55,
            linewidth=0.9,
            zorder=2,
        )
        figure.add_artist(connection)

        for axis, point in ((first_axis, first_point), (second_axis, second_point)):
            axis.annotate(
                str(line_number), (float(point[0]), float(point[1])),
                xytext=(4, 4), textcoords="offset points",
                color="white", fontsize=7, fontweight="bold", zorder=7,
            )

    if shared is None:
        region = "Shared annotation region: not comparable"
    else:
        n = len(shared.source_indices) + len(shared.target_indices)
        unsupported = len(shared.source_unsupported) + len(shared.target_unsupported)
        region = (
            f"Annotated overlap={shared.annotated_overlap:.2f}; "
            f"spots={len(shared.source_indices)}+{len(shared.target_indices)}; "
            f"density ratio={shared.density_ratio:.2f}; "
            f"unsupported={unsupported}/{n} ({unsupported / n:.1%})"
        )

    figure.suptitle(
        f"{first_label[0]}/{first_label[1]} → {second_label[0]}/{second_label[1]}"
        f" | original score={score:.3f} | matches={len(match.pairs)}"
    )
    figure.text(
        0.5, 0.055,
        "Cyan: proposed matches (up to 20 numbered links)  |  "
        "Yellow: unpaired annotation  |  "
        "Red X: no nearby spot in comparable annotated region  |  "
        "White +/square: eye/mouth",
        ha="center", va="center", fontsize=10,
    )
    figure.text(
        0.5, 0.025, region + "  |  Red X is NOT proof of a biological difference",
        ha="center", va="center", fontsize=9,
    )
    figure.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.12, wspace=0.08)

    if SHOW_PLOTS:
        plt.show(block=True)

    plt.close(figure)


def run(project: Project) -> None:
    """
    Load selected pairs using Project stores and inspect matcher evidence.
    """
    encoder = create_encoder()
    photos: dict[tuple[str, int], SpotPhoto] = {}

    for source_label, target_label in PAIRS:
        for label in (source_label, target_label):
            if label in photos:
                continue

            points = project.metadata.read_subject_spots(*label)

            if len(points) < MIN_SPOTS:
                raise ValueError(f"{label} has fewer than {MIN_SPOTS} annotated spots")

            photo = load_photo(project, label, encoder)

            if photo is None:
                raise ValueError(f"Missing eye/mouth or invalid annotation: {label}")

            photos[label] = photo

    for source_label, target_label in PAIRS:
        source = photos[source_label]
        target = photos[target_label]
        match = geometric_visual_match(source, target)
        score = pair_score(source, target)
        shared = shared_evidence(source, target, match.matrix)

        print(
            f"{source_label[0]}/{source_label[1]} -> "
            f"{target_label[0]}/{target_label[1]}: "
            f"score={score:.3f}, matched={len(match.pairs)}, "
            f"coverage={len(match.pairs) / min(len(source.points), len(target.points)):.1%}"
        )

        if shared is None:
            print("  Shared annotation footprint: not comparable")
        else:
            count = len(shared.source_indices) + len(shared.target_indices)
            unmatched = sum(
                i not in shared.source_indices or j not in shared.target_indices
                for i, j in match.pairs
            )
            missing = len(shared.source_unsupported) + len(shared.target_unsupported)
            print(
                f"  Shared overlap={shared.annotated_overlap:.2f}, "
                f"density ratio={shared.density_ratio:.2f}, "
                f"unsupported={missing}/{count} ({missing / count:.1%}), "
                f"matched outside shared interior={unmatched}"
            )

        if SHOW_PLOTS:
            first_image = project.media.read_subject_image(*source_label)
            second_image = project.media.read_subject_image(*target_label)
            show_pair(
                source_label, target_label, first_image, second_image,
                source, target, match, score, shared,
            )

    print(
        "Annotated convex-hull overlap is not confirmed visible skin. "
        "Unmatched or unsupported spots may reflect viewpoint, illumination, "
        "annotation coverage, or longitudinal spot appearance changes."
    )


if __name__ == "__main__":
    run(Project.open())
