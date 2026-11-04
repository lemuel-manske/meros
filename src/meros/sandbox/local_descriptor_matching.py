"""
Explore frozen local image descriptors at annotated head spots.

Run from the repository root:
    python -m meros.sandbox.local_descriptor_matching

Set BACKBONE to "sift" for the untrained control. DINOv2 requires torch
and an initial download of the official pretrained weights via torch.hub.
Verified spot correspondences are read only after inference.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

from meros.project import Project


BACKBONE = "dinov2"  # "dinov2" or "sift"
PAIRS = (
    (("rogerio", 10), ("rogerio", 12)),
    (("rafael", 3), ("rogerio", 9)),
)
PATCH_SPACING_MULTIPLIER = 2.5
PATCH_IMAGE_SIZE = 224
BATCH_SIZE = 16
ANNOTATION_TOLERANCE_PX = 16.0
MAX_DRAWN_MATCHES = 35


def median_spacing(points: np.ndarray) -> float:
    """
    Calculate median nearest-neighbor distance in a set of annotated spots.
    """
    distances = cdist(points, points)
    np.fill_diagonal(distances, np.inf)
    return float(np.median(distances.min(axis=1)))


def image_patches(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Extract spot-centered RGB patches with radius based on local spot density.
    """
    half_width = max(16, int(round(PATCH_SPACING_MULTIPLIER * median_spacing(points))))
    padded = cv.copyMakeBorder(
        image, half_width, half_width, half_width, half_width, cv.BORDER_REFLECT_101
    )
    patches = []

    for point in points:
        x, y = np.rint(point).astype(int)
        patch = padded[y : y + 2 * half_width, x : x + 2 * half_width]
        patch = cv.resize(patch, (PATCH_IMAGE_SIZE, PATCH_IMAGE_SIZE))
        patches.append(cv.cvtColor(patch, cv.COLOR_BGR2RGB))

    return np.stack(patches)


def dinov2_descriptors(image: np.ndarray, points: np.ndarray, model: Any) -> np.ndarray:
    """
    Extract frozen DINOv2 class tokens from image patches.
    """
    torch = import_module("torch")

    patches = image_patches(image, points)
    device = next(model.parameters()).device
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).reshape(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).reshape(1, 3, 1, 1)
    descriptors = []

    for start in range(0, len(patches), BATCH_SIZE):
        batch = torch.from_numpy(patches[start : start + BATCH_SIZE])
        batch = batch.permute(0, 3, 1, 2).float() / 255.0
        batch = ((batch - mean) / std).to(device)

        with torch.inference_mode():
            features = model.forward_features(batch)["x_norm_clstoken"]

        descriptors.append(features.cpu().numpy())

    return np.concatenate(descriptors, axis=0)


def sift_descriptors(image: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute fixed-position SIFT descriptors for a non-learned control.
    """
    size = max(8.0, 2.0 * PATCH_SPACING_MULTIPLIER * median_spacing(points))
    keypoints = [cv.KeyPoint(float(x), float(y), size) for x, y in points]
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    returned, descriptors = getattr(cv, "SIFT_create")().compute(gray, keypoints)

    if descriptors is None or not returned:
        raise ValueError("No SIFT descriptors")

    positions = np.array([point.pt for point in returned])
    indices = cdist(positions, points).argmin(axis=1)
    return points[indices], descriptors


def cosine_matrix(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Compare L2-normalized descriptor vectors.
    """
    first = source / np.maximum(np.linalg.norm(source, axis=1, keepdims=True), 1e-12)
    second = target / np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-12)
    return first @ second.T


def reciprocal_matches(similarity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Match only mutual nearest neighbors, without fitting a geometric transform.
    """
    target = similarity.argmax(axis=1)
    source = similarity.argmax(axis=0)
    indices = np.arange(len(target))
    accepted = source[target] == indices
    pairs = np.column_stack((indices[accepted], target[accepted]))
    values = similarity[pairs[:, 0], pairs[:, 1]]
    order = np.argsort(-values)
    return pairs[order], values[order]


def evaluate_verified(
    first: np.ndarray,
    second: np.ndarray,
    similarity: np.ndarray,
    matches: np.ndarray,
    annotations: list[dict],
) -> tuple[set[tuple[int, int]], int, int, int, int]:
    """
    Evaluate manually confirmed correspondences without treating other pairs as negatives.
    """
    truth: set[tuple[int, int]] = set()
    top1 = 0
    top3 = 0
    proposed = {tuple(map(int, pair)) for pair in matches}

    for annotation in annotations:
        source = np.asarray(annotation["first"], dtype=float)
        target = np.asarray(annotation["second"], dtype=float)
        i = int(cdist(source[None], first).argmin())
        j = int(cdist(target[None], second).argmin())

        if (
            np.linalg.norm(first[i] - source) > ANNOTATION_TOLERANCE_PX
            or np.linalg.norm(second[j] - target) > ANNOTATION_TOLERANCE_PX
        ):
            continue

        truth.add((i, j))
        order = np.argsort(-similarity[i])
        top1 += int(order[0] == j)
        top3 += int(j in order[:3])

    correct = len(truth.intersection(proposed))
    return truth, top1, top3, correct, len(annotations)


def match_label(
    source_index: int,
    target_index: int,
    truth: set[tuple[int, int]],
) -> str:
    """
    Classify a proposed link using only explicitly verified annotations.
    """
    if (source_index, target_index) in truth:
        return "confirmed"

    known_source = {i for i, _ in truth}
    known_target = {j for _, j in truth}

    if source_index in known_source or target_index in known_target:
        return "contradicted"

    return "unverified"


def diagnostic_report(
    similarity: np.ndarray,
    matches: np.ndarray,
    truth: set[tuple[int, int]] | None,
    title: str,
) -> None:
    """
    Inspect proposal ambiguity without selecting a rejection threshold.
    """
    source_order = np.argsort(-similarity, axis=1)
    source_gap = np.sort(similarity, axis=1)[:, -1] - np.sort(similarity, axis=1)[:, -2]
    target_gap = np.sort(similarity, axis=0)[-1] - np.sort(similarity, axis=0)[-2]

    print(f"\n{title}: mutual proposals with top-1 / top-2 cosine margins")
    print("  source -> target | cosine | source margin | target margin | verification")

    labels = []

    for source_index, target_index in matches:
        i, j = int(source_index), int(target_index)
        label = match_label(i, j, truth) if truth is not None else "unknown identity"
        labels.append(label)
        print(
            f"  {i:3d} -> {j:3d} | {similarity[i, j]:+.4f} | "
            f"{source_gap[i]:.4f} | {target_gap[j]:.4f} | {label}"
        )

    if truth is None:
        print("  No ground-truth correspondence labels are available for this impostor pair.")
        return

    for label in ("confirmed", "contradicted", "unverified"):
        print(f"  {label}: {labels.count(label)}/{len(matches)} proposals")

    print("\n  Verified marks: rank and ambiguity of their correct partner")
    print("  source -> target | target rank | correct cosine | best cosine | gap to best")

    ranks = []

    for i, j in sorted(truth):
        rank = int(np.flatnonzero(source_order[i] == j)[0]) + 1
        ranks.append(rank)
        best = float(similarity[i, source_order[i, 0]])
        correct = float(similarity[i, j])
        print(
            f"  {i:3d} -> {j:3d} | {rank:3d} | {correct:+.4f} | "
            f"{best:+.4f} | {best - correct:.4f}"
        )

    print(
        f"  Verified top-1={sum(rank == 1 for rank in ranks)}/{len(ranks)}, "
        f"top-3={sum(rank <= 3 for rank in ranks)}/{len(ranks)}."
    )
    print(
        "  Margins are descriptive; neither a cosine margin nor a mutual match "
        "is calibrated identity evidence."
    )

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    ranks_array = np.asarray(ranks)
    axes[0].hist(
        ranks_array,
        bins=np.arange(0.5, similarity.shape[1] + 1.5),
        edgecolor="black",
    )
    axes[0].set_title("Rank of manually verified partner")
    axes[0].set_xlabel("1 = best candidate")
    axes[0].set_ylabel("Verified spots")

    categories = {"confirmed": 0, "contradicted": 1, "unverified": 2}
    colors = {"confirmed": "#36b46b", "contradicted": "#df595a", "unverified": "#b6bdc9"}

    for i, j in matches:
        source_index, target_index = int(i), int(j)
        label = match_label(source_index, target_index, truth)
        axes[1].scatter(
            source_gap[source_index],
            target_gap[target_index],
            color=colors[label],
            label=label,
            s=36,
        )

    handles, labels_for_legend = axes[1].get_legend_handles_labels()
    unique = dict(zip(labels_for_legend, handles, strict=True))
    axes[1].legend(unique.values(), unique.keys())
    axes[1].set_title("Reciprocal match ambiguity")
    axes[1].set_xlabel("Source best-vs-second cosine gap")
    axes[1].set_ylabel("Target best-vs-second cosine gap")
    figure.suptitle(title)
    figure.tight_layout()
    plt.show(block=True)
    plt.close(figure)


def show_matches(
    first_image: np.ndarray,
    second_image: np.ndarray,
    first_points: np.ndarray,
    second_points: np.ndarray,
    matches: np.ndarray,
    truth: set[tuple[int, int]] | None,
    title: str,
) -> None:
    """
    Visualize inferred links: green confirmed, red contradicted, gray unverified.
    """
    height = 700
    first_scale = height / first_image.shape[0]
    second_scale = height / second_image.shape[0]
    left = cv.resize(first_image, None, fx=first_scale, fy=first_scale)
    right = cv.resize(second_image, None, fx=second_scale, fy=second_scale)
    canvas = cv.cvtColor(np.concatenate((left, right), axis=1), cv.COLOR_BGR2RGB)
    figure, axis = plt.subplots(figsize=(16, 8))
    axis.imshow(canvas)

    for i, j in matches[:MAX_DRAWN_MATCHES]:
        source = first_points[i] * first_scale
        target = second_points[j] * second_scale + np.array([left.shape[1], 0])

        label = match_label(int(i), int(j), truth) if truth is not None else "unverified"
        color = {
            "confirmed": "#36b46b",
            "contradicted": "#df595a",
            "unverified": "#b6bdc9",
        }[label]

        axis.plot((source[0], target[0]), (source[1], target[1]), color=color, alpha=0.75)
        axis.scatter((source[0], target[0]), (source[1], target[1]), s=7, color=color)

    axis.set_title(title)
    axis.set_axis_off()
    figure.tight_layout()
    plt.show(block=True)
    plt.close(figure)


def run() -> None:
    """
    Inspect a known genuine pair and a difficult impostor pair.
    """
    project = Project.open()
    model = None

    if BACKBONE == "dinov2":
        try:
            torch = import_module("torch")
        except ImportError as error:
            raise RuntimeError("Install PyTorch for DINOv2") from error

        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True)
        model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

    elif BACKBONE != "sift":
        raise ValueError(f"Unknown backbone: {BACKBONE}")

    for source_id, target_id in PAIRS:
        first_image = project.media.read_subject_image(*source_id)
        second_image = project.media.read_subject_image(*target_id)
        first_points = np.asarray(project.metadata.read_subject_spots(*source_id), dtype=float)
        second_points = np.asarray(project.metadata.read_subject_spots(*target_id), dtype=float)

        if len(first_points) < 4 or len(second_points) < 4:
            print(f"Skipping {source_id} / {target_id}: insufficient annotated spots")
            continue

        if BACKBONE == "sift":
            first_points, first_features = sift_descriptors(first_image, first_points)
            second_points, second_features = sift_descriptors(second_image, second_points)
        else:
            assert model is not None
            first_features = dinov2_descriptors(first_image, first_points, model)
            second_features = dinov2_descriptors(second_image, second_points, model)

        similarity = cosine_matrix(first_features, second_features)
        matches, _ = reciprocal_matches(similarity)
        truth = None

        if source_id[0] == target_id[0]:
            annotations = project.metadata.read_subject_spot_correspondences(
                source_id[0], source_id[1], target_id[1]
            )
            truth, top1, top3, correct, total = evaluate_verified(
                first_points, second_points, similarity, matches, annotations
            )
            print(
                f"{BACKBONE} {source_id} -> {target_id}: "
                f"verified={len(truth)}/{total}, top1={top1}, top3={top3}, "
                f"reciprocal_correct={correct}, proposals={len(matches)}"
            )

        else:
            print(
                f"{BACKBONE} {source_id} -> {target_id}: "
                f"proposals={len(matches)} (not a false acceptance rate)"
            )

        diagnostic_report(similarity, matches, truth, f"{BACKBONE}: {source_id} vs {target_id}")

        show_matches(
            first_image,
            second_image,
            first_points,
            second_points,
            matches,
            truth,
            f"{BACKBONE}: {source_id} vs {target_id}",
        )


if __name__ == "__main__":
    run()
