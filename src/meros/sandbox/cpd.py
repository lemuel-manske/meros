"""
Demonstrate non-rigid Coherent Point Drift alignment of real grouper photographs.

Uses annotated spots and the actual photographs rogerio/10 and rogerio/12
from Project. Compares a global similarity baseline with the local deformation.
The alignment is exploratory; close spots do not prove fish-spot identity.

Run: python -m meros.sandbox.cpd
Requires the DVC subject images to be available locally.
"""

from __future__ import annotations

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RadioButtons, Slider
from scipy.spatial.distance import cdist

from meros.project import Project


SUBJECT = "rogerio"
SOURCE_ID = 10
TARGET_ID = 12

MAX_IMAGE_SIDE = 1100
WARP_GRID_STEP = 8
INITIAL_OVERLAY_ALPHA = 0.50
MASK_HULL_MARGIN = 1.5
COLOR_SOURCE = "#33ccdd"
COLOR_TARGET = "#ffcc59"


CPD_BETA = 2.5
CPD_LAMBDA = 3.0
CPD_OUTLIER_FRACTION = 0.15
CPD_ITERATIONS = 75
CPD_MIN_VARIANCE = 0.025
CPD_INVERSE_STEPS = 10


def load_photo(project: Project, image_id: int) -> tuple[np.ndarray, np.ndarray]:
    image = project.media.read_subject_image(SUBJECT, image_id)
    points = np.asarray(project.metadata.read_subject_spots(SUBJECT, image_id), dtype=float)

    if len(points) < 4:
        raise ValueError(f"{SUBJECT}/{image_id} has too few annotated spots")

    height, width = image.shape[:2]
    scale = min(1.0, MAX_IMAGE_SIDE / max(height, width))
    output_width, output_height = round(width * scale), round(height * scale)

    if scale < 1.0:
        image = cv.resize(image, (output_width, output_height), interpolation=cv.INTER_AREA)

    points *= (output_width / width, output_height / height)
    return image, points


def spot_spacing(points: np.ndarray) -> float:
    distances = cdist(points, points)
    np.fill_diagonal(distances, np.inf)
    return float(np.median(np.sort(distances, axis=1)[:, 1]))


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix[:2, :2].T + matrix[:2, 2]


def initial_similarity(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    a = source - source_center
    b = target - target_center

    source_values, source_vectors = np.linalg.eigh(np.cov(a.T))
    target_values, target_vectors = np.linalg.eigh(np.cov(b.T))
    source_angle = float(np.arctan2(source_vectors[1, -1], source_vectors[0, -1]))
    target_angle = float(np.arctan2(target_vectors[1, -1], target_vectors[0, -1]))
    scale = float(np.sqrt(target_values.sum() / source_values.sum()))

    candidates = []

    for extra_rotation in (0.0, np.pi):
        angle = target_angle - source_angle + extra_rotation
        rotation = scale * np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        matrix = np.eye(3)
        matrix[:2, :2] = rotation
        matrix[:2, 2] = target_center - rotation @ source_center

        distances = cdist(transform_points(source, matrix), target)
        symmetric_error = np.median(distances.min(axis=0)) + np.median(distances.min(axis=1))
        candidates.append((float(symmetric_error), matrix))

    return min(candidates, key=lambda item: item[0])[1]


def nonrigid_cpd(
    aligned_source: np.ndarray, target: np.ndarray, spacing: float
) -> tuple[np.ndarray, np.ndarray, int, float]:
    y = aligned_source / spacing
    x = target / spacing
    beta_sq = CPD_BETA**2
    kernel = np.exp(-cdist(y, y, "sqeuclidean") / (2.0 * beta_sq))
    weights = np.zeros_like(y)
    transformed = y.copy()

    initial_distances = cdist(y, x, "sqeuclidean")
    variance = max(1.0, float(np.median(initial_distances.min(axis=1)) / 2))
    m, n = len(y), len(x)
    identity = np.eye(m)

    iteration = -1

    for iteration in range(CPD_ITERATIONS):
        distances = cdist(transformed, x, "sqeuclidean")
        probabilities = np.exp(-distances / (2.0 * variance))
        outlier_term = (
            (2.0 * np.pi * variance) * CPD_OUTLIER_FRACTION / (1.0 - CPD_OUTLIER_FRACTION) * m / n
        )
        probabilities /= probabilities.sum(axis=0)[None, :] + outlier_term

        source_mass = probabilities.sum(axis=1)
        target_mass = probabilities.sum(axis=0)
        total_mass = source_mass.sum()

        if total_mass < 1e-8:
            raise RuntimeError("CPD lost all soft correspondences")

        weighted_targets = probabilities @ x
        linear_system = source_mass[:, None] * kernel + CPD_LAMBDA * variance * identity
        rhs = weighted_targets - source_mass[:, None] * y
        weights = np.linalg.solve(linear_system, rhs)
        transformed = y + kernel @ weights

        squared_error = (
            (target_mass[:, None] * x**2).sum()
            - 2.0 * (weighted_targets * transformed).sum()
            + (source_mass[:, None] * transformed**2).sum()
        )
        updated_variance = max(CPD_MIN_VARIANCE, float(squared_error / (2.0 * total_mass)))

        if abs(updated_variance - variance) / variance < 1e-5:
            variance = updated_variance
            break

        variance = updated_variance

    return y, weights, iteration + 1, variance


def cpd_forward(
    source_controls: np.ndarray, weights: np.ndarray, points: np.ndarray, spacing: float
) -> np.ndarray:
    normalized = points / spacing
    gaussian = np.exp(-cdist(normalized, source_controls, "sqeuclidean") / (2.0 * CPD_BETA**2))
    return points + spacing * (gaussian @ weights)


def cpd_inverse_displacement(
    points: np.ndarray, source_controls: np.ndarray, weights: np.ndarray, spacing: float
) -> np.ndarray:
    normalized = points / spacing
    estimate = normalized.copy()

    for _ in range(CPD_INVERSE_STEPS):
        gaussian = np.exp(-cdist(estimate, source_controls, "sqeuclidean") / (2.0 * CPD_BETA**2))
        proposed = normalized - gaussian @ weights
        estimate = 0.5 * estimate + 0.5 * proposed

    return (estimate - normalized) * spacing


def warp_photo(
    source: np.ndarray,
    target: np.ndarray,
    initial_matrix: np.ndarray,
    inverse_displacement,
    spot_region: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = target.shape[:2]
    grid_width = max(2, int(np.ceil(width / WARP_GRID_STEP)))
    grid_height = max(2, int(np.ceil(height / WARP_GRID_STEP)))
    sx = (np.arange(grid_width) + 0.5) * width / grid_width - 0.5
    sy = (np.arange(grid_height) + 0.5) * height / grid_height - 0.5
    gx, gy = np.meshgrid(sx, sy)
    sample = np.column_stack((gx.ravel(), gy.ravel()))

    delta = inverse_displacement(sample).reshape(grid_height, grid_width, 2)
    dx = cv.resize(delta[:, :, 0].astype(np.float32), (width, height))
    dy = cv.resize(delta[:, :, 1].astype(np.float32), (width, height))
    y, x = np.indices((height, width), dtype=np.float32)
    aligned_x, aligned_y = x + dx, y + dy

    inverse = np.linalg.inv(initial_matrix)
    map_x = inverse[0, 0] * aligned_x + inverse[0, 1] * aligned_y + inverse[0, 2]
    map_y = inverse[1, 0] * aligned_x + inverse[1, 1] * aligned_y + inverse[1, 2]
    warped = cv.remap(
        source,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT,
    )
    region = cv.remap(
        spot_region,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv.INTER_NEAREST,
        borderMode=cv.BORDER_CONSTANT,
    )
    return warped, region


def source_region(points: np.ndarray, shape: tuple[int, ...], spacing: float) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=np.uint8)
    hull = cv.convexHull(np.rint(points).astype(np.int32))
    cv.fillConvexPoly(mask, hull, 255)
    radius = max(3, round(MASK_HULL_MARGIN * spacing))
    cv.polylines(mask, [hull], isClosed=True, color=255, thickness=2 * radius)
    return mask


def nearest_error(points: np.ndarray, target: np.ndarray) -> float:
    distances = cdist(points, target)
    return float(np.median(distances.min(axis=1)))


def show_comparison(
    source_image: np.ndarray,
    target_image: np.ndarray,
    source_spots: np.ndarray,
    target_spots: np.ndarray,
    warped: dict[str, tuple[np.ndarray, np.ndarray]],
    transformed: dict[str, np.ndarray],
    method: str,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(15, 11))
    figure.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.16, wspace=0.08)

    axes[0, 0].imshow(cv.cvtColor(source_image, cv.COLOR_BGR2RGB))
    axes[0, 0].scatter(source_spots[:, 0], source_spots[:, 1], color=COLOR_SOURCE, s=9, alpha=0.8)
    axes[0, 0].set_title(f"{SUBJECT}/{SOURCE_ID}: original source")

    axes[0, 1].imshow(cv.cvtColor(target_image, cv.COLOR_BGR2RGB))
    axes[0, 1].scatter(target_spots[:, 0], target_spots[:, 1], color=COLOR_TARGET, s=9, alpha=0.8)
    axes[0, 1].set_title(f"{SUBJECT}/{TARGET_ID}: target")

    axes[1, 0].imshow(cv.cvtColor(warped[method][0], cv.COLOR_BGR2RGB))
    axes[1, 0].scatter(
        transformed[method][:, 0], transformed[method][:, 1], color=COLOR_SOURCE, s=9
    )
    axes[1, 0].scatter(target_spots[:, 0], target_spots[:, 1], color=COLOR_TARGET, s=9)
    axes[1, 0].set_title(f"{method}: warped real source photo")

    overlay_axis = axes[1, 1]
    overlay_axis.imshow(cv.cvtColor(target_image, cv.COLOR_BGR2RGB))
    overlay_artist = overlay_axis.imshow(
        cv.cvtColor(warped[method][0], cv.COLOR_BGR2RGB),
        alpha=INITIAL_OVERLAY_ALPHA * (warped[method][1] > 0),
    )
    source_artist = overlay_axis.scatter(
        transformed[method][:, 0],
        transformed[method][:, 1],
        color=COLOR_SOURCE,
        s=9,
    )
    overlay_axis.scatter(target_spots[:, 0], target_spots[:, 1], color=COLOR_TARGET, s=9)
    overlay_axis.set_title(f"Overlay on target: {method}")

    for axis in axes.flat:
        axis.set_axis_off()

    method_axis = figure.add_axes((0.21, 0.02, 0.19, 0.115))
    methods = RadioButtons(method_axis, ("Global", method), active=1)
    opacity_axis = figure.add_axes((0.53, 0.055, 0.38, 0.025))
    opacity = Slider(opacity_axis, "Overlay opacity", 0.0, 1.0, valinit=INITIAL_OVERLAY_ALPHA)

    def update(_value) -> None:
        selection = methods.value_selected
        image, region = warped[selection]
        overlay_artist.set_data(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        overlay_artist.set_alpha(opacity.val * (region > 0))
        source_artist.set_offsets(transformed[selection])
        overlay_axis.set_title(f"Overlay on target: {selection}")
        figure.canvas.draw_idle()

    methods.on_clicked(update)
    opacity.on_changed(update)

    print(
        "Nearest-spot median distances (not proof of correct correspondences): "
        + ", ".join(
            f"{name}={nearest_error(points, target_spots):.1f}px"
            for name, points in transformed.items()
        )
    )
    figure.suptitle(
        f"Real grouper photos | Global versus {method} | Cyan: source spots | Yellow: target spots"
    )
    plt.show(block=True)
    plt.close(figure)


def run() -> None:
    project = Project.open()
    source_image, source_points = load_photo(project, SOURCE_ID)
    target_image, target_points = load_photo(project, TARGET_ID)
    spacing = spot_spacing(target_points)
    initial_matrix = initial_similarity(source_points, target_points)
    global_points = transform_points(source_points, initial_matrix)

    source_controls, cpd_weights, iterations, variance = nonrigid_cpd(
        global_points, target_points, spacing
    )
    cpd_points = cpd_forward(source_controls, cpd_weights, global_points, spacing)
    region = source_region(source_points, source_image.shape, spot_spacing(source_points))

    def zero_displacement(points: np.ndarray) -> np.ndarray:
        return np.zeros_like(points)

    def inverse_cpd(points: np.ndarray) -> np.ndarray:
        return cpd_inverse_displacement(points, source_controls, cpd_weights, spacing)

    fields = {"Global": zero_displacement, "CPD": inverse_cpd}
    warped = {
        name: warp_photo(source_image, target_image, initial_matrix, field, region)
        for name, field in fields.items()
    }
    transformed = {"Global": global_points, "CPD": cpd_points}

    print(
        f"Real photos: {SUBJECT}/{SOURCE_ID} ({len(source_points)} spots) -> "
        f"{SUBJECT}/{TARGET_ID} ({len(target_points)} spots)"
    )
    print(f"CPD: {iterations} iterations, final normalized variance={variance:.4f}")
    show_comparison(
        source_image, target_image, source_points, target_points, warped, transformed, "CPD"
    )


if __name__ == "__main__":
    run()
