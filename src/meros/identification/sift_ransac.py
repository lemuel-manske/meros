from dataclasses import dataclass

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from meros.project import Project

HOMOGRAPHY = True
RATIO_THRESHOLD = 0.80
RANSAC_THRESHOLD = 4.0
MIN_INLIERS = 20


@dataclass
class AlignmentResult:
    matrix: np.ndarray
    inliers: np.ndarray
    matches: list[cv.DMatch]
    kp_a: list[cv.KeyPoint]
    kp_b: list[cv.KeyPoint]
    errors: np.ndarray

    @property
    def num_matches(self) -> int:
        return len(self.matches)

    @property
    def num_inliers(self) -> int:
        return int(np.count_nonzero(self.inliers))

    @property
    def inlier_ratio(self) -> float:
        return self.num_inliers / max(self.num_matches, 1)

    @property
    def median_error(self) -> float:
        return float(np.median(self.errors[self.inliers]))


def to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv.cvtColor(image, cv.COLOR_BGRA2BGR)
    return image


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv.cvtColor(to_bgr(image), cv.COLOR_BGR2GRAY)


def body_silhouette(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] == 4:
        return (image[:, :, 3] > 0).astype(np.uint8)

    bgr = to_bgr(image)
    return np.any(bgr > 3, axis=2).astype(np.uint8)


def load_pattern(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape

    mask = cv.resize(
        mask,
        (width, height),
        interpolation=cv.INTER_NEAREST,
    )

    return mask > 127


def is_valid_homography(
    H: np.ndarray,
    shape: tuple[int, int],
    *,
    min_area_ratio: float = 0.05,
    max_area_ratio: float = 10.0,
) -> bool:
    height, width = shape

    corners = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ], dtype=np.float32).reshape(-1, 1, 2)

    transformed = cv.perspectiveTransform(corners, H)

    if not np.all(np.isfinite(transformed)):
        return False

    points = transformed.reshape(-1, 2)

    area = abs(cv.contourArea(points))
    original_area = width * height

    ratio = area / original_area

    if not min_area_ratio <= ratio <= max_area_ratio:
        return False

    # Check that the projected quadrilateral is convex.
    contour = points.reshape(-1, 1, 2).astype(np.float32)

    if not cv.isContourConvex(contour):
        return False

    return True


def inlier_coverage(
    keypoints: list[cv.KeyPoint],
    matches: list[cv.DMatch],
    inliers: np.ndarray,
    overlap: np.ndarray,
) -> float:
    points = np.array([
        keypoints[m.trainIdx].pt
        for m, valid in zip(matches, inliers, strict=True)
        if valid
    ], dtype=np.float32)

    if len(points) < 3:
        return 0.0

    hull = cv.convexHull(points)

    hull_mask = np.zeros(
        overlap.shape,
        dtype=np.uint8,
    )

    cv.fillConvexPoly(
        hull_mask,
        np.round(hull.reshape(-1, 2)).astype(np.int32),
        1,
    )

    overlap = overlap.astype(bool)

    hull_area = np.count_nonzero(
        hull_mask.astype(bool) & overlap
    )

    overlap_area = np.count_nonzero(overlap)

    if overlap_area == 0:
        return 0.0

    return hull_area / overlap_area

def estimate_alignment(
    image_a: np.ndarray,
    image_b: np.ndarray,
    silhouette_a: np.ndarray,
    silhouette_b: np.ndarray,
    *,
    homography: bool = False,
) -> AlignmentResult | None:
    sift = cv.SIFT.create(nfeatures=4000)

    kp_a, des_a = sift.detectAndCompute(
        to_gray(image_a),
        silhouette_a * 255,
    )
    kp_b, des_b = sift.detectAndCompute(
        to_gray(image_b),
        silhouette_b * 255,
    )

    if des_a is None or des_b is None:
        print("No descriptors found in one of the images")

        return None

    matcher = cv.BFMatcher(cv.NORM_L2)

    pairs = matcher.knnMatch(des_b, des_a, k=2)

    matches = [
        m
        for pair in pairs
        if len(pair) == 2
        for m, n in [pair]
        if m.distance < RATIO_THRESHOLD * n.distance
    ]

    if len(matches) < (4 if homography else 3):
        print(f"Not enough matches found: {len(matches)} (need at least {4 if homography else 3})")

        return None

    src = np.array([kp_b[m.queryIdx].pt for m in matches], dtype=np.float32).reshape(-1, 1, 2)

    dst = np.array([kp_a[m.trainIdx].pt for m in matches], dtype=np.float32).reshape(-1, 1, 2)

    if homography:
        matrix, inliers = cv.findHomography(
            src,
            dst,
            cv.RANSAC,
            RANSAC_THRESHOLD,
            maxIters=5000,
            confidence=0.995,
        )
    else:
        matrix, inliers = cv.estimateAffinePartial2D(
            src,
            dst,
            method=cv.RANSAC,
            ransacReprojThreshold=RANSAC_THRESHOLD,
            maxIters=5000,
            confidence=0.995,
        )

    if homography and not is_valid_homography(
        matrix,
        image_b.shape[:2],
    ):
        print("Estimated homography is invalid")

        return None

    if matrix is None or inliers is None:
        print("Failed to estimate transformation")
        return None

    inliers = inliers.ravel().astype(bool)

    if np.count_nonzero(inliers) < MIN_INLIERS:
        print(
            f"Not enough inliers found: {np.count_nonzero(inliers)} (need at least {MIN_INLIERS})"
        )

        return None

    if homography:
        projected = cv.perspectiveTransform(src, matrix)
    else:
        projected = cv.transform(src, matrix)

    errors = np.linalg.norm(
        projected.reshape(-1, 2) - dst.reshape(-1, 2),
        axis=1,
    )

    return AlignmentResult(
        matrix=matrix,
        inliers=inliers,
        matches=matches,
        kp_a=list(kp_a),
        kp_b=list(kp_b),
        errors=errors,
    )


def warp(
    image: np.ndarray,
    matrix: np.ndarray,
    shape: tuple[int, int],
    *,
    nearest: bool = False,
) -> np.ndarray:
    height, width = shape

    interpolation = cv.INTER_NEAREST if nearest else cv.INTER_LINEAR

    if matrix.shape == (3, 3):
        return cv.warpPerspective(
            image,
            matrix,
            (width, height),
            flags=interpolation,
        )

    return cv.warpAffine(
        image,
        matrix,
        (width, height),
        flags=interpolation,
    )


def visualize_matches(
    image_a: np.ndarray,
    image_b: np.ndarray,
    result: AlignmentResult,
) -> None:
    inlier_matches = [
        match
        for match, valid in zip(
            result.matches,
            result.inliers,
            strict=True,
        )
        if valid
    ]

    visualization = cv.drawMatches(
        to_bgr(image_b),
        result.kp_b,
        to_bgr(image_a),
        result.kp_a,
        inlier_matches,
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(0, 0, 255),
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    plt.figure(figsize=(18, 8))

    plt.imshow(cv.cvtColor(visualization, cv.COLOR_BGR2RGB))

    plt.title(
        f"RANSAC inliers: {result.num_inliers} / {result.num_matches} ({result.inlier_ratio:.1%})"
    )

    plt.axis("off")
    plt.tight_layout()
    plt.show()


def visualize_alignment(
    image_a: np.ndarray,
    image_b: np.ndarray,
    pattern_a: np.ndarray,
    pattern_b: np.ndarray,
    silhouette_a: np.ndarray,
    silhouette_b: np.ndarray,
    result: AlignmentResult,
) -> None:
    height, width = image_a.shape[:2]
    shape = (height, width)

    aligned_b = warp(image_b, result.matrix, shape)

    aligned_pattern_b = warp(
        pattern_b.astype(np.uint8),
        result.matrix,
        shape,
        nearest=True,
    ).astype(bool)

    aligned_silhouette_b = warp(
        silhouette_b,
        result.matrix,
        shape,
        nearest=True,
    ).astype(bool)

    # Valid pixels from B after geometric transformation.
    valid_b = warp(
        np.ones(image_b.shape[:2], dtype=np.uint8),
        result.matrix,
        shape,
        nearest=True,
    ).astype(bool)

    # Only compare pixels belonging to both animal silhouettes.
    overlap = silhouette_a.astype(bool) & aligned_silhouette_b & valid_b

    rgb_a = cv.cvtColor(to_bgr(image_a), cv.COLOR_BGR2RGB)
    rgb_b = cv.cvtColor(to_bgr(aligned_b), cv.COLOR_BGR2RGB)

    # RGB overlay only where B is valid.
    overlay = rgb_a.copy()

    blended = cv.addWeighted(rgb_a, 0.5, rgb_b, 0.5, 0)

    overlay[overlap] = blended[overlap]

    # Pattern comparison restricted to the shared animal region.
    binary_a = pattern_a & overlap
    binary_b = aligned_pattern_b & overlap

    intersection = binary_a & binary_b
    union = binary_a | binary_b

    intersection_count = np.count_nonzero(intersection)
    union_count = np.count_nonzero(union)

    iou = intersection_count / union_count if union_count > 0 else float("nan")

    coverage_a = np.count_nonzero(overlap) / max(np.count_nonzero(silhouette_a), 1)

    coverage_b = np.count_nonzero(overlap) / max(np.count_nonzero(aligned_silhouette_b), 1)

    # Red = A only, green = B only, yellow = agreement.
    pattern_overlay = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    inlier_cov = inlier_coverage(
        result.kp_a,
        result.matches,
        result.inliers,
        overlap
    )

    pattern_overlay[binary_a, 0] = 255
    pattern_overlay[binary_b, 1] = 255

    # Show the shared region on the reference image.
    overlap_visualization = rgb_a.copy()
    overlap_visualization[~overlap] = (overlap_visualization[~overlap] * 0.2).astype(np.uint8)

    # fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # axes[0, 0].imshow(rgb_a)
    # axes[0, 0].set_title("Image A (reference)")

    # axes[0, 1].imshow(rgb_b)
    # axes[0, 1].set_title("Image B (aligned)")

    # axes[0, 2].imshow(overlay)
    # axes[0, 2].set_title("RGB overlay (shared region)")

    # axes[1, 0].imshow(overlap_visualization)
    # axes[1, 0].set_title("Shared animal region")

    # axes[1, 1].imshow(pattern_overlay)
    # axes[1, 1].set_title("Patterns: red=A, green=B, yellow=both")

    # axes[1, 2].imshow(pattern_overlay)
    # axes[1, 2].imshow(
    #     overlap.astype(np.uint8),
    #     cmap="gray",
    #     alpha=0.18,
    #     vmin=0,
    #     vmax=1,
    # )
    # axes[1, 2].set_title(f"Pattern IoU: {iou:.3f}")

    # for ax in axes.flat:
    #     ax.axis("off")

    # fig.suptitle(
    #     f"Matches: {result.num_matches} | "
    #     f"Inliers: {result.num_inliers} | "
    #     f"Ratio: {result.inlier_ratio:.1%} | "
    #     f"Median error: {result.median_error:.2f}px",
    #     fontsize=14,
    # )

    # plt.tight_layout()
    # plt.show()

    print("\nAlignment metrics")
    print("-----------------")
    print(f"Matches:          {result.num_matches}")
    print(f"Inliers:          {result.num_inliers}")
    print(f"Inlier ratio:     {result.inlier_ratio:.2%}")
    print(f"Median error:     {result.median_error:.2f} px")
    print(f"Coverage of A:    {coverage_a:.2%}")
    print(f"Coverage of B:    {coverage_b:.2%}")
    print(f"Pattern IoU:      {iou:.4f}")
    print(f"Inlier coverage:  {inlier_cov:.2%}")
    print(f"\nTransformation B -> A:\n{result.matrix}")


def diff_individuals(project: Project):
    from itertools import combinations, product

    subjects = list(project.subjects.get_all())

    for subject_a, subject_b in combinations(subjects, 2):
        images_a = [
            img for img in subject_a.images
            if img.binary_im is not None
        ]

        images_b = [
            img for img in subject_b.images
            if img.binary_im is not None
        ]

        for a, b in product(images_a, images_b):
            yield subject_a, subject_b, a, b


def same_individual(project: Project):
    from itertools import combinations

    for subject in project.subjects.get_all():
        images = [
            img for img in subject.images
            if img.binary_im is not None
        ]

        for a, b in combinations(images, 2):
            yield subject, a, b


def run(project: Project) -> None:
    for subject, a, b in same_individual(project):
        subject_a = subject
        subject_b = subject

        image_a = a.crop_im
        image_b = b.crop_im

        mask_a = a.binary_im
        mask_b = b.binary_im

        assert mask_a is not None
        assert mask_b is not None

        pattern_a = load_pattern(mask_a, image_a.shape[:2])
        pattern_b = load_pattern(mask_b, image_b.shape[:2])

        silhouette_a = body_silhouette(image_a)
        silhouette_b = body_silhouette(image_b)

        print(f"\nAligning {subject_a.subject_id}/{a.image_id} with {subject_b.subject_id}/{b.image_id}")

        result = estimate_alignment(
            image_a,
            image_b,
            silhouette_a,
            silhouette_b,
            homography=HOMOGRAPHY,
        )

        if result is None:
            continue

        # visualize_matches(image_a, image_b, result)

        visualize_alignment(
            image_a,
            image_b,
            pattern_a,
            pattern_b,
            silhouette_a,
            silhouette_b,
            result,
        )


if __name__ == "__main__":
    run(Project.open())
