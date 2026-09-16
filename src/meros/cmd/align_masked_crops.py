import cv2 as cv
import numpy as np

from dataclasses import dataclass, replace

from src.meros import media, metadata
from src.meros.domain import AlignmentMetadata


SIFT_FEATURES = 4000
MATCH_RATIO = 0.75

MIN_INLIERS = 12
MAX_ERROR = 3.0

RANSAC_MAX_ITERS = 5000
RANSAC_CONFIDENCE = 0.999

MIN_INLIER_RATIO = 0.4
MIN_HULL_FRACTION = 0.1


@dataclass(frozen=True)
class ImageFeatures:
    gray: np.ndarray
    mask: np.ndarray
    keys: list
    descriptors: np.ndarray | None


def run() -> None:
    for individual in metadata.read_individuals():
        for track in individual.tracks:
            accepted, total = align_sequence(
                track.video_id,
                track.track_id,
                track.start_frame,
                track.end_frame,
                track.reference_frame,
            )

            print(
                f"{track.video_id}/{track.track_id}: "
                f"{accepted}/{total} alignment candidates."
            )


def gray_it(
    image: np.ndarray,
    apply_erode: bool = True,
    erode_size: int = 21,
) -> tuple[np.ndarray, np.ndarray]:
    gray = cv.cvtColor(
        image[:, :, :3],
        cv.COLOR_BGR2GRAY,
    )

    mask = (
        (image[:, :, 3] == 255)
        .astype(np.uint8)
        * 255
    )

    if apply_erode:
        mask = cv.erode(
            mask,
            np.ones(
                (erode_size, erode_size),
                dtype=np.uint8,
            ),
        )

    gray = cv.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    ).apply(gray)

    return gray, mask


def extract_features(
    image: np.ndarray,
    sift,
) -> ImageFeatures:
    gray, mask = gray_it(image)

    keys, descriptors = sift.detectAndCompute(
        gray,
        mask,
    )

    return ImageFeatures(
        gray=gray,
        mask=mask,
        keys=keys,
        descriptors=descriptors,
    )


def estimate_alignment_from_features(
    source: ImageFeatures,
    reference: ImageFeatures,
    min_inliers: int = MIN_INLIERS,
    max_error: float = MAX_ERROR,
) -> tuple[np.ndarray | None, AlignmentMetadata]:
    result = AlignmentMetadata(
        status="insufficient_matches",
        matches=0,
        inliers=0,
    )

    if (
        source.descriptors is None
        or reference.descriptors is None
    ):
        return None, result

    matcher = cv.BFMatcher()

    forward = matcher.knnMatch(
        source.descriptors,
        reference.descriptors,
        k=2,
    )

    backward = matcher.knnMatch(
        reference.descriptors,
        source.descriptors,
        k=2,
    )

    reverse = {
        pair[0].queryIdx: pair[0].trainIdx
        for pair in backward
        if (
            len(pair) == 2
            and pair[0].distance
            < MATCH_RATIO * pair[1].distance
        )
    }

    matches = [
        pair[0]
        for pair in forward
        if (
            len(pair) == 2
            and pair[0].distance
            < MATCH_RATIO * pair[1].distance
            and reverse.get(pair[0].trainIdx)
            == pair[0].queryIdx
        )
    ]

    if len(matches) < min_inliers:
        return None, AlignmentMetadata(
            status="insufficient_matches",
            matches=len(matches),
            inliers=0,
        )

    source_points = np.asarray(
        [
            source.keys[match.queryIdx].pt
            for match in matches
        ],
        dtype=np.float32,
    )

    reference_points = np.asarray(
        [
            reference.keys[match.trainIdx].pt
            for match in matches
        ],
        dtype=np.float32,
    )

    cv.setRNGSeed(0)

    matrix, inliers = cv.estimateAffine2D(
        source_points,
        reference_points,
        method=cv.RANSAC,
        ransacReprojThreshold=max_error,
        maxIters=RANSAC_MAX_ITERS,
        confidence=RANSAC_CONFIDENCE,
    )

    if matrix is None or inliers is None:
        return None, AlignmentMetadata(
            status="estimation_failed",
            matches=len(matches),
            inliers=0,
        )

    keep = inliers.ravel().astype(bool)

    predicted = cv.transform(
        source_points[:, None, :],
        matrix,
    )[:, 0, :]

    errors = np.linalg.norm(
        predicted - reference_points,
        axis=1,
    )

    hull = cv.convexHull(
        reference_points[keep]
    )

    coverage = (
        cv.contourArea(hull)
        / max(
            1,
            cv.countNonZero(reference.mask),
        )
    )

    accepted = (
        keep.sum() >= min_inliers
        and keep.mean() >= MIN_INLIER_RATIO
        and coverage >= MIN_HULL_FRACTION
        and np.linalg.det(matrix[:, :2]) > 0
    )

    result = AlignmentMetadata(
        status=(
            "candidate"
            if accepted
            else "rejected_geometry"
        ),
        matches=len(matches),
        inliers=int(keep.sum()),
        inlier_ratio=float(keep.mean()),
        median_error_px=float(
            np.median(errors[keep])
        ),
        inlier_hull_fraction=float(coverage),
        source_to_reference=matrix.tolist(),
    )

    return (
        matrix if accepted else None,
        result,
    )


def estimate_alignment(
    source: np.ndarray,
    reference: np.ndarray,
    min_inliers: int = MIN_INLIERS,
    max_error: float = MAX_ERROR,
) -> tuple[np.ndarray | None, AlignmentMetadata]:
    """
    Convenience function for tests / individual comparisons.

    align_sequence() uses the cached-feature version directly.
    """
    sift = cv.SIFT.create(
        nfeatures=SIFT_FEATURES,
    )

    return estimate_alignment_from_features(
        extract_features(source, sift),
        extract_features(reference, sift),
        min_inliers=min_inliers,
        max_error=max_error,
    )


def align_sequence(
    vid_id: str,
    track_id: str,
    start_frame: int,
    end_frame: int,
    reference_frame: int,
) -> tuple[int, int]:
    if not (
        start_frame
        <= reference_frame
        <= end_frame
    ):
        raise ValueError(
            "Reference frame must belong to the selected interval."
        )

    reference = media.read_masked_crop(
        vid_id,
        track_id,
        reference_frame,
    )

    height, width = reference.shape[:2]

    sift = cv.SIFT.create(
        nfeatures=SIFT_FEATURES,
    )

    # This is now calculated ONCE for the entire track.
    reference_features = extract_features(
        reference,
        sift,
    )

    rows: list[AlignmentMetadata] = []
    accepted = 0

    for frame_idx in range(
        start_frame,
        end_frame + 1,
    ):
        if frame_idx == reference_frame:
            continue

        source = media.read_masked_crop(
            vid_id,
            track_id,
            frame_idx,
        )

        # Source preprocessing + SIFT is also done only once.
        source_features = extract_features(
            source,
            sift,
        )

        matrix, result = (
            estimate_alignment_from_features(
                source_features,
                reference_features,
            )
        )

        result = replace(
            result,
            frame_idx=frame_idx,
        )

        rows.append(result)

        if matrix is None:
            media.remove_alignment(
                vid_id,
                track_id,
                frame_idx,
            )
            continue

        aligned = cv.warpAffine(
            source,
            matrix,
            (width, height),
        )

        aligned_gray = cv.warpAffine(
            source_features.gray,
            matrix,
            (width, height),
        )

        aligned_mask = cv.warpAffine(
            source_features.mask,
            matrix,
            (width, height),
            flags=cv.INTER_NEAREST,
        )

        overlap = (
            (reference_features.mask > 0)
            & (aligned_mask > 0)
        )

        overlay = np.zeros(
            (height, width, 3),
            dtype=np.uint8,
        )

        overlay[:, :, 0] = (
            reference_features.gray
        )

        overlay[:, :, 1] = aligned_gray

        overlay[:, :, 2] = (
            reference_features.gray
        )

        overlay[~overlap] = 0

        media.write_aligned_crop(
            vid_id,
            track_id,
            frame_idx,
            aligned,
        )

        media.write_aligned_overlay(
            vid_id,
            track_id,
            frame_idx,
            overlay,
        )

        accepted += 1

    metadata.write_alignment(
        vid_id,
        track_id,
        start_frame,
        end_frame,
        reference_frame,
        rows,
    )

    return accepted, len(rows)
