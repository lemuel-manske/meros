import cv2 as cv
import numpy as np

from src.alignment import (
    AlignmentMetadata,
    persist_aligned_crop,
    persist_alignment_metadata,
    persist_overlay,
    remove_alignment_images,
)
from src.artifacts import (
    path_to_alignment_media,
)
from src.fn import gray_it
from src.individuals import INDIVIDUALS
from src.masked_crop import (
    read_masked_crop,
)


def estimate_alignment(
    source: np.ndarray,
    reference: np.ndarray,
    min_inliers: int = 12,
    max_error: float = 3.0,
) -> tuple[np.ndarray | None, AlignmentMetadata]:
    source_gray, source_mask = gray_it(source)
    reference_gray, reference_mask = gray_it(reference)

    sift = cv.SIFT.create(nfeatures=4000)
    source_keys, source_descriptors = sift.detectAndCompute(source_gray, source_mask)
    reference_keys, reference_descriptors = sift.detectAndCompute(reference_gray, reference_mask)

    result: AlignmentMetadata = {"status": "insufficient_matches", "matches": 0, "inliers": 0}

    if source_descriptors is None or reference_descriptors is None:
        return None, result

    matcher = cv.BFMatcher()
    forward = matcher.knnMatch(source_descriptors, reference_descriptors, k=2)
    backward = matcher.knnMatch(reference_descriptors, source_descriptors, k=2)

    reverse = {
        pair[0].queryIdx: pair[0].trainIdx
        for pair in backward
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance
    }
    matches = [
        pair[0] for pair in forward
        if len(pair) == 2
        and pair[0].distance < 0.75 * pair[1].distance
        and reverse.get(pair[0].trainIdx) == pair[0].queryIdx
    ]
    result["matches"] = len(matches)

    if len(matches) < min_inliers:
        return None, result

    source_points = np.asarray([source_keys[m.queryIdx].pt for m in matches], dtype=np.float32)
    reference_points = np.asarray([reference_keys[m.trainIdx].pt for m in matches], dtype=np.float32)

    cv.setRNGSeed(0)
    matrix, inliers = cv.estimateAffine2D(
        source_points,
        reference_points,
        method=cv.RANSAC,
        ransacReprojThreshold=max_error,
        maxIters=5000,
        confidence=0.999,
    )

    if matrix is None or inliers is None:
        result["status"] = "estimation_failed"

        return None, result

    keep = inliers.ravel().astype(bool)
    predicted = cv.transform(source_points[:, None, :], matrix)[:, 0, :]
    errors = np.linalg.norm(predicted - reference_points, axis=1)
    hull = cv.convexHull(reference_points[keep])
    coverage = cv.contourArea(hull) / max(1, cv.countNonZero(reference_mask))

    result.update({
        "inliers": int(keep.sum()),
        "inlier_ratio": float(keep.mean()),
        "median_error_px": float(np.median(errors[keep])),
        "inlier_hull_fraction": float(coverage),
        "source_to_reference": matrix.tolist(),
    })

    # these are experiment guards, not calibrated alignment confidence.
    accepted = (
        keep.sum() >= min_inliers
        and keep.mean() >= 0.4
        and coverage >= 0.1
        and np.linalg.det(matrix[:, :2]) > 0
    )
    result["status"] = "candidate" if accepted else "rejected_geometry"

    return matrix if accepted else None, result


def align_sequence(
    vid_id: str,
    track_id: str,
    start_frame: int,
    end_frame: int,
    reference_frame: int,
) -> tuple[int, int]:
    if not start_frame <= reference_frame <= end_frame:
        raise ValueError("Reference frame must belong to the selected interval.")

    reference = read_masked_crop(vid_id, track_id, reference_frame)

    reference_gray, reference_mask = gray_it(reference)
    height, width = reference.shape[:2]

    rows = []
    accepted = 0

    for frame_idx in range(start_frame, end_frame + 1):
        if frame_idx == reference_frame:
            continue

        source = read_masked_crop(vid_id, track_id, frame_idx)

        matrix, result = estimate_alignment(source, reference)
        result["frame_idx"] = frame_idx

        rows.append(result)

        if matrix is None:
            remove_alignment_images(vid_id, track_id, frame_idx)

            continue

        source_gray, source_mask = gray_it(source)

        aligned = cv.warpAffine(source, matrix, (width, height))

        aligned_gray = cv.warpAffine(source_gray, matrix, (width, height))
        aligned_mask = cv.warpAffine(
            source_mask, matrix, (width, height), flags=cv.INTER_NEAREST,
        )

        overlap = (reference_mask > 0) & (aligned_mask > 0)

        # aligned structures appear gray; displaced structures have colored edges.
        overlay = np.zeros((height, width, 3), dtype=np.uint8)
        overlay[:, :, 0] = reference_gray
        overlay[:, :, 1] = aligned_gray
        overlay[:, :, 2] = reference_gray
        overlay[~overlap] = 0

        persist_aligned_crop(vid_id, track_id, frame_idx, aligned)
        persist_overlay(vid_id, track_id, frame_idx, overlay)

        accepted += 1

    persist_alignment_metadata(
        vid_id,
        track_id,
        start_frame,
        end_frame,
        reference_frame,
        rows,
    )

    total = len(rows)

    return accepted, total


if __name__ == "__main__":
    for individual in INDIVIDUALS:
        for track in individual["tracks"]:
            vid_id = track["video_id"]
            track_id = track["track_id"]

            accepted, total = align_sequence(
                vid_id,
                track_id,
                track["start_frame"],
                track["end_frame"],
                track["reference_frame"],
            )

            print(
                f"{vid_id}/{track_id}: {accepted}/{total} alignment candidates; "
                f"inspect {path_to_alignment_media(vid_id, track_id)}."
            )
