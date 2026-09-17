import numpy as np

from src.meros.cmd.match_composites import MatchStats, match_composites


def compare_representation(
    name: str,
    source: np.ndarray,
    target: np.ndarray,
) -> MatchStats:
    _, stats = match_composites(
        source,
        target,
    )

    print(
        f"{name}: "
        f"keys={stats.source_keypoints}/{stats.target_keypoints}, "
        f"forward={stats.forward_good}, "
        f"backward={stats.backward_good}, "
        f"mutual={stats.mutual_matches}, "
        f"inliers={stats.inliers}"
    )

    return stats
