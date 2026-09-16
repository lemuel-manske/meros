import numpy as np

from dataclasses import dataclass
from itertools import combinations

from src.meros import metadata, media
from src.meros.cmd.match_composites import match_composites
from src.meros.engine import pipeline


@dataclass(frozen=True)
class Pair:
    source_video_id: str
    source_track_id: str
    target_video_id: str
    target_track_id: str
    same_individual: bool


@dataclass(frozen=True)
class PairResult:
    pair: Pair
    tentative_matches: int
    affine_inliers: int
    score: float


def score_composites(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[int, int, float]:
    _, matches, inliers = match_composites(
        source,
        target,
    )

    score = (
        inliers / matches
        if matches > 0
        else 0.0
    )

    return matches, inliers, score


def build_pairs() -> list[Pair]:
    individuals = metadata.read_individuals()

    tracks = [
        (
            individual.individual_id,
            track,
        )
        for individual in individuals
        for track in individual.tracks
    ]

    pairs = []

    for (
        source_individual,
        source,
    ), (
        target_individual,
        target,
    ) in combinations(tracks, 2):

        # avoid trivial within-video comparisons.
        if source.video_id == target.video_id:
            continue

        pairs.append(
            Pair(
                source_video_id=source.video_id,
                source_track_id=source.track_id,
                target_video_id=target.video_id,
                target_track_id=target.track_id,
                same_individual=(
                    source_individual
                    == target_individual
                ),
            )
        )

    return pairs


class Exp:

    def run(self) -> list[PairResult]:
        pipeline.run("composites")

        results = []

        for pair in build_pairs():
            source = media.read_composite(
                pair.source_video_id,
                pair.source_track_id,
            )

            target = media.read_composite(
                pair.target_video_id,
                pair.target_track_id,
            )

            matches, inliers, score = score_composites(
                source,
                target,
            )

            print(
                f"Comparing {pair.source_video_id}:{pair.source_track_id} "
                f"to {pair.target_video_id}:{pair.target_track_id} "
                f"(same individual: {pair.same_individual}) "
                f"-> matches: {matches}, inliers: {inliers}, score: {score}"
            )

            results.append(
                PairResult(
                    pair=pair,
                    tentative_matches=matches,
                    affine_inliers=inliers,
                    score=score,
                )
            )

        return results


def summarize(
    results: list[PairResult],
) -> None:
    positive = [
        result.score
        for result in results
        if result.pair.same_individual
    ]

    negative = [
        result.score
        for result in results
        if not result.pair.same_individual
    ]

    print("same individual:", positive)
    print("different individual:", negative)


if __name__ == "__main__":
    experiment = Exp()
    results = experiment.run()
    summarize(results)
