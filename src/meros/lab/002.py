import numpy as np

from itertools import combinations

from src.meros import media, metadata, Individual
from src.meros.cmd.match_composites import match_composites
from src.meros.engine import pipeline


def compare_representation(
    name: str,
    source: np.ndarray,
    target: np.ndarray,
) -> None:
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


def compare_same_individuals(individual: Individual) -> None:
    print(f"Same individual: {individual.individual_id}")

    for source, target in combinations(
        individual.tracks,
        2,
    ):
        if (
            source.video_id
            == target.video_id
        ):
            continue

        print(
            f"Comparing {source.video_id}/{source.track_id} "
            f"with {target.video_id}/{target.track_id}"
        )

        source_reference = media.read_masked_crop(
            source.video_id,
            source.track_id,
            source.reference_frame,
        )

        target_reference = media.read_masked_crop(
            target.video_id,
            target.track_id,
            target.reference_frame,
        )

        source_composite = media.read_composite(
            source.video_id,
            source.track_id,
        )

        target_composite = media.read_composite(
            target.video_id,
            target.track_id,
        )

        source_enhanced = media.read_composite_enhanced(
            source.video_id,
            source.track_id,
        )

        target_enhanced = media.read_composite_enhanced(
            target.video_id,
            target.track_id,
        )

        compare_representation(
            "reference",
            source_reference,
            target_reference,
        )

        compare_representation(
            "composite",
            source_composite,
            target_composite,
        )

        compare_representation(
            "enhanced",
            source_enhanced,
            target_enhanced,
        )


def compare_different_individuals(individual: Individual) -> None:
    print(f"Different individuals: {individual.individual_id}")

    for source in individual.tracks:
        for target in metadata.read_individuals():
            if (
                target.individual_id
                == individual.individual_id
            ):
                continue

            for target_track in target.tracks:
                if (
                    source.video_id
                    == target_track.video_id
                ):
                    continue

                print(
                    f"Comparing {source.video_id}/{source.track_id} "
                    f"with {target_track.video_id}/{target_track.track_id}"
                )

                source_reference = media.read_masked_crop(
                    source.video_id,
                    source.track_id,
                    source.reference_frame,
                )

                target_reference = media.read_masked_crop(
                    target_track.video_id,
                    target_track.track_id,
                    target_track.reference_frame,
                )

                source_composite = media.read_composite(
                    source.video_id,
                    source.track_id,
                )

                target_composite = media.read_composite(
                    target_track.video_id,
                    target_track.track_id,
                )

                source_enhanced = media.read_composite_enhanced(
                    source.video_id,
                    source.track_id,
                )

                target_enhanced = media.read_composite_enhanced(
                    target_track.video_id,
                    target_track.track_id,
                )

                compare_representation(
                    "reference",
                    source_reference,
                    target_reference,
                )

                compare_representation(
                    "composite",
                    source_composite,
                    target_composite,
                )

                compare_representation(
                    "enhanced",
                    source_enhanced,
                    target_enhanced,
                )


class Exp:

    def run(self) -> None:
        pipeline.run("enhanced_composites")

        for individual in metadata.read_individuals():
            compare_same_individuals(individual)
            compare_different_individuals(individual)


if __name__ == "__main__":
    exp = Exp()
    exp.run()
