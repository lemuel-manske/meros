import numpy as np

from itertools import combinations, product

from src.meros import metadata, media
from src.meros.cmd.match_composites import MatchStats
from src.meros.domain.storage import IndividualTrack
from src.meros.engine import pipeline

from .utils import compare_representation


def compare_pairs(a: dict, b: dict) -> MatchStats:
    print(
        f"Comparing "
        f"{a['video_id']}/{a['track_id']}@{a['frame']} with "
        f"{b['video_id']}/{b['track_id']}@{b['frame']}"
    )

    source_reference = media.read_masked_crop(
        a["video_id"],
        a["track_id"],
        a["frame"],
    )

    target_reference = media.read_masked_crop(
        b["video_id"],
        b["track_id"],
        b["frame"],
    )

    return compare_representation(
        "reference",
        source_reference,
        target_reference,
    )


def sample_frames(track: IndividualTrack, n=5) -> list[int]:
    s = track.start_frame
    e = track.end_frame

    frames = np.linspace(s, e, num=min(n, e - s + 1), dtype=int)

    return sorted(set(map(round, frames)))


def positive_pairs() -> list[dict]:
    pairs = []

    for individual in metadata.read_individuals():
        tracks = individual.tracks

        for a, b in combinations(tracks, 2):
            frames_a = sample_frames(a)
            frames_b = sample_frames(b)

            for frame_a, frame_b in product(frames_a, frames_b):
                pairs.append({
                    "individual_id": individual.individual_id,
                    "a": {
                        "video_id": a.video_id,
                        "track_id": a.track_id,
                        "frame": frame_a,
                    },
                    "b": {
                        "video_id": b.video_id,
                        "track_id": b.track_id,
                        "frame": frame_b,
                    },
                })

    return pairs


def negative_pairs() -> list[dict]:
    pairs = []

    individuals = metadata.read_individuals()

    for a, b in combinations(individuals, 2):
        for track_a in a.tracks:
            for track_b in b.tracks:
                frames_a = sample_frames(track_a)
                frames_b = sample_frames(track_b)

                for frame_a, frame_b in product(frames_a, frames_b):
                    pairs.append({
                        "a": {
                            "individual_id": a.individual_id,
                            "video_id": track_a.video_id,
                            "track_id": track_a.track_id,
                            "frame": frame_a,
                        },
                        "b": {
                            "individual_id": b.individual_id,
                            "video_id": track_b.video_id,
                            "track_id": track_b.track_id,
                            "frame": frame_b,
                        },
                    })

    return pairs


def save_csv(pairs: list[dict], fname: str) -> None:
    import csv

    with open(fname, "w", newline="") as csvfile:
        fieldnames = [
            "individual_id_a",
            "video_id_a",
            "track_id_a",
            "frame_a",
            "individual_id_b",
            "video_id_b",
            "track_id_b",
            "frame_b",
            "source_keypoints",
            "target_keypoints",
            "forward_good",
            "backward_good",
            "mutual_matches",
            "inliers",
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for pair in pairs:
            stats = compare_pairs(pair["a"], pair["b"])

            row = {
                "individual_id_a": pair["a"].get("individual_id", ""),
                "video_id_a": pair["a"]["video_id"],
                "track_id_a": pair["a"]["track_id"],
                "frame_a": pair["a"]["frame"],
                "individual_id_b": pair["b"].get("individual_id", ""),
                "video_id_b": pair["b"]["video_id"],
                "track_id_b": pair["b"]["track_id"],
                "frame_b": pair["b"]["frame"],
                "source_keypoints": stats.source_keypoints,
                "target_keypoints": stats.target_keypoints,
                "forward_good": stats.forward_good,
                "backward_good": stats.backward_good,
                "mutual_matches": stats.mutual_matches,
                "inliers": stats.inliers,
            }

            writer.writerow(row)


class Exp:

    def run(self) -> None:
        pipeline.run("enhanced_composites")

        save_csv(positive_pairs(), "positive_pairs.csv")
        save_csv(negative_pairs(), "negative_pairs.csv")


if __name__ == "__main__":
    exp = Exp()
    exp.run()
