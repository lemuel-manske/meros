import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LEGACY_CSV = Path("data/media/sandbox/sift_ransac_adalam/results.csv")
PATTERN_CSV = Path("data/media/sandbox/sift_ransac_adalam_bin_mask/results.csv")
OUTPUT = Path("data/media/sandbox/identification/comparison.png")
SHOW_PLOTS = True

type Pair = tuple[tuple[str, str], tuple[str, str]]


def load_results(path: Path) -> dict[Pair, dict[str, str]]:
    results = {}

    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            endpoints = sorted(
                (
                    (row["subject_a"], row["image_a"]),
                    (row["subject_b"], row["image_b"]),
                )
            )

            pair: Pair = (endpoints[0], endpoints[1])
            assert pair not in results, f"Duplicate pair in {path}: {pair}"
            assert (row["same_individual"] == "True") == (pair[0][0] == pair[1][0])

            results[pair] = row

    return results


def metric(row: dict[str, str], column: str) -> float:
    value = row[column]

    return float(value) if value else float("nan")


def run() -> None:
    legacy = load_results(LEGACY_CSV)
    pattern = load_results(PATTERN_CSV)

    assert legacy, "No baseline results found"
    assert legacy.keys() == pattern.keys(), (
        f"Missing from pattern: {len(legacy.keys() - pattern.keys())}; "
        f"missing from baseline: {len(pattern.keys() - legacy.keys())}"
    )

    pairs = sorted(legacy)
    positive = np.array([a[0] == b[0] for a, b in pairs], dtype=bool)

    global_legacy = np.array([metric(legacy[p], "global_inliers") for p in pairs])
    global_pattern = np.array([metric(pattern[p], "global_inliers") for p in pairs])
    local_legacy = np.array([metric(legacy[p], "local_inliers") for p in pairs])
    iou = np.array([metric(pattern[p], "pattern_iou") for p in pairs])

    accepted_legacy = np.isfinite(local_legacy) & (local_legacy > 0)
    accepted_pattern = np.array([pattern[p]["status"] == "ok" for p in pairs])

    print(f"Compared pairs: {len(pairs)}")
    print(f"Positive pairs: {int(positive.sum())}")
    print(f"Negative pairs: {int((~positive).sum())}")

    for name, accepted in (
        ("Baseline (AdaLAM local > 0)", accepted_legacy),
        ("Pattern (status == ok)", accepted_pattern),
    ):
        tp = int(np.count_nonzero(accepted & positive))
        fn = int(np.count_nonzero(~accepted & positive))
        fp = int(np.count_nonzero(accepted & ~positive))
        tn = int(np.count_nonzero(~accepted & ~positive))

        print(f"{name}: TP={tp} FN={fn} FP={fp} TN={tn}")

    print(f"Pattern statuses: {dict(Counter(pattern[p]['status'] for p in pairs))}")

    for pair, is_positive, old, new in zip(
        pairs, positive, accepted_legacy, accepted_pattern, strict=True
    ):
        if old != new:
            print(
                f"Different decisions: {pair} | same={is_positive} | "
                f"baseline={old} | pattern={new} | {pattern[pair]['status']}"
            )

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    ax = axes[0, 0]

    finite = np.isfinite(global_legacy) & np.isfinite(global_pattern)

    for selection, label in ((positive, "Same individual"), (~positive, "Different")):
        chosen = selection & finite

        ax.scatter(
            global_legacy[chosen],
            global_pattern[chosen],
            alpha=0.65,
            label=f"{label} ({chosen.sum()})",
        )

    bounds = np.concatenate((global_legacy[finite], global_pattern[finite]))

    if bounds.size:
        maximum = max(float(bounds.max()), 1.0)
        ax.plot([0, maximum], [0, maximum], linestyle="--", color="gray")

    ax.set(
        title="Global RANSAC inliers: paired comparison",
        xlabel="Baseline",
        ylabel="Baseline matching + GloPER",
    )
    ax.legend()
    ax.grid(alpha=0.2)

    ax = axes[0, 1]
    finite_local = np.isfinite(global_legacy) & np.isfinite(local_legacy)

    for selection, label in ((positive, "Same individual"), (~positive, "Different")):
        chosen = selection & finite_local

        ax.scatter(
            global_legacy[chosen],
            local_legacy[chosen],
            alpha=0.65,
            label=label,
        )

    ax.set(
        title="Global RANSAC versus local AdaLAM",
        xlabel="Global inliers",
        ylabel="AdaLAM local correspondences",
    )
    ax.legend()
    ax.grid(alpha=0.2)

    ax = axes[1, 0]

    x = np.arange(2)
    positive_count = int(positive.sum())
    negative_count = int((~positive).sum())

    true_positive = [
        np.count_nonzero(accepted_legacy & positive) / positive_count,
        np.count_nonzero(accepted_pattern & positive) / positive_count,
    ]
    false_positive = [
        np.count_nonzero(accepted_legacy & ~positive) / negative_count,
        np.count_nonzero(accepted_pattern & ~positive) / negative_count,
    ]

    ax.bar(x - 0.18, true_positive, 0.36, label="True positive rate")
    ax.bar(x + 0.18, false_positive, 0.36, label="False positive rate")
    ax.set_xticks(x, ["Baseline local", "GloPER + geometry"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction")
    ax.set_title("Different operational acceptance criteria")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1, 1]
    finite_iou = np.isfinite(iou) & np.isfinite(global_pattern)

    for selection, label in ((positive, "Same individual"), (~positive, "Different")):
        chosen = selection & finite_iou

        if np.any(chosen):
            ax.scatter(global_pattern[chosen], iou[chosen], alpha=0.7, label=label)

    if np.any(finite_iou):
        ax.legend()
    else:
        ax.text(
            0.5,
            0.5,
            "No valid GloPER comparisons",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.set(
        title=f"GloPER IoU (available for {int(finite_iou.sum())}/{len(pairs)} pairs)",
        xlabel="Global RANSAC inliers",
        ylabel="Pattern IoU",
    )
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)

    fig.suptitle(
        "SIFT + RANSAC + AdaLAM: matching baseline vs GloPER verification\n"
        "The two decision rules are not equivalent",
        fontsize=14,
    )
    fig.tight_layout()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180)

    print(f"Saved plot: {OUTPUT}")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    run()
