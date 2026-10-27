# scripts/analyze_allocation_results.py

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# CONFIG
# ============================================================

RESULT_DIR = Path("results/test_evaluation")

SEEDS = [42, 123, 456]

ALLOCATIONS = [
    "100_0",
    "75_25",
    "50_50",
    "25_75",
    "0_100",
]

VISION_PERCENT = {
    "100_0": 100,
    "75_25": 75,
    "50_50": 50,
    "25_75": 25,
    "0_100": 0,
}

LANGUAGE_PERCENT = {
    allocation: 100 - vision
    for allocation, vision in VISION_PERCENT.items()
}

OUTPUT_DIR = Path("results/analysis")

CSV_OUTPUT = (
    OUTPUT_DIR / "allocation_summary.csv"
)

JSON_OUTPUT = (
    OUTPUT_DIR / "allocation_summary.json"
)

FIGURE_OUTPUT = (
    OUTPUT_DIR / "allocation_curve.png"
)


# ============================================================
# HELPERS
# ============================================================

def load_results(seed: int) -> dict:
    """
    Load test results for one seed.

    Seed 42 may use the older JSON structure:
        {
            "frozen": {...},
            "100_0": {...},
            ...
        }

    Seeds 123/456 use:
        {
            "seed": 123,
            "results": {
                ...
            }
        }

    Support both formats.
    """

    if seed == 42:
        path = (
            RESULT_DIR
            / "flickr30k_test_results.json"
        )
    else:
        path = (
            RESULT_DIR
            / f"flickr30k_test_results_seed{seed}.json"
        )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing result file:\n{path}"
        )

    with open(path, "r") as f:
        data = json.load(f)

    if "results" in data:
        return data["results"]

    return data


def validate_results(results: dict, seed: int):
    required = {
        "frozen",
        *ALLOCATIONS,
    }

    missing = required - set(results.keys())

    if missing:
        raise ValueError(
            f"Seed {seed} is missing results for: "
            f"{sorted(missing)}"
        )


def metric(results: dict, allocation: str):
    return float(
        results[allocation]["mean_R@1"]
    )


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results = {}

    # --------------------------------------------------------
    # Load all three seeds
    # --------------------------------------------------------

    for seed in SEEDS:

        results = load_results(seed)

        validate_results(
            results,
            seed,
        )

        all_results[seed] = results

    # --------------------------------------------------------
    # Build summary
    # --------------------------------------------------------

    summary = []

    for allocation in ALLOCATIONS:

        values = np.array(
            [
                metric(
                    all_results[seed],
                    allocation,
                )
                for seed in SEEDS
            ],
            dtype=float,
        )

        summary.append(
            {
                "allocation": allocation,
                "vision_percent": VISION_PERCENT[
                    allocation
                ],
                "language_percent": LANGUAGE_PERCENT[
                    allocation
                ],
                "seed_42": values[0],
                "seed_123": values[1],
                "seed_456": values[2],
                "mean_R@1": float(
                    np.mean(values)
                ),
                "std_R@1": float(
                    np.std(
                        values,
                        ddof=1,
                    )
                ),
                "min_R@1": float(
                    np.min(values)
                ),
                "max_R@1": float(
                    np.max(values)
                ),
            }
        )

    # --------------------------------------------------------
    # Frozen baseline
    # --------------------------------------------------------

    frozen_values = np.array(
        [
            metric(
                all_results[seed],
                "frozen",
            )
            for seed in SEEDS
        ],
        dtype=float,
    )

    frozen_mean = float(
        np.mean(frozen_values)
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    fieldnames = [
        "allocation",
        "vision_percent",
        "language_percent",
        "seed_42",
        "seed_123",
        "seed_456",
        "mean_R@1",
        "std_R@1",
        "min_R@1",
        "max_R@1",
        "improvement_vs_frozen",
    ]

    with open(
        CSV_OUTPUT,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in summary:

            row["improvement_vs_frozen"] = (
                row["mean_R@1"]
                - frozen_mean
            )

            writer.writerow(row)

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    json_output = {
        "seeds": SEEDS,
        "frozen_mean_R@1": frozen_mean,
        "allocations": summary,
    }

    with open(
        JSON_OUTPUT,
        "w",
    ) as f:

        json.dump(
            json_output,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("=" * 80)
    print(
        "3-SEED FLICKR30K ALLOCATION ANALYSIS"
    )
    print("=" * 80)

    print(
        f"\nFrozen CLIP Mean R@1: "
        f"{frozen_mean:.4f}"
    )

    print()

    print(
        f"{'Allocation':<12}"
        f"{'Vision':>10}"
        f"{'Language':>10}"
        f"{'Seed 42':>10}"
        f"{'Seed 123':>10}"
        f"{'Seed 456':>10}"
        f"{'Mean':>10}"
        f"{'Std':>10}"
    )

    print("-" * 82)

    for row in summary:

        print(
            f"{row['allocation']:<12}"
            f"{row['vision_percent']:>9}%"
            f"{row['language_percent']:>9}%"
            f"{row['seed_42']:>10.4f}"
            f"{row['seed_123']:>10.4f}"
            f"{row['seed_456']:>10.4f}"
            f"{row['mean_R@1']:>10.4f}"
            f"{row['std_R@1']:>10.4f}"
        )

    # --------------------------------------------------------
    # Best mean allocation
    # --------------------------------------------------------

    best = max(
        summary,
        key=lambda x: x["mean_R@1"],
    )

    print("\n" + "=" * 80)

    print(
        "BEST AVERAGE ALLOCATION"
    )

    print("=" * 80)

    print(
        f"Allocation: "
        f"{best['allocation']}"
    )

    print(
        f"Mean R@1: "
        f"{best['mean_R@1']:.4f}"
    )

    print(
        f"Std R@1: "
        f"{best['std_R@1']:.4f}"
    )

    print(
        f"Improvement vs frozen: "
        f"{best['mean_R@1'] - frozen_mean:+.4f}"
    )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    x = np.array(
        [
            row["vision_percent"]
            for row in summary
        ]
    )

    y = np.array(
        [
            row["mean_R@1"]
            for row in summary
        ]
    )

    yerr = np.array(
        [
            row["std_R@1"]
            for row in summary
        ]
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.errorbar(
        x,
        y,
        yerr=yerr,
        marker="o",
        markersize=6,
        linewidth=1.5,
        capsize=4,
    )

    plt.axhline(
        frozen_mean,
        linestyle="--",
        linewidth=1.2,
    )

    plt.xlabel(
        "Vision LoRA Budget (%)"
    )

    plt.ylabel(
        "Mean Recall@1"
    )

    plt.title(
        "Flickr30k Retrieval vs. LoRA Budget Allocation"
    )

    plt.xticks(
        [0, 25, 50, 75, 100]
    )

    plt.grid(
        alpha=0.25,
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_OUTPUT,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"\nCSV saved to:\n{CSV_OUTPUT}"
    )

    print(
        f"JSON saved to:\n{JSON_OUTPUT}"
    )

    print(
        f"Figure saved to:\n{FIGURE_OUTPUT}"
    )


if __name__ == "__main__":
    main()