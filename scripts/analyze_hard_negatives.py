from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import CLIPModel, CLIPProcessor

from scripts.analyze_alignment import (
    ALLOCATIONS,
    MODEL_NAME,
    adapter_path,
    cleanup_model,
    extract_all_embeddings,
    load_test_data,
)


OUTPUT_DIR = Path("results/analysis/alignment")
RAW_OUTPUT = OUTPUT_DIR / "hard_negative_results.csv"
AGG_OUTPUT = OUTPUT_DIR / "hard_negative_aggregate.csv"

SEEDS = [42, 123, 456]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze Flickr30k CLIP hard-negative retrieval margins."
    )

    parser.add_argument(
        "--budget",
        type=int,
        choices=(294912, 589824),
        default=294912,
        help="LoRA parameter budget (default: 294912).",
    )

    return parser.parse_args()


@torch.no_grad()
def calculate_retrieval_margins(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    caption_to_image: np.ndarray,
):
    """
    Calculate positive-versus-hardest-negative retrieval margins.

    Image -> Text:
        Positive = mean similarity to the five correct captions.
        Negative = maximum similarity to any incorrect caption.

    Text -> Image:
        Positive = similarity to the matching image.
        Negative = maximum similarity to any other image.
    """
    similarity = image_embeddings @ text_embeddings.T

    num_images = similarity.shape[0]
    num_captions = similarity.shape[1]

    # Image -> Text
    i2t_positive = []
    i2t_hard_negative = []

    for image_idx in range(num_images):
        gt_start = image_idx * 5
        gt_end = gt_start + 5

        positive_scores = similarity[
            image_idx,
            gt_start:gt_end,
        ]

        positive_mean = positive_scores.mean().item()

        negative_mask = torch.ones(
            num_captions,
            dtype=torch.bool,
        )

        negative_mask[gt_start:gt_end] = False

        hardest_negative = similarity[
            image_idx
        ][negative_mask].max().item()

        i2t_positive.append(positive_mean)
        i2t_hard_negative.append(hardest_negative)

    i2t_positive = np.array(i2t_positive)
    i2t_hard_negative = np.array(i2t_hard_negative)

    i2t_margin = i2t_positive - i2t_hard_negative

    # Text -> Image
    t2i_positive = []
    t2i_hard_negative = []

    for caption_idx in range(num_captions):
        gt_image = int(caption_to_image[caption_idx])

        positive = similarity[
            gt_image,
            caption_idx,
        ].item()

        negative_mask = torch.ones(
            num_images,
            dtype=torch.bool,
        )

        negative_mask[gt_image] = False

        hardest_negative = similarity[
            :,
            caption_idx,
        ][negative_mask].max().item()

        t2i_positive.append(positive)
        t2i_hard_negative.append(hardest_negative)

    t2i_positive = np.array(t2i_positive)
    t2i_hard_negative = np.array(t2i_hard_negative)

    t2i_margin = t2i_positive - t2i_hard_negative

    return {
        "image_to_text_positive_mean": float(i2t_positive.mean()),
        "image_to_text_hard_negative_mean": float(
            i2t_hard_negative.mean()
        ),
        "image_to_text_margin_mean": float(i2t_margin.mean()),
        "text_to_image_positive_mean": float(t2i_positive.mean()),
        "text_to_image_hard_negative_mean": float(
            t2i_hard_negative.mean()
        ),
        "text_to_image_margin_mean": float(t2i_margin.mean()),
        "mean_margin": float(
            (i2t_margin.mean() + t2i_margin.mean()) / 2
        ),
    }


def main():
    global OUTPUT_DIR, RAW_OUTPUT, AGG_OUTPUT

    args = parse_args()
    budget = args.budget

    OUTPUT_DIR = (
        Path("results/analysis/alignment")
        if budget == 294912
        else Path("results/analysis/alignment/budget_589824")
    )

    RAW_OUTPUT = OUTPUT_DIR / "hard_negative_results.csv"
    AGG_OUTPUT = OUTPUT_DIR / "hard_negative_aggregate.csv"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Verify the complete set of checkpoints before running inference.
    expected_adapters = {
        (seed, allocation): adapter_path(
            allocation,
            seed,
            budget=budget,
        )
        for seed in SEEDS
        for allocation in ALLOCATIONS
    }

    missing_adapters = [
        path
        for path in expected_adapters.values()
        if not (path / "adapter_config.json").is_file()
    ]

    if missing_adapters:
        raise FileNotFoundError(
            "Missing adapter checkpoint(s) for the selected budget:\n"
            + "\n".join(str(path) for path in missing_adapters)
        )

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    print("=" * 80)
    print("HARD-NEGATIVE RETRIEVAL ANALYSIS")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Parameter budget: {budget:,}")

    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    images, captions, caption_to_image = load_test_data()

    # Frozen baseline
    print("\n" + "=" * 80)
    print("FROZEN CLIP")
    print("=" * 80)

    frozen_model = CLIPModel.from_pretrained(MODEL_NAME)
    frozen_model.to(device)

    frozen_images, frozen_texts = extract_all_embeddings(
        frozen_model,
        processor,
        images,
        captions,
        device,
    )

    cleanup_model(frozen_model)

    frozen_metrics = calculate_retrieval_margins(
        frozen_images,
        frozen_texts,
        caption_to_image,
    )

    print(
        "Frozen mean margin: "
        f"{frozen_metrics['mean_margin']:.6f}"
    )

    results = []

    for seed in SEEDS:
        for allocation in ALLOCATIONS:
            print("\n" + "=" * 80)
            print(f"Seed {seed} | Allocation {allocation}")
            print("=" * 80)

            path = expected_adapters[(seed, allocation)]

            if not path.exists():
                raise FileNotFoundError(
                    f"Adapter not found:\n{path}"
                )

            base_model = CLIPModel.from_pretrained(MODEL_NAME)

            model = PeftModel.from_pretrained(
                base_model,
                path,
                is_trainable=False,
            )

            model.to(device)

            adapted_images, adapted_texts = extract_all_embeddings(
                model,
                processor,
                images,
                captions,
                device,
            )

            metrics = calculate_retrieval_margins(
                adapted_images,
                adapted_texts,
                caption_to_image,
            )

            result = {
                "budget": budget,
                "seed": seed,
                "allocation": allocation,
                **metrics,
            }

            results.append(result)

            print(
                "Image -> Text margin: "
                f"{metrics['image_to_text_margin_mean']:.6f}"
            )
            print(
                "Text -> Image margin: "
                f"{metrics['text_to_image_margin_mean']:.6f}"
            )
            print(
                f"Mean margin: {metrics['mean_margin']:.6f}"
            )

            cleanup_model(model)

    # Save raw results
    df = pd.DataFrame(results)
    df.to_csv(RAW_OUTPUT, index=False)

    # Aggregate across seeds
    aggregate = (
        df.groupby("allocation")
        .agg(
            i2t_margin_mean=("image_to_text_margin_mean", "mean"),
            i2t_margin_std=("image_to_text_margin_mean", "std"),
            t2i_margin_mean=("text_to_image_margin_mean", "mean"),
            t2i_margin_std=("text_to_image_margin_mean", "std"),
            mean_margin_mean=("mean_margin", "mean"),
            mean_margin_std=("mean_margin", "std"),
        )
        .reset_index()
    )

    aggregate.to_csv(AGG_OUTPUT, index=False)

    print("\n" + "=" * 90)
    print("HARD-NEGATIVE SUMMARY")
    print("=" * 90)

    print(
        f"{'Allocation':<12}"
        f"{'I→T Margin':>16}"
        f"{'T→I Margin':>16}"
        f"{'Mean Margin':>16}"
    )

    print("-" * 60)

    for _, row in aggregate.iterrows():
        print(
            f"{row['allocation']:<12}"
            f"{row['i2t_margin_mean']:>16.6f}"
            f"{row['t2i_margin_mean']:>16.6f}"
            f"{row['mean_margin_mean']:>16.6f}"
        )

    print(
        "\nFrozen mean margin: "
        f"{frozen_metrics['mean_margin']:.6f}"
    )
    print(f"\nRaw results:\n{RAW_OUTPUT}")
    print(f"Aggregate results:\n{AGG_OUTPUT}")


if __name__ == "__main__":
    main()