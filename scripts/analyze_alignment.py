from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm

from scripts.train import Flickr30kEvalDataset


MODEL_NAME = "openai/clip-vit-base-patch32"
ANNOTATIONS = "data/raw/flickr30k/flickr_annotations_30k.csv"
IMAGE_DIR = "data/processed/flickr30k/images"
TEST_SPLIT = "test"
BATCH_SIZE = 16
SEEDS = [42, 123, 456]

ALLOCATIONS = [
    "100_0",
    "75_25",
    "50_50",
    "25_75",
    "0_100",
]

NUM_STRUCTURE_PAIRS = 100_000
RANDOM_SEED = 2026

OUTPUT_DIR = Path("results/analysis/alignment")
CSV_OUTPUT = OUTPUT_DIR / "alignment_results.csv"
JSON_OUTPUT = OUTPUT_DIR / "alignment_results.json"


def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def cleanup_model(model):
    del model
    gc.collect()

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def extract_embedding(output):
    if isinstance(output, torch.Tensor):
        return output

    if hasattr(output, "pooler_output"):
        return output.pooler_output

    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state

    raise TypeError(f"Unsupported output type: {type(output)}")


def normalize_embeddings(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


@torch.no_grad()
def encode_images(model, processor, images, device):
    inputs = processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    output = model.get_image_features(pixel_values=pixel_values)
    features = normalize_embeddings(extract_embedding(output))

    return features.cpu()


@torch.no_grad()
def encode_texts(model, processor, texts, device):
    inputs = processor(
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    output = model.get_text_features(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
    )

    features = normalize_embeddings(extract_embedding(output))

    return features.cpu()


def load_test_data():
    from PIL import Image

    dataset = Flickr30kEvalDataset(
        csv_path=ANNOTATIONS,
        image_dir=IMAGE_DIR,
        split=TEST_SPLIT,
    )

    if len(dataset) != 1000:
        raise RuntimeError(
            f"Expected 1,000 test images, got {len(dataset)}"
        )

    images = []
    captions = []
    caption_to_image = []

    for image_index, row in enumerate(dataset.rows):
        image_path = dataset.image_dir / row["filename"]

        image = Image.open(image_path).convert("RGB")
        images.append(image)

        for caption in row["captions"]:
            captions.append(caption)
            caption_to_image.append(image_index)

    return (
        images,
        captions,
        np.array(caption_to_image, dtype=np.int64),
    )


def extract_all_embeddings(model, processor, images, captions, device):
    image_embeddings = []

    for start in tqdm(
        range(0, len(images), BATCH_SIZE),
        desc="Encoding images",
    ):
        end = min(start + BATCH_SIZE, len(images))

        features = encode_images(
            model,
            processor,
            images[start:end],
            device,
        )
        image_embeddings.append(features)

    image_embeddings = torch.cat(image_embeddings, dim=0)

    text_embeddings = []

    for start in tqdm(
        range(0, len(captions), BATCH_SIZE),
        desc="Encoding captions",
    ):
        end = min(start + BATCH_SIZE, len(captions))

        features = encode_texts(
            model,
            processor,
            captions[start:end],
            device,
        )
        text_embeddings.append(features)

    text_embeddings = torch.cat(text_embeddings, dim=0)

    return image_embeddings, text_embeddings


def calculate_drift(frozen_embeddings, adapted_embeddings):
    cosine = (
        frozen_embeddings * adapted_embeddings
    ).sum(dim=1)

    mean_cosine = cosine.mean().item()
    mean_drift = 1.0 - mean_cosine

    return mean_cosine, mean_drift


def calculate_paired_alignment(
    image_embeddings,
    text_embeddings,
    caption_to_image,
):
    image_vectors = image_embeddings[
        torch.from_numpy(caption_to_image)
    ]

    similarities = (
        image_vectors * text_embeddings
    ).sum(dim=1)

    return similarities


def calculate_structure_correlation(
    frozen_images,
    frozen_texts,
    adapted_images,
    adapted_texts,
    num_pairs=100_000,
    seed=2026,
):
    rng = np.random.default_rng(seed)

    num_images = frozen_images.shape[0]
    num_texts = frozen_texts.shape[0]
    total_pairs = num_images * num_texts

    sample_size = min(num_pairs, total_pairs)

    flat_indices = rng.choice(
        total_pairs,
        size=sample_size,
        replace=False,
    )

    image_indices = torch.from_numpy(
        flat_indices // num_texts
    )
    text_indices = torch.from_numpy(
        flat_indices % num_texts
    )

    frozen_scores = (
        frozen_images[image_indices]
        * frozen_texts[text_indices]
    ).sum(dim=1)

    adapted_scores = (
        adapted_images[image_indices]
        * adapted_texts[text_indices]
    ).sum(dim=1)

    correlation = np.corrcoef(
        frozen_scores.numpy(),
        adapted_scores.numpy(),
    )[0, 1]

    return float(correlation)


def adapter_path(allocation, seed, budget=294912):
    """Resolve the adapter for the selected budget, allocation, and seed."""
    experiments_dir = Path("results") / "experiments"

    if budget == 589824:
        candidates = [
            experiments_dir
            / f"flickr30k_lora_{allocation}_budget589824_seed{seed}"
            / "best_adapter"
        ]

    elif budget == 294912:
        legacy = (
            experiments_dir
            / f"flickr30k_lora_{allocation}"
            / "best_adapter"
        )

        seed_specific = (
            experiments_dir
            / f"flickr30k_lora_{allocation}_seed{seed}"
            / "best_adapter"
        )

        # Seed 42 uses the legacy directory.
        # Seeds 123 and 456 must use their own directories.
        candidates = [legacy] if seed == 42 else [seed_specific]

    else:
        raise ValueError(
            f"Unsupported parameter budget: {budget}. "
            "Choose 294912 or 589824."
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Flickr30k CLIP representation drift and alignment."
        )
    )

    parser.add_argument(
        "--budget",
        type=int,
        choices=(294912, 589824),
        default=294912,
        help="LoRA parameter budget (default: 294912).",
    )

    return parser.parse_args()


def main():
    global OUTPUT_DIR, CSV_OUTPUT, JSON_OUTPUT

    args = parse_args()
    budget = args.budget

    OUTPUT_DIR = (
        Path("results/analysis/alignment")
        if budget == 294912
        else Path("results/analysis/alignment/budget_589824")
    )

    CSV_OUTPUT = OUTPUT_DIR / "alignment_results.csv"
    JSON_OUTPUT = OUTPUT_DIR / "alignment_results.json"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = select_device()

    print("=" * 80)
    print("CLIP ALIGNMENT / REPRESENTATION ANALYSIS")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Parameter budget: {budget:,}")

    # Check all adapter paths before expensive model inference.
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

    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    print("\nLoading Flickr30k test data...")

    images, captions, caption_to_image = load_test_data()

    print(f"Images:   {len(images):,}")
    print(f"Captions: {len(captions):,}")

    # Frozen CLIP baseline
    print("\n" + "=" * 80)
    print("FROZEN CLIP")
    print("=" * 80)

    frozen_model = CLIPModel.from_pretrained(MODEL_NAME)
    frozen_model.to(device)

    (
        frozen_image_embeddings,
        frozen_text_embeddings,
    ) = extract_all_embeddings(
        frozen_model,
        processor,
        images,
        captions,
        device,
    )

    cleanup_model(frozen_model)

    print(
        "Image embedding shape: "
        f"{tuple(frozen_image_embeddings.shape)}"
    )
    print(
        "Text embedding shape: "
        f"{tuple(frozen_text_embeddings.shape)}"
    )

    frozen_paired = calculate_paired_alignment(
        frozen_image_embeddings,
        frozen_text_embeddings,
        caption_to_image,
    )

    frozen_paired_mean = frozen_paired.mean().item()

    print(
        f"Frozen paired alignment: {frozen_paired_mean:.6f}"
    )

    # Evaluate each seed and allocation.
    all_results = []

    for seed in SEEDS:
        for allocation in ALLOCATIONS:
            print("\n" + "=" * 80)
            print(f"Seed {seed} | Allocation {allocation}")
            print("=" * 80)

            path = expected_adapters[(seed, allocation)]

            print(f"Adapter:\n{path}")

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

            (
                adapted_image_embeddings,
                adapted_text_embeddings,
            ) = extract_all_embeddings(
                model,
                processor,
                images,
                captions,
                device,
            )

            image_cosine, image_drift = calculate_drift(
                frozen_image_embeddings,
                adapted_image_embeddings,
            )

            text_cosine, text_drift = calculate_drift(
                frozen_text_embeddings,
                adapted_text_embeddings,
            )

            adapted_paired = calculate_paired_alignment(
                adapted_image_embeddings,
                adapted_text_embeddings,
                caption_to_image,
            )

            adapted_paired_mean = adapted_paired.mean().item()

            paired_delta = (
                adapted_paired_mean - frozen_paired_mean
            )

            structure_corr = calculate_structure_correlation(
                frozen_image_embeddings,
                frozen_text_embeddings,
                adapted_image_embeddings,
                adapted_text_embeddings,
                num_pairs=NUM_STRUCTURE_PAIRS,
                seed=RANDOM_SEED,
            )

            result = {
                "budget": budget,
                "seed": seed,
                "allocation": allocation,
                "image_embedding_cosine": image_cosine,
                "image_representation_drift": image_drift,
                "text_embedding_cosine": text_cosine,
                "text_representation_drift": text_drift,
                "frozen_paired_alignment": frozen_paired_mean,
                "adapted_paired_alignment": adapted_paired_mean,
                "paired_alignment_delta": paired_delta,
                "structure_correlation": structure_corr,
            }

            all_results.append(result)

            print(f"\nImage embedding cosine: {image_cosine:.6f}")
            print(f"Image representation drift: {image_drift:.6f}")
            print(f"Text embedding cosine: {text_cosine:.6f}")
            print(f"Text representation drift: {text_drift:.6f}")
            print(f"Adapted paired alignment: {adapted_paired_mean:.6f}")
            print(f"Paired alignment delta: {paired_delta:+.6f}")
            print(f"Structure correlation: {structure_corr:.6f}")

            cleanup_model(model)

    # Save raw results.
    with open(JSON_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": MODEL_NAME,
                "dataset": "Flickr30k",
                "budget": budget,
                "test_images": len(images),
                "test_captions": len(captions),
                "structure_pairs": NUM_STRUCTURE_PAIRS,
                "random_seed": RANDOM_SEED,
                "frozen_paired_alignment": frozen_paired_mean,
                "results": all_results,
            },
            f,
            indent=2,
        )

    df = pd.DataFrame(all_results)

    aggregate = (
        df.groupby("allocation")
        .agg(
            image_cosine_mean=("image_embedding_cosine", "mean"),
            image_cosine_std=("image_embedding_cosine", "std"),
            image_drift_mean=("image_representation_drift", "mean"),
            image_drift_std=("image_representation_drift", "std"),
            text_cosine_mean=("text_embedding_cosine", "mean"),
            text_cosine_std=("text_embedding_cosine", "std"),
            text_drift_mean=("text_representation_drift", "mean"),
            text_drift_std=("text_representation_drift", "std"),
            paired_alignment_mean=("adapted_paired_alignment", "mean"),
            paired_alignment_std=("adapted_paired_alignment", "std"),
            paired_delta_mean=("paired_alignment_delta", "mean"),
            paired_delta_std=("paired_alignment_delta", "std"),
            structure_corr_mean=("structure_correlation", "mean"),
            structure_corr_std=("structure_correlation", "std"),
        )
        .reset_index()
    )

    aggregate.to_csv(
        OUTPUT_DIR / "alignment_aggregate.csv",
        index=False,
    )

    df.to_csv(CSV_OUTPUT, index=False)

    print("\n" + "=" * 100)
    print("ALIGNMENT SUMMARY")
    print("=" * 100)

    print(
        f"{'Allocation':<12}"
        f"{'Img Drift':>14}"
        f"{'Text Drift':>14}"
        f"{'Pair Δ':>14}"
        f"{'Structure Corr':>18}"
    )

    print("-" * 100)

    for _, row in aggregate.iterrows():
        print(
            f"{row['allocation']:<12}"
            f"{row['image_drift_mean']:>14.6f}"
            f"{row['text_drift_mean']:>14.6f}"
            f"{row['paired_delta_mean']:>14.6f}"
            f"{row['structure_corr_mean']:>18.6f}"
        )

    print(f"\nFrozen paired alignment: {frozen_paired_mean:.6f}")
    print(f"\nRaw results saved to:\n{CSV_OUTPUT}")
    print(f"JSON saved to:\n{JSON_OUTPUT}")
    print(
        "Aggregate results saved to:\n"
        f"{OUTPUT_DIR / 'alignment_aggregate.csv'}"
    )


if __name__ == "__main__":
    main()