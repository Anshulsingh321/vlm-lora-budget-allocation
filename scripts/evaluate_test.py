from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import CLIPModel, CLIPProcessor

from scripts.train import (
    Flickr30kEvalDataset,
    evaluate_retrieval,
)


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "openai/clip-vit-base-patch32"

ANNOTATIONS = "data/raw/flickr30k/flickr_annotations_30k.csv"
IMAGE_DIR = "data/processed/flickr30k/images"
TEST_SPLIT = "test"
BATCH_SIZE = 16

ALLOCATIONS = [
    "100_0",
    "75_25",
    "50_50",
    "25_75",
    "0_100",
]

SUPPORTED_BUDGETS = (294_912, 589_824)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate CLIP + LoRA adapters on Flickr30k."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Training seed whose adapters should be evaluated.",
    )

    parser.add_argument(
        "--budget",
        type=int,
        choices=SUPPORTED_BUDGETS,
        default=294_912,
        help="Trainable LoRA parameter budget.",
    )

    parser.add_argument(
        "--allocation",
        choices=ALLOCATIONS,
        default=None,
        help="Evaluate only one allocation.",
    )

    return parser.parse_args()


# ============================================================
# DEVICE
# ============================================================

def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# ADAPTER PATHS
# ============================================================

def build_adapter_path(
    allocation: str,
    seed: int,
    budget: int = 294_912,
) -> Path:
    """Return the correct adapter directory for a budget/allocation/seed."""

    if budget not in SUPPORTED_BUDGETS:
        raise ValueError(
            f"Unsupported budget {budget}. "
            f"Supported budgets: {SUPPORTED_BUDGETS}."
        )

    if budget == 294_912:
        # Seed 42 uses legacy directories without a seed suffix.
        if seed == 42:
            experiment_name = f"flickr30k_lora_{allocation}"
        else:
            experiment_name = (
                f"flickr30k_lora_{allocation}_seed{seed}"
            )
    else:
        # The larger budget uses budget- and seed-specific directories.
        experiment_name = (
            f"flickr30k_lora_{allocation}_budget{budget}_seed{seed}"
        )

    return (
        Path("results")
        / "experiments"
        / experiment_name
        / "best_adapter"
    )


# ============================================================
# CLEANUP
# ============================================================

def count_lora_parameters(model) -> int:
    """Count loaded LoRA A/B weights, even when frozen for evaluation."""

    return sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if "lora_A" in name or "lora_B" in name
    )


def cleanup_model(model):
    del model
    gc.collect()

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    seed = args.seed
    budget = args.budget
    device = select_device()

    # Preserve the historical output path for full 294,912 runs.
    # Isolate new-budget results and single-allocation pilot runs.
    if budget == 294_912 and args.allocation is None:
        output_dir = Path("results/test_evaluation")
        output_file = (
            output_dir / f"flickr30k_test_results_seed{seed}.json"
        )
    else:
        output_dir = (
            Path("results/test_evaluation") / f"budget_{budget}"
        )

        scope = args.allocation or "all_allocations"

        output_file = (
            output_dir
            / f"flickr30k_test_results_{scope}_seed{seed}.json"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FLICKR30K TEST EVALUATION")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Seed:   {seed}")
    print(f"Budget: {budget:,} trainable LoRA parameters")

    if args.allocation:
        print(f"Allocation filter: {args.allocation}")

    # --------------------------------------------------------
    # PROCESSOR
    # --------------------------------------------------------

    print("\nLoading CLIP processor...")

    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    # --------------------------------------------------------
    # TEST DATASET
    # --------------------------------------------------------

    print("\nLoading Flickr30k test dataset...")

    test_dataset = Flickr30kEvalDataset(
        csv_path=ANNOTATIONS,
        image_dir=IMAGE_DIR,
        split=TEST_SPLIT,
    )

    print(f"Test images: {len(test_dataset):,}")

    if len(test_dataset) != 1000:
        raise RuntimeError(
            "Expected exactly 1,000 Flickr30k test images, "
            f"got {len(test_dataset)}."
        )

    results = {}

    # --------------------------------------------------------
    # FROZEN BASELINE
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FROZEN CLIP BASELINE")
    print("=" * 70)

    frozen_model = CLIPModel.from_pretrained(MODEL_NAME)
    frozen_model.to(device)

    frozen_metrics = evaluate_retrieval(
        model=frozen_model,
        processor=processor,
        dataset=test_dataset,
        device=device,
        batch_size=BATCH_SIZE,
    )

    results["frozen"] = frozen_metrics

    print("\nFrozen CLIP test results:")

    print(
        f"Image -> Text  "
        f"R@1={frozen_metrics['image_to_text']['R@1']:.4f}  "
        f"R@5={frozen_metrics['image_to_text']['R@5']:.4f}  "
        f"R@10={frozen_metrics['image_to_text']['R@10']:.4f}"
    )

    print(
        f"Text -> Image  "
        f"R@1={frozen_metrics['text_to_image']['R@1']:.4f}  "
        f"R@5={frozen_metrics['text_to_image']['R@5']:.4f}  "
        f"R@10={frozen_metrics['text_to_image']['R@10']:.4f}"
    )

    print(f"Mean R@1: {frozen_metrics['mean_R@1']:.4f}")

    cleanup_model(frozen_model)

    # --------------------------------------------------------
    # LORA ADAPTERS
    # --------------------------------------------------------

    allocations_to_evaluate = (
        [args.allocation] if args.allocation else ALLOCATIONS
    )

    for allocation in allocations_to_evaluate:
        print("\n" + "=" * 70)
        print(f"LoRA TEST EVALUATION: {allocation}")
        print("=" * 70)

        adapter_path = build_adapter_path(
            allocation=allocation,
            seed=seed,
            budget=budget,
        )

        print(f"\nExpected adapter:\n{adapter_path}")

        if not adapter_path.exists():
            raise FileNotFoundError(
                f"\nAdapter not found:\n{adapter_path}\n\n"
                "Check that the corresponding training run has completed."
            )

        # Load base CLIP.
        print("\nLoading base CLIP...")

        base_model = CLIPModel.from_pretrained(MODEL_NAME)

        # Load LoRA adapter.
        print("Loading LoRA adapter...")

        model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            is_trainable=False,
        )

        model.to(device)

        adapter_params = count_lora_parameters(model)

        print(
            f"LoRA adapter parameters loaded: {adapter_params:,}"
        )

        if adapter_params != budget:
            raise RuntimeError(
                f"Loaded adapter has {adapter_params:,} LoRA parameters, "
                f"but budget {budget:,} was requested."
            )

        metrics = evaluate_retrieval(
            model=model,
            processor=processor,
            dataset=test_dataset,
            device=device,
            batch_size=BATCH_SIZE,
        )

        results[allocation] = metrics

        print(f"\n{allocation} test results:")

        print(
            f"Image -> Text  "
            f"R@1={metrics['image_to_text']['R@1']:.4f}  "
            f"R@5={metrics['image_to_text']['R@5']:.4f}  "
            f"R@10={metrics['image_to_text']['R@10']:.4f}"
        )

        print(
            f"Text -> Image  "
            f"R@1={metrics['text_to_image']['R@1']:.4f}  "
            f"R@5={metrics['text_to_image']['R@5']:.4f}  "
            f"R@10={metrics['text_to_image']['R@10']:.4f}"
        )

        print(f"Mean R@1: {metrics['mean_R@1']:.4f}")

        cleanup_model(model)

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": seed,
                "budget": budget,
                "allocation_filter": args.allocation,
                "results": results,
            },
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # FINAL TABLE
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        f"FINAL FLICKR30K TEST RESULTS — "
        f"BUDGET {budget:,} — SEED {seed}"
    )
    print("=" * 70)

    print(
        f"{'Setup':<12}"
        f"{'I->T R@1':>12}"
        f"{'T->I R@1':>12}"
        f"{'Mean R@1':>12}"
    )

    print("-" * 48)

    frozen = results["frozen"]

    print(
        f"{'Frozen':<12}"
        f"{frozen['image_to_text']['R@1']:>12.4f}"
        f"{frozen['text_to_image']['R@1']:>12.4f}"
        f"{frozen['mean_R@1']:>12.4f}"
    )

    for allocation in allocations_to_evaluate:
        metrics = results[allocation]

        print(
            f"{allocation:<12}"
            f"{metrics['image_to_text']['R@1']:>12.4f}"
            f"{metrics['text_to_image']['R@1']:>12.4f}"
            f"{metrics['mean_R@1']:>12.4f}"
        )

    print("-" * 48)
    print(f"\nResults saved to:\n{output_file}")


if __name__ == "__main__":
    main()