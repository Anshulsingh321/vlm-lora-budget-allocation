from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import statistics
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


MODEL_NAME = "openai/clip-vit-base-patch32"
WDS_DATASET = "undefined443/coco-karpathy-wds"

METADATA_PATH = (
    Path("data")
    / "raw"
    / "coco"
    / "karpathy"
    / "coco_test_metadata.json"
)

COCO_RETRIEVAL_SUMMARY = (
    Path("results")
    / "analysis"
    / "coco"
    / "coco_summary.csv"
)

RESULTS_DIR = (
    Path("results")
    / "analysis"
    / "coco"
    / "hard_negatives"
)

EXPERIMENTS_DIR = Path("results") / "experiments"

ALLOCATIONS = [
    "100_0",
    "75_25",
    "50_50",
    "25_75",
    "0_100",
]

SEEDS = [42, 123, 456]

EXPECTED_IMAGES = 5000
CAPTIONS_PER_IMAGE = 5

DEFAULT_IMAGE_BATCH_SIZE = 32
DEFAULT_TEXT_BATCH_SIZE = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Hard-negative retrieval analysis on the verified "
            "COCO Karpathy 5K benchmark."
        )
    )

    parser.add_argument(
        "--budget",
        type=int,
        choices=(294912, 589824),
        default=294912,
        help="LoRA parameter budget (default: 294912).",
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "mps", "cuda", "cpu"),
        help="Evaluation device.",
    )

    parser.add_argument(
        "--image-batch-size",
        type=int,
        default=DEFAULT_IMAGE_BATCH_SIZE,
        help="Image encoding batch size.",
    )

    parser.add_argument(
        "--text-batch-size",
        type=int,
        default=DEFAULT_TEXT_BATCH_SIZE,
        help="Text encoding batch size.",
    )

    parser.add_argument(
        "--allocation",
        choices=ALLOCATIONS,
        help="Evaluate only one LoRA allocation.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        choices=SEEDS,
        help="Evaluate only one LoRA seed.",
    )

    parser.add_argument(
        "--skip-frozen",
        action="store_true",
        help="Skip frozen CLIP.",
    )

    parser.add_argument(
        "--skip-lora",
        action="store_true",
        help="Skip all LoRA adapters.",
    )

    parser.add_argument(
        "--limit-images",
        type=int,
        default=None,
        help=(
            "Debug-only subset. Do not use subset results as "
            "paper results."
        ),
    )

    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS was requested but is not available."
            )
        return torch.device("mps")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available."
            )
        return torch.device("cuda")

    return torch.device("cpu")


def clear_memory() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if (
        hasattr(torch, "mps")
        and torch.backends.mps.is_available()
    ):
        torch.mps.empty_cache()


def load_metadata(
    limit_images: int | None = None,
) -> tuple[list[int], list[list[str]]]:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            "COCO metadata not found:\n"
            f"{METADATA_PATH}\n\n"
            "Run:\n"
            "python -m scripts.verify_coco"
        )

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if not isinstance(metadata, dict):
        raise RuntimeError(
            "COCO metadata must be a JSON object."
        )

    if len(metadata) != EXPECTED_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_IMAGES:,} images; "
            f"found {len(metadata):,}."
        )

    items = sorted(
        metadata.values(),
        key=lambda item: int(item["cocoid"]),
    )

    if limit_images is not None:
        if not 1 <= limit_images <= EXPECTED_IMAGES:
            raise ValueError(
                f"--limit-images must be between "
                f"1 and {EXPECTED_IMAGES}."
            )
        items = items[:limit_images]

    image_ids: list[int] = []
    captions_by_image: list[list[str]] = []

    for item in items:
        required = {
            "cocoid",
            "filename",
            "filepath",
            "sentids",
            "captions",
        }

        missing = required - set(item.keys())

        if missing:
            raise RuntimeError(
                f"Metadata entry is missing fields: {sorted(missing)}"
            )

        coco_id = int(item["cocoid"])
        captions = item.get("captions")

        if not isinstance(captions, list):
            raise RuntimeError(
                f"COCO ID {coco_id} has invalid captions."
            )

        if len(captions) != CAPTIONS_PER_IMAGE:
            raise RuntimeError(
                f"COCO ID {coco_id} has {len(captions)} captions; "
                f"expected {CAPTIONS_PER_IMAGE}."
            )

        for caption in captions:
            if not isinstance(caption, str) or not caption.strip():
                raise RuntimeError(
                    f"COCO ID {coco_id} contains an invalid/empty caption."
                )

        image_ids.append(coco_id)
        captions_by_image.append(captions)

    if not image_ids:
        raise RuntimeError("No COCO images were selected.")

    return image_ids, captions_by_image


def find_adapter_path(
    allocation: str,
    seed: int,
    budget: int = 294912,
) -> Path:
    """Find the adapter for the selected budget/allocation/seed."""
    if budget == 589824:
        experiment_dirs = [
            EXPERIMENTS_DIR
            / f"flickr30k_lora_{allocation}_budget589824_seed{seed}"
        ]

    elif budget == 294912:
        seed_specific = (
            EXPERIMENTS_DIR
            / f"flickr30k_lora_{allocation}_seed{seed}"
        )
        legacy = EXPERIMENTS_DIR / f"flickr30k_lora_{allocation}"

        # The legacy directory is only the seed-42 directory.
        experiment_dirs = [legacy] if seed == 42 else [seed_specific]

    else:
        raise ValueError(
            f"Unsupported parameter budget: {budget}. "
            "Choose 294912 or 589824."
        )

    existing_dirs = [
        path
        for path in experiment_dirs
        if path.exists()
    ]

    if not existing_dirs:
        checked = "\n".join(
            f"  {path}"
            for path in experiment_dirs
        )
        raise FileNotFoundError(
            "No experiment directory found for "
            f"budget={budget}, allocation={allocation}, seed={seed}.\n"
            f"Checked:\n{checked}"
        )

    for experiment_dir in existing_dirs:
        direct_best = experiment_dir / "best_adapter"

        if (direct_best / "adapter_config.json").exists():
            return direct_best

        configs = list(experiment_dir.rglob("adapter_config.json"))

        best_configs = [
            path
            for path in configs
            if any("best" in part.lower() for part in path.parts)
        ]

        if len(best_configs) == 1:
            return best_configs[0].parent

        if len(configs) == 1:
            return configs[0].parent

        if len(configs) > 1:
            raise RuntimeError(
                "Multiple adapters were found and no unique 'best' "
                "adapter could be identified.\n"
                f"Experiment directory: {experiment_dir}\n"
                + "\n".join(
                    f"  - {path.parent}"
                    for path in configs
                )
            )

    raise FileNotFoundError(
        "No adapter_config.json found under the checked directories:\n"
        + "\n".join(str(path) for path in existing_dirs)
    )


def get_clip_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, PeftModel):
        return model.get_base_model()
    return model


def load_model(
    adapter_path: Path | None,
    device: torch.device,
) -> tuple[torch.nn.Module, CLIPProcessor]:
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    base_model = CLIPModel.from_pretrained(MODEL_NAME)

    if adapter_path is None:
        model = base_model
    else:
        model = PeftModel.from_pretrained(
            base_model,
            str(adapter_path),
            is_trainable=False,
        )

    model.to(device)
    model.eval()

    clip_model = get_clip_model(model)

    required = (
        "vision_model",
        "visual_projection",
        "text_model",
        "text_projection",
    )

    missing = [
        name
        for name in required
        if not hasattr(clip_model, name)
    ]

    if missing:
        raise RuntimeError(
            "Loaded model is missing required CLIP components: "
            f"{missing}"
        )

    return model, processor


@torch.inference_mode()
def encode_images(
    model: torch.nn.Module,
    processor: CLIPProcessor,
    image_ids: list[int],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    dataset = load_dataset(
        WDS_DATASET,
        split="test",
        streaming=True,
    )

    clip_model = get_clip_model(model)

    num_images = len(image_ids)

    id_to_index = {
        coco_id: index
        for index, coco_id in enumerate(image_ids)
    }

    seen_ids: set[int] = set()
    batch_images: list[Image.Image] = []
    batch_ids: list[int] = []
    embeddings = None

    def flush_batch() -> None:
        nonlocal embeddings
        nonlocal batch_images
        nonlocal batch_ids

        if not batch_images:
            return

        inputs = processor(
            images=batch_images,
            return_tensors="pt",
        )

        pixel_values = inputs["pixel_values"].to(device)

        vision_outputs = clip_model.vision_model(
            pixel_values=pixel_values
        )

        batch_embeddings = clip_model.visual_projection(
            vision_outputs.pooler_output
        )

        batch_embeddings = (
            batch_embeddings
            / batch_embeddings.norm(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-12)
        )

        batch_embeddings = (
            batch_embeddings.detach().float().cpu()
        )

        if embeddings is None:
            embedding_dim = batch_embeddings.shape[1]

            embeddings = torch.empty(
                (num_images, embedding_dim),
                dtype=torch.float32,
            )

        indices = [
            id_to_index[coco_id]
            for coco_id in batch_ids
        ]

        embeddings[indices] = batch_embeddings

        batch_images = []
        batch_ids = []

    scanned = 0

    for sample in dataset:
        scanned += 1

        if "jpg" not in sample:
            raise RuntimeError("WDS sample is missing 'jpg'.")

        if "json" not in sample:
            raise RuntimeError("WDS sample is missing 'json'.")

        metadata = sample["json"]

        if "cocoid" not in metadata:
            raise RuntimeError("WDS metadata is missing 'cocoid'.")

        coco_id = int(metadata["cocoid"])

        # The streamed dataset may contain non-target images.
        if coco_id not in id_to_index:
            continue

        if coco_id in seen_ids:
            raise RuntimeError(
                f"Duplicate target COCO ID in WDS: {coco_id}"
            )

        image = sample["jpg"]

        if not isinstance(image, Image.Image):
            raise RuntimeError(
                f"COCO ID {coco_id} did not return a PIL image."
            )

        if image.mode != "RGB":
            image = image.convert("RGB")

        batch_images.append(image)
        batch_ids.append(coco_id)
        seen_ids.add(coco_id)

        if len(batch_images) >= batch_size:
            flush_batch()

        if len(seen_ids) >= num_images:
            break

        if scanned % 1000 == 0 and num_images < EXPECTED_IMAGES:
            print(
                f"  WDS scanned: {scanned:,} | "
                f"targets collected: {len(seen_ids):,}/{num_images:,}"
            )

    flush_batch()

    missing = set(image_ids) - seen_ids

    if missing:
        raise RuntimeError(
            "Could not collect all requested COCO images.\n"
            f"Collected: {len(seen_ids):,}/{num_images:,}\n"
            f"Missing examples: {sorted(missing)[:20]}"
        )

    if embeddings is None:
        raise RuntimeError("No image embeddings were generated.")

    return embeddings


@torch.inference_mode()
def encode_texts(
    model: torch.nn.Module,
    processor: CLIPProcessor,
    captions_by_image: list[list[str]],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    clip_model = get_clip_model(model)

    flat_captions = [
        caption
        for captions in captions_by_image
        for caption in captions
    ]

    batches = []

    for start in range(0, len(flat_captions), batch_size):
        batch = flat_captions[start:start + batch_size]

        inputs = processor(
            text=batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")

        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        text_outputs = clip_model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        batch_embeddings = clip_model.text_projection(
            text_outputs.pooler_output
        )

        batch_embeddings = (
            batch_embeddings
            / batch_embeddings.norm(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-12)
        )

        batches.append(
            batch_embeddings.detach().float().cpu()
        )

    if not batches:
        raise RuntimeError("No text embeddings were generated.")

    return torch.cat(batches, dim=0)


def compute_hard_negative_margins(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    num_images: int,
) -> dict[str, float]:
    """
    Image -> Text:
      Positive = mean similarity to five correct captions.
      Negative = highest similarity to any incorrect caption.

    Text -> Image:
      Positive = similarity to matching image.
      Negative = highest similarity to any other image.
    """
    expected_texts = num_images * CAPTIONS_PER_IMAGE

    if image_embeddings.shape[0] != num_images:
        raise RuntimeError(
            "Image embedding count does not match num_images."
        )

    if text_embeddings.shape[0] != expected_texts:
        raise RuntimeError(
            f"Text embedding count does not equal {expected_texts:,}."
        )

    # Image -> Text
    i2t_margin_sum = 0.0
    i2t_chunk_size = 32

    for start in range(0, num_images, i2t_chunk_size):
        end = min(start + i2t_chunk_size, num_images)

        similarities = (
            image_embeddings[start:end]
            @ text_embeddings.T
        )

        for local_index in range(end - start):
            image_index = start + local_index
            row = similarities[local_index]

            positive_start = image_index * CAPTIONS_PER_IMAGE
            positive_end = positive_start + CAPTIONS_PER_IMAGE

            positive_score = row[
                positive_start:positive_end
            ].mean().item()

            if positive_start > 0:
                left_max = row[:positive_start].max().item()
            else:
                left_max = float("-inf")

            if positive_end < row.shape[0]:
                right_max = row[positive_end:].max().item()
            else:
                right_max = float("-inf")

            hard_negative_score = max(left_max, right_max)

            i2t_margin_sum += (
                positive_score - hard_negative_score
            )

    i2t_margin = i2t_margin_sum / num_images

    # Text -> Image
    t2i_margin_sum = 0.0
    t2i_chunk_size = 256

    for start in range(0, expected_texts, t2i_chunk_size):
        end = min(start + t2i_chunk_size, expected_texts)

        similarities = (
            text_embeddings[start:end]
            @ image_embeddings.T
        )

        for local_index in range(end - start):
            caption_index = start + local_index
            ground_truth_image = caption_index // CAPTIONS_PER_IMAGE
            row = similarities[local_index]

            positive_score = row[ground_truth_image].item()

            if ground_truth_image == 0:
                hard_negative_score = row[1:].max().item()

            elif ground_truth_image == num_images - 1:
                hard_negative_score = row[:-1].max().item()

            else:
                left_max = row[:ground_truth_image].max().item()
                right_max = row[ground_truth_image + 1:].max().item()

                hard_negative_score = max(left_max, right_max)

            t2i_margin_sum += (
                positive_score - hard_negative_score
            )

    t2i_margin = t2i_margin_sum / expected_texts

    return {
        "i2t_margin": i2t_margin,
        "t2i_margin": t2i_margin,
        "mean_margin": (i2t_margin + t2i_margin) / 2.0,
    }


def load_retrieval_results() -> dict[tuple[str, int | None], float]:
    results = {}

    if not COCO_RETRIEVAL_SUMMARY.exists():
        print(
            "\nCOCO retrieval summary not found:"
            f"\n{COCO_RETRIEVAL_SUMMARY}"
        )
        print("Retrieval/margin correlation will be skipped.")
        return results

    with open(
        COCO_RETRIEVAL_SUMMARY,
        "r",
        encoding="utf-8",
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            row_type = row.get("type", "")

            if row_type == "frozen":
                results[("frozen", None)] = float(row["mean_r@1"])

            elif row_type == "lora":
                allocation = row["allocation"]
                seed = int(row["seed"])

                results[(allocation, seed)] = float(
                    row["mean_r@1"]
                )

    return results


def pearson_correlation(
    xs: list[float],
    ys: list[float],
) -> float | None:
    if len(xs) != len(ys):
        raise ValueError("x/y length mismatch.")

    if len(xs) < 2:
        return None

    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)

    numerator = sum(
        (x - mean_x) * (y - mean_y)
        for x, y in zip(xs, ys)
    )

    denom_x = math.sqrt(
        sum((x - mean_x) ** 2 for x in xs)
    )

    denom_y = math.sqrt(
        sum((y - mean_y) ** 2 for y in ys)
    )

    denominator = denom_x * denom_y

    if denominator == 0:
        return None

    return numerator / denominator


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "name",
        "budget",
        "type",
        "allocation",
        "seed",
        "num_images",
        "i2t_margin",
        "t2i_margin",
        "mean_margin",
        "mean_r@1",
        "elapsed_seconds",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_aggregate_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "allocation",
        "num_seeds",
        "mean_i2t_margin",
        "std_i2t_margin",
        "mean_t2i_margin",
        "std_t2i_margin",
        "mean_margin",
        "std_margin",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_results(rows: list[dict]) -> list[dict]:
    aggregate = []

    for allocation in ALLOCATIONS:
        selected = [
            row
            for row in rows
            if row["type"] == "lora"
            and row["allocation"] == allocation
        ]

        if not selected:
            continue

        i2t = [float(row["i2t_margin"]) for row in selected]
        t2i = [float(row["t2i_margin"]) for row in selected]
        mean = [float(row["mean_margin"]) for row in selected]

        aggregate.append(
            {
                "allocation": allocation,
                "num_seeds": len(selected),
                "mean_i2t_margin": statistics.mean(i2t),
                "std_i2t_margin": (
                    statistics.stdev(i2t)
                    if len(i2t) > 1
                    else 0.0
                ),
                "mean_t2i_margin": statistics.mean(t2i),
                "std_t2i_margin": (
                    statistics.stdev(t2i)
                    if len(t2i) > 1
                    else 0.0
                ),
                "mean_margin": statistics.mean(mean),
                "std_margin": (
                    statistics.stdev(mean)
                    if len(mean) > 1
                    else 0.0
                ),
            }
        )

    return aggregate


def main() -> None:
    global RESULTS_DIR, COCO_RETRIEVAL_SUMMARY

    args = parse_args()
    budget = args.budget

    if budget == 294912:
        RESULTS_DIR = Path("results/analysis/coco/hard_negatives")
        COCO_RETRIEVAL_SUMMARY = Path(
            "results/analysis/coco/coco_summary.csv"
        )
    else:
        RESULTS_DIR = Path(
            "results/analysis/coco/budget_589824/hard_negatives"
        )
        COCO_RETRIEVAL_SUMMARY = Path(
            "results/analysis/coco/budget_589824/coco_summary.csv"
        )

    if args.image_batch_size <= 0 or args.text_batch_size <= 0:
        raise ValueError("Batch sizes must be positive.")

    device = resolve_device(args.device)

    image_ids, captions_by_image = load_metadata(
        limit_images=args.limit_images
    )

    if args.limit_images is not None:
        print("\nWARNING: --limit-images is enabled.")
        print("These results are DEBUG ONLY.")

    print("=" * 80)
    print("COCO HARD-NEGATIVE RETRIEVAL ANALYSIS")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Parameter budget: {budget:,}")
    print(f"Device: {device}")
    print(f"Images: {len(image_ids):,}")
    print(
        f"Captions: {len(image_ids) * CAPTIONS_PER_IMAGE:,}"
    )

    print("\nHard-negative definition:")
    print("  I→T positive = mean similarity to 5 matched captions")
    print("  I→T negative = highest similarity to a non-matching caption")
    print("  T→I positive = similarity to matching image")
    print("  T→I negative = highest similarity to a non-matching image")

    results = []

    allocations = (
        [args.allocation]
        if args.allocation
        else ALLOCATIONS
    )

    seeds = (
        [args.seed]
        if args.seed
        else SEEDS
    )

    adapter_paths = {}

    if not args.skip_lora:
        for allocation in allocations:
            for seed in seeds:
                adapter_paths[(allocation, seed)] = find_adapter_path(
                    allocation=allocation,
                    seed=seed,
                    budget=budget,
                )

    retrieval_results = load_retrieval_results()

    # Frozen CLIP baseline
    if not args.skip_frozen:
        print("\n" + "=" * 80)
        print("FROZEN CLIP")
        print("=" * 80)

        clear_memory()

        model, processor = load_model(
            adapter_path=None,
            device=device,
        )

        start_time = time.time()

        image_embeddings = encode_images(
            model=model,
            processor=processor,
            image_ids=image_ids,
            device=device,
            batch_size=args.image_batch_size,
        )

        text_embeddings = encode_texts(
            model=model,
            processor=processor,
            captions_by_image=captions_by_image,
            device=device,
            batch_size=args.text_batch_size,
        )

        margins = compute_hard_negative_margins(
            image_embeddings=image_embeddings,
            text_embeddings=text_embeddings,
            num_images=len(image_ids),
        )

        elapsed = time.time() - start_time
        retrieval_r1 = retrieval_results.get(("frozen", None))

        row = {
            "name": "frozen",
            "budget": budget,
            "type": "frozen",
            "allocation": "",
            "seed": "",
            "num_images": len(image_ids),
            "i2t_margin": margins["i2t_margin"],
            "t2i_margin": margins["t2i_margin"],
            "mean_margin": margins["mean_margin"],
            "mean_r@1": (
                retrieval_r1
                if retrieval_r1 is not None
                else ""
            ),
            "elapsed_seconds": round(elapsed, 2),
        }

        results.append(row)

        print(f"\nImage -> Text margin: {margins['i2t_margin']:.6f}")
        print(f"Text -> Image margin: {margins['t2i_margin']:.6f}")
        print(f"Mean margin: {margins['mean_margin']:.6f}")

        del model, processor, image_embeddings, text_embeddings
        clear_memory()

    # LoRA adapters
    if not args.skip_lora:
        for allocation in allocations:
            for seed in seeds:
                print("\n" + "=" * 80)
                print(f"Seed {seed} | Allocation {allocation}")
                print("=" * 80)

                selected_adapter_path = adapter_paths[
                    (allocation, seed)
                ]

                print(f"Adapter:\n{selected_adapter_path}")

                clear_memory()

                model, processor = load_model(
                    adapter_path=selected_adapter_path,
                    device=device,
                )

                start_time = time.time()

                image_embeddings = encode_images(
                    model=model,
                    processor=processor,
                    image_ids=image_ids,
                    device=device,
                    batch_size=args.image_batch_size,
                )

                text_embeddings = encode_texts(
                    model=model,
                    processor=processor,
                    captions_by_image=captions_by_image,
                    device=device,
                    batch_size=args.text_batch_size,
                )

                margins = compute_hard_negative_margins(
                    image_embeddings=image_embeddings,
                    text_embeddings=text_embeddings,
                    num_images=len(image_ids),
                )

                elapsed = time.time() - start_time

                retrieval_r1 = retrieval_results.get(
                    (allocation, seed)
                )

                row = {
                    "name": (
                        f"lora_{allocation}_seed{seed}_budget{budget}"
                    ),
                    "budget": budget,
                    "type": "lora",
                    "allocation": allocation,
                    "seed": seed,
                    "num_images": len(image_ids),
                    "i2t_margin": margins["i2t_margin"],
                    "t2i_margin": margins["t2i_margin"],
                    "mean_margin": margins["mean_margin"],
                    "mean_r@1": (
                        retrieval_r1
                        if retrieval_r1 is not None
                        else ""
                    ),
                    "elapsed_seconds": round(elapsed, 2),
                }

                results.append(row)

                print(
                    "\nImage -> Text margin: "
                    f"{margins['i2t_margin']:.6f}"
                )
                print(
                    "Text -> Image margin: "
                    f"{margins['t2i_margin']:.6f}"
                )
                print(
                    f"Mean margin: {margins['mean_margin']:.6f}"
                )

                del model, processor, image_embeddings, text_embeddings
                clear_memory()

    aggregate = aggregate_results(results)

    # Exploratory retrieval association
    paired_margin_values = []
    paired_r1_values = []

    for row in results:
        value = row.get("mean_r@1")

        if value in ("", None):
            continue

        paired_margin_values.append(float(row["mean_margin"]))
        paired_r1_values.append(float(value))

    correlation = pearson_correlation(
        paired_margin_values,
        paired_r1_values,
    )

    lora_margin_values = []
    lora_r1_values = []

    for row in results:
        if row["type"] != "lora":
            continue

        value = row.get("mean_r@1")

        if value in ("", None):
            continue

        lora_margin_values.append(float(row["mean_margin"]))
        lora_r1_values.append(float(value))

    lora_correlation = pearson_correlation(
        lora_margin_values,
        lora_r1_values,
    )

    # Save outputs
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_csv = RESULTS_DIR / "coco_hard_negative_results.csv"
    aggregate_csv = RESULTS_DIR / "coco_hard_negative_aggregate.csv"
    json_path = RESULTS_DIR / "coco_hard_negative_results.json"

    save_csv(raw_csv, results)
    save_aggregate_csv(aggregate_csv, aggregate)

    payload = {
        "model": MODEL_NAME,
        "budget": budget,
        "wds_dataset": WDS_DATASET,
        "metadata_path": str(METADATA_PATH),
        "num_images": len(image_ids),
        "captions_per_image": CAPTIONS_PER_IMAGE,
        "results": results,
        "aggregate": aggregate,
        "exploratory_association": {
            "all_runs_pearson_margin_vs_mean_r1": correlation,
            "lora_runs_pearson_margin_vs_mean_r1": lora_correlation,
            "warning": (
                "These correlations are descriptive only. "
                "The 15 LoRA runs are not independent statistical "
                "samples because allocations and seeds are structured "
                "experimental conditions."
            ),
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 90)
    print("HARD-NEGATIVE SUMMARY")
    print("=" * 90)
    print("Allocation        I→T Margin      T→I Margin     Mean Margin")
    print("-" * 60)

    for row in aggregate:
        print(
            f"{row['allocation']:>10}        "
            f"{row['mean_i2t_margin']:.6f}        "
            f"{row['mean_t2i_margin']:.6f}        "
            f"{row['mean_margin']:.6f}"
        )

    frozen_rows = [
        row for row in results if row["type"] == "frozen"
    ]

    if frozen_rows:
        print(
            "\nFrozen mean margin: "
            f"{float(frozen_rows[0]['mean_margin']):.6f}"
        )

    mixed = [
        row["mean_margin"]
        for row in aggregate
        if row["allocation"] in {"75_25", "50_50", "25_75"}
    ]

    extremes = [
        row["mean_margin"]
        for row in aggregate
        if row["allocation"] in {"100_0", "0_100"}
    ]

    if mixed and extremes:
        mixed_mean = statistics.mean(mixed)
        extreme_mean = statistics.mean(extremes)

        print(f"\nMixed-allocation average margin: {mixed_mean:.6f}")
        print(f"Extreme-allocation average margin: {extreme_mean:.6f}")
        print(
            "Mixed-over-extreme margin advantage: "
            f"{mixed_mean - extreme_mean:+.6f}"
        )

    if correlation is not None:
        print(
            "\nExploratory Pearson correlation "
            "between hard-negative mean margin and COCO Mean R@1:"
        )
        print(f"  All evaluated runs: {correlation:.4f}")

    if lora_correlation is not None:
        print(f"  LoRA runs only: {lora_correlation:.4f}")

    print("\nSaved:")
    print(f"  {raw_csv}")
    print(f"  {aggregate_csv}")
    print(f"  {json_path}")


if __name__ == "__main__":
    main()