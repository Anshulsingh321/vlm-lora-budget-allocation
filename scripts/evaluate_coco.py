# scripts/evaluate_coco.py
from __future__ import annotations
import argparse
import csv
import gc
import json
import statistics
import tarfile
import time
from pathlib import Path
import torch
from datasets import load_dataset
from peft import PeftModel
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# ============================================================================
# CONFIGURATION

# ============================================================================
MODEL_NAME = "openai/clip-vit-base-patch32"
WDS_DATASET = "undefined443/coco-karpathy-wds"
METADATA_PATH = (
    Path("data")
    / "raw"
    / "coco"
    / "karpathy"
    / "coco_test_metadata.json"
)
RESULTS_DIR = (
    Path("results")
    / "analysis"
    / "coco"
)
EXPERIMENTS_DIR = (
    Path("results")
    / "experiments"
)
ALLOCATIONS = [
    "100_0",
    "75_25",
    "50_50",
    "25_75",
    "0_100",
]
SEEDS = [42, 123, 456]
SUPPORTED_BUDGETS = (294_912, 589_824)
EXPECTED_IMAGES = 5000
CAPTIONS_PER_IMAGE = 5
DEFAULT_IMAGE_BATCH_SIZE = 32
DEFAULT_TEXT_BATCH_SIZE = 256
RECALL_KS = (1, 5, 10)
DEFAULT_DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# ============================================================================
# ARGUMENTS

# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen CLIP and Flickr30k-trained LoRA "
            "adapters on the verified COCO Karpathy 5K benchmark."
        )
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "mps", "cuda", "cpu"),
        help="Evaluation device. Default: auto-detected device.",
    )
    parser.add_argument(
        "--image-batch-size",
        type=int,
        default=DEFAULT_IMAGE_BATCH_SIZE,
        help=f"Image encoding batch size. Default: {DEFAULT_IMAGE_BATCH_SIZE}.",
    )
    parser.add_argument(
        "--text-batch-size",
        type=int,
        default=DEFAULT_TEXT_BATCH_SIZE,
        help=f"Text encoding batch size. Default: {DEFAULT_TEXT_BATCH_SIZE}.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        choices=SUPPORTED_BUDGETS,
        default=294_912,
        help="Trainable LoRA parameter budget (default: 294912).",
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
        help="Evaluate only one seed.",
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
            "Debug-only limit. Must be omitted for final results. "
            "Uses the first N verified COCO images."
        ),
    )
    return parser.parse_args()

# ============================================================================
# DEVICE / MEMORY

# ============================================================================

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

# ============================================================================
# VERIFIED COCO METADATA

# ============================================================================

def load_metadata(
    limit_images: int | None = None,
) -> tuple[list[int], list[list[str]]]:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            "COCO metadata file was not found:\n"
            f"{METADATA_PATH}\n\n"
            "Run:\n"
            "python -m scripts.verify_coco\n"
            "first."
        )
    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)
    if not isinstance(metadata, dict):
        raise RuntimeError(
            "COCO metadata must be a JSON object."
        )
    if len(metadata) != EXPECTED_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_IMAGES:,} COCO images, "
            f"found {len(metadata):,}."
        )
    items = sorted(
        metadata.values(),
        key=lambda item: int(item["cocoid"]),
    )
    if limit_images is not None:
        if (
            limit_images <= 0
            or limit_images > EXPECTED_IMAGES
        ):
            raise ValueError(
                f"--limit-images must be between 1 and "
                f"{EXPECTED_IMAGES}."
            )
        items = items[:limit_images]
    image_ids: list[int] = []
    captions_by_image: list[list[str]] = []
    for index, item in enumerate(items):
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
                f"Metadata entry {index} is missing fields: "
                f"{sorted(missing)}"
            )
        coco_id = int(item["cocoid"])
        captions = item["captions"]
        if len(captions) != CAPTIONS_PER_IMAGE:
            raise RuntimeError(
                f"COCO ID {coco_id} has "
                f"{len(captions)} captions; "
                f"expected {CAPTIONS_PER_IMAGE}."
            )
        if any(
            not isinstance(caption, str)
            or not caption.strip()
            for caption in captions
        ):
            raise RuntimeError(
                f"COCO ID {coco_id} contains an invalid caption."
            )
        image_ids.append(coco_id)
        captions_by_image.append(captions)
    expected_images = (
        EXPECTED_IMAGES
        if limit_images is None
        else limit_images
    )
    if len(image_ids) != expected_images:
        raise RuntimeError(
            f"Expected {expected_images} selected images, "
            f"found {len(image_ids)}."
        )
    return image_ids, captions_by_image

# ============================================================================
# MODEL LOADING

# ============================================================================

def find_adapter_path(
    allocation: str,
    seed: int,
    budget: int = 294_912,
) -> Path:
    """Locate the best saved adapter for one budget/allocation/seed."""
    if budget not in SUPPORTED_BUDGETS:
        raise ValueError(
            f"Unsupported budget {budget}. Supported budgets: {SUPPORTED_BUDGETS}."
        )
    if budget == 294_912:
        candidates = [
            EXPERIMENTS_DIR / f"flickr30k_lora_{allocation}_seed{seed}",
            # Legacy directory used by older 294,912-budget runs.
            EXPERIMENTS_DIR / f"flickr30k_lora_{allocation}",
        ]
    else:
        candidates = [
            EXPERIMENTS_DIR
            / f"flickr30k_lora_{allocation}_budget{budget}_seed{seed}"
        ]
    experiment_dir = next(
        (candidate for candidate in candidates if candidate.exists()),
        None,
    )
    if experiment_dir is None:
        checked = "\n".join(f"  {candidate}" for candidate in candidates)
        raise FileNotFoundError(
            "No experiment directory found for:\n"
            f"allocation={allocation}, seed={seed}, budget={budget:,}\n"
            f"Checked:\n{checked}"
        )
    configs = list(experiment_dir.rglob("adapter_config.json"))
    if not configs:
        raise FileNotFoundError(
            "No adapter_config.json found under:\n"
            f"{experiment_dir}"
        )
    best_configs = [
        path
        for path in configs
        if any("best" in part.lower() for part in path.parts)
    ]
    if len(best_configs) == 1:
        return best_configs[0].parent
    if len(configs) == 1:
        return configs[0].parent
    config_list = "\n".join(f"  - {path.parent}" for path in configs)
    raise RuntimeError(
        "Multiple LoRA adapters were found and no unique 'best' adapter "
        "could be identified.\n"
        f"Experiment directory:\n{experiment_dir}\n"
        f"Found:\n{config_list}"
    )

def count_lora_parameters(model: torch.nn.Module) -> int:
    """Count adapter A/B weights, irrespective of requires_grad flags."""
    return sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if "lora_A" in name or "lora_B" in name
    )


def get_clip_model(
    model: torch.nn.Module,
) -> torch.nn.Module:
    """
    Return the underlying CLIPModel.
    For PeftModel, get_base_model() returns the CLIP model
    with LoRA modules still active.
    """
    if isinstance(model, PeftModel):
        return model.get_base_model()
    return model

def load_model(
    adapter_path: Path | None,
    device: torch.device,
) -> tuple[torch.nn.Module, CLIPProcessor]:
    processor = CLIPProcessor.from_pretrained(
        MODEL_NAME
    )
    base_model = CLIPModel.from_pretrained(
        MODEL_NAME
    )
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

# ============================================================================
# EMBEDDING FUNCTIONS

# ============================================================================
@torch.inference_mode()
def encode_images(
    model: torch.nn.Module,
    processor: CLIPProcessor,
    image_ids: list[int],
    device: torch.device,
    batch_size: int,
    max_stream_retries: int = 5,
) -> torch.Tensor:
    """Stream and encode COCO images, retrying interrupted WDS streams.

    On transient failures, previously encoded embeddings are kept. The stream
    is restarted and already-seen COCO image IDs are skipped.
    """
    num_images = len(image_ids)
    clip_model = get_clip_model(model)
    image_embeddings = None
    id_to_index = {coco_id: index for index, coco_id in enumerate(image_ids)}
    seen_ids: set[int] = set()
    batch_images: list[Image.Image] = []
    batch_ids: list[int] = []
    retry_count = 0

    def flush_batch() -> None:
        nonlocal image_embeddings, batch_images, batch_ids
        if not batch_images:
            return

        inputs = processor(images=batch_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        vision_outputs = clip_model.vision_model(pixel_values=pixel_values)
        embeds = clip_model.visual_projection(vision_outputs.pooler_output)
        embeds = embeds / embeds.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        embeds = embeds.detach().float().cpu()

        if image_embeddings is None:
            image_embeddings = torch.empty(
                (num_images, embeds.shape[1]), dtype=torch.float32
            )

        target_indices = [id_to_index[coco_id] for coco_id in batch_ids]
        image_embeddings[target_indices] = embeds
        batch_images = []
        batch_ids = []

    def is_retryable_stream_error(exc: Exception) -> bool:
        if isinstance(exc, (tarfile.TarError, ConnectionError, TimeoutError, OSError, EOFError)):
            return True
        message = str(exc).lower()
        transient_markers = (
            "disconnected from remote data host",
            "unexpected end of data",
            "incomplete read",
            "connection reset",
            "connection aborted",
            "timed out",
            "timeout",
            "temporarily unavailable",
        )
        return any(marker in message for marker in transient_markers)

    while len(seen_ids) < num_images:
        # A fresh iterable is required after a stream/tar reader has failed.
        dataset = load_dataset(WDS_DATASET, split="test", streaming=True)
        try:
            for sample in dataset:
                if "jpg" not in sample:
                    raise RuntimeError("WDS sample is missing 'jpg'.")
                if "json" not in sample:
                    raise RuntimeError("WDS sample is missing 'json'.")

                metadata = sample["json"]
                if "cocoid" not in metadata:
                    raise RuntimeError("WDS sample metadata is missing 'cocoid'.")

                coco_id = int(metadata["cocoid"])
                if coco_id not in id_to_index:
                    continue

                if coco_id in seen_ids:
                    # A retry restarts from the beginning; skip known IDs.
                    if retry_count > 0:
                        continue
                    raise RuntimeError(f"Duplicate WDS image encountered: {coco_id}")

                image = sample["jpg"]
                if not isinstance(image, Image.Image):
                    raise RuntimeError(
                        f"WDS COCO ID {coco_id} did not return a PIL image."
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

            flush_batch()
            if len(seen_ids) != num_images:
                raise tarfile.ReadError(
                    "WDS stream ended before all requested images were read; "
                    f"{num_images - len(seen_ids)} image ID(s) are missing."
                )

        except Exception as exc:
            if not is_retryable_stream_error(exc):
                raise

            # Flush successfully received samples before reopening the stream.
            flush_batch()
            if retry_count >= max_stream_retries:
                raise RuntimeError(
                    f"COCO WDS stream failed after {max_stream_retries + 1} "
                    f"attempts; collected {len(seen_ids):,}/{num_images:,} "
                    f"images. Last error: {exc}"
                ) from exc

            retry_count += 1
            wait_seconds = min(5 * retry_count, 30)
            print(
                f"\nCOCO stream interrupted ({exc}). Retry "
                f"{retry_count}/{max_stream_retries} in {wait_seconds}s; "
                f"retaining {len(seen_ids):,}/{num_images:,} images.",
                flush=True,
            )
            time.sleep(wait_seconds)
            continue

        break

    if image_embeddings is None:
        raise RuntimeError("No image embeddings were generated.")
    if len(seen_ids) != num_images:
        raise RuntimeError(
            f"Expected {num_images:,} images, found {len(seen_ids):,}."
        )
    return image_embeddings

@torch.inference_mode()

def encode_texts(
    model: torch.nn.Module,
    processor: CLIPProcessor,
    captions: list[str],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """
    Encode captions in exact canonical order:
        image 0: caption 0..4
        image 1: caption 0..4
        ...
    """
    clip_model = get_clip_model(model)
    batches = []
    for start in range(
        0,
        len(captions),
        batch_size,
    ):
        batch = captions[
            start:
            start + batch_size
        ]
        inputs = processor(
            text=batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = inputs[
            "input_ids"
        ].to(device)
        attention_mask = inputs.get(
            "attention_mask"
        )
        if attention_mask is not None:
            attention_mask = (
                attention_mask.to(device)
            )
        text_outputs = clip_model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        embeds = clip_model.text_projection(
            text_outputs.pooler_output
        )
        embeds = embeds / embeds.norm(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-12)
        batches.append(
            embeds.detach().float().cpu()
        )
    if not batches:
        raise RuntimeError(
            "No text embeddings were generated."
        )
    return torch.cat(
        batches,
        dim=0,
    )

# ============================================================================
# RETRIEVAL METRICS

# ============================================================================

def compute_image_to_text_recall(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    num_images: int,
) -> dict[str, float]:
    expected_texts = (
        num_images
        * CAPTIONS_PER_IMAGE
    )
    if text_embeddings.shape[0] != expected_texts:
        raise RuntimeError(
            "Unexpected text embedding count "
            "for I->T retrieval."
        )
    ks = max(RECALL_KS)
    hits = {
        k: 0
        for k in RECALL_KS
    }
    text_embeddings = text_embeddings.contiguous()
    chunk_size = 32
    for start in range(
        0,
        num_images,
        chunk_size,
    ):
        end = min(
            start + chunk_size,
            num_images,
        )
        similarities = (
            image_embeddings[start:end]
            @ text_embeddings.T
        )
        top_indices = similarities.topk(
            ks,
            dim=1,
            largest=True,
            sorted=True,
        ).indices
        for local_index in range(
            end - start
        ):
            image_index = (
                start + local_index
            )
            ground_truth = set(
                range(
                    image_index * CAPTIONS_PER_IMAGE,
                    (
                        image_index + 1
                    ) * CAPTIONS_PER_IMAGE,
                )
            )
            retrieved = top_indices[
                local_index
            ].tolist()
            for k in RECALL_KS:
                if any(
                    index in ground_truth
                    for index in retrieved[:k]
                ):
                    hits[k] += 1
    return {
        f"i2t_r@{k}": (
            hits[k] / num_images
        )
        for k in RECALL_KS
    }

def compute_text_to_image_recall(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    num_images: int,
) -> dict[str, float]:
    expected_texts = (
        num_images
        * CAPTIONS_PER_IMAGE
    )
    if text_embeddings.shape[0] != expected_texts:
        raise RuntimeError(
            "Unexpected text embedding count "
            "for T->I retrieval."
        )
    ks = max(RECALL_KS)
    hits = {
        k: 0
        for k in RECALL_KS
    }
    image_embeddings = image_embeddings.contiguous()
    chunk_size = 256
    for start in range(
        0,
        expected_texts,
        chunk_size,
    ):
        end = min(
            start + chunk_size,
            expected_texts,
        )
        similarities = (
            text_embeddings[start:end]
            @ image_embeddings.T
        )
        top_indices = similarities.topk(
            ks,
            dim=1,
            largest=True,
            sorted=True,
        ).indices
        for local_index in range(
            end - start
        ):
            caption_index = (
                start + local_index
            )
            ground_truth_image = (
                caption_index
                // CAPTIONS_PER_IMAGE
            )
            retrieved = top_indices[
                local_index
            ].tolist()
            for k in RECALL_KS:
                if (
                    ground_truth_image
                    in retrieved[:k]
                ):
                    hits[k] += 1
    return {
        f"t2i_r@{k}": (
            hits[k] / expected_texts
        )
        for k in RECALL_KS
    }

def calculate_metrics(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    num_images: int,
) -> dict[str, float]:
    i2t = compute_image_to_text_recall(
        image_embeddings,
        text_embeddings,
        num_images,
    )
    t2i = compute_text_to_image_recall(
        image_embeddings,
        text_embeddings,
        num_images,
    )
    metrics = {
        **i2t,
        **t2i,
    }
    for k in RECALL_KS:
        metrics[
            f"mean_r@{k}"
        ] = (
            metrics[f"i2t_r@{k}"]
            + metrics[f"t2i_r@{k}"]
        ) / 2.0
    return metrics

# ============================================================================
# EVALUATION

# ============================================================================

def flatten_captions(
    captions_by_image: list[list[str]],
) -> list[str]:
    flat = []
    for captions in captions_by_image:
        if len(captions) != CAPTIONS_PER_IMAGE:
            raise RuntimeError(
                "Expected exactly five captions per image."
            )
        flat.extend(captions)
    return flat

def evaluate_model(
    name: str,
    model: torch.nn.Module,
    processor: CLIPProcessor,
    image_ids: list[int],
    captions_by_image: list[list[str]],
    device: torch.device,
    image_batch_size: int,
    text_batch_size: int,
) -> dict:
    start_time = time.time()
    image_embeddings = encode_images(
        model=model,
        processor=processor,
        image_ids=image_ids,
        device=device,
        batch_size=image_batch_size,
    )
    captions = flatten_captions(
        captions_by_image
    )
    text_embeddings = encode_texts(
        model=model,
        processor=processor,
        captions=captions,
        device=device,
        batch_size=text_batch_size,
    )
    metrics = calculate_metrics(
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        num_images=len(image_ids),
    )
    elapsed = time.time() - start_time
    return {
        "name": name,
        "num_images": len(image_ids),
        "num_captions": len(captions),
        "device": str(device),
        "metrics": metrics,
        "elapsed_seconds": round(
            elapsed,
            2,
        ),
    }

# ============================================================================
# RESULT SAVING

# ============================================================================

def save_json(
    path: Path,
    payload,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
        )

def save_summary_csv(
    results: list[dict],
) -> Path:
    path = (
        RESULTS_DIR
        / "coco_summary.csv"
    )
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    fields = [
        "name",
        "allocation",
        "seed",
        "type",
        "i2t_r@1",
        "i2t_r@5",
        "i2t_r@10",
        "t2i_r@1",
        "t2i_r@5",
        "t2i_r@10",
        "mean_r@1",
        "mean_r@5",
        "mean_r@10",
        "num_images",
        "num_captions",
        "elapsed_seconds",
    ]
    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        for result in results:
            metrics = result["metrics"]
            writer.writerow(
                {
                    "name": result["name"],
                    "allocation": result.get(
                        "allocation",
                        "",
                    ),
                    "seed": result.get(
                        "seed",
                        "",
                    ),
                    "type": result.get(
                        "type",
                        "",
                    ),
                    "i2t_r@1": metrics[
                        "i2t_r@1"
                    ],
                    "i2t_r@5": metrics[
                        "i2t_r@5"
                    ],
                    "i2t_r@10": metrics[
                        "i2t_r@10"
                    ],
                    "t2i_r@1": metrics[
                        "t2i_r@1"
                    ],
                    "t2i_r@5": metrics[
                        "t2i_r@5"
                    ],
                    "t2i_r@10": metrics[
                        "t2i_r@10"
                    ],
                    "mean_r@1": metrics[
                        "mean_r@1"
                    ],
                    "mean_r@5": metrics[
                        "mean_r@5"
                    ],
                    "mean_r@10": metrics[
                        "mean_r@10"
                    ],
                    "num_images": result[
                        "num_images"
                    ],
                    "num_captions": result[
                        "num_captions"
                    ],
                    "elapsed_seconds": result[
                        "elapsed_seconds"
                    ],
                }
            )
    return path

def save_aggregate_csv(
    aggregate: list[dict],
) -> Path:
    path = (
        RESULTS_DIR
        / "coco_allocation_aggregate.csv"
    )
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    fields = [
        "allocation",
        "num_seeds",
        "mean_i2t_r@1",
        "std_i2t_r@1",
        "mean_i2t_r@5",
        "std_i2t_r@5",
        "mean_i2t_r@10",
        "std_i2t_r@10",
        "mean_t2i_r@1",
        "std_t2i_r@1",
        "mean_t2i_r@5",
        "std_t2i_r@5",
        "mean_t2i_r@10",
        "std_t2i_r@10",
        "mean_r@1",
        "std_r@1",
        "mean_r@5",
        "std_r@5",
        "mean_r@10",
        "std_r@10",
    ]
    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        for row in aggregate:
            metrics = row["metrics"]
            writer.writerow(
                {
                    "allocation": row[
                        "allocation"
                    ],
                    "num_seeds": row[
                        "num_seeds"
                    ],
                    "mean_i2t_r@1": metrics[
                        "i2t_r@1"
                    ]["mean"],
                    "std_i2t_r@1": metrics[
                        "i2t_r@1"
                    ]["std"],
                    "mean_i2t_r@5": metrics[
                        "i2t_r@5"
                    ]["mean"],
                    "std_i2t_r@5": metrics[
                        "i2t_r@5"
                    ]["std"],
                    "mean_i2t_r@10": metrics[
                        "i2t_r@10"
                    ]["mean"],
                    "std_i2t_r@10": metrics[
                        "i2t_r@10"
                    ]["std"],
                    "mean_t2i_r@1": metrics[
                        "t2i_r@1"
                    ]["mean"],
                    "std_t2i_r@1": metrics[
                        "t2i_r@1"
                    ]["std"],
                    "mean_t2i_r@5": metrics[
                        "t2i_r@5"
                    ]["mean"],
                    "std_t2i_r@5": metrics[
                        "t2i_r@5"
                    ]["std"],
                    "mean_t2i_r@10": metrics[
                        "t2i_r@10"
                    ]["mean"],
                    "std_t2i_r@10": metrics[
                        "t2i_r@10"
                    ]["std"],
                    "mean_r@1": metrics[
                        "mean_r@1"
                    ]["mean"],
                    "std_r@1": metrics[
                        "mean_r@1"
                    ]["std"],
                    "mean_r@5": metrics[
                        "mean_r@5"
                    ]["mean"],
                    "std_r@5": metrics[
                        "mean_r@5"
                    ]["std"],
                    "mean_r@10": metrics[
                        "mean_r@10"
                    ]["mean"],
                    "std_r@10": metrics[
                        "mean_r@10"
                    ]["std"],
                }
            )
    return path

def aggregate_lora_results(
    results: list[dict],
) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for result in results:
        if result.get("type") != "lora":
            continue
        grouped.setdefault(
            result["allocation"],
            [],
        ).append(result)
    aggregate = []
    for allocation in ALLOCATIONS:
        rows = grouped.get(
            allocation,
            [],
        )
        if not rows:
            continue
        metrics = {}
        for metric in (
            "i2t_r@1",
            "i2t_r@5",
            "i2t_r@10",
            "t2i_r@1",
            "t2i_r@5",
            "t2i_r@10",
            "mean_r@1",
            "mean_r@5",
            "mean_r@10",
        ):
            values = [
                row["metrics"][metric]
                for row in rows
            ]
            metrics[metric] = {
                "mean": statistics.mean(
                    values
                ),
                "std": (
                    statistics.stdev(values)
                    if len(values) > 1
                    else 0.0
                ),
                "values": values,
            }
        aggregate.append(
            {
                "allocation": allocation,
                "seeds": [
                    row["seed"]
                    for row in rows
                ],
                "num_seeds": len(rows),
                "metrics": metrics,
            }
        )
    return aggregate

# ============================================================================
# OUTPUT

# ============================================================================

def print_results(
    results: list[dict],
    aggregate: list[dict],
) -> None:
    print(
        "\n"
        + "=" * 80
    )
    print(
        "COCO CROSS-DATASET EVALUATION RESULTS"
    )
    print(
        "=" * 80
    )
    for result in results:
        metrics = result["metrics"]
        print(
            f"\n{result['name']}"
        )
        print(
            f"  I→T  "
            f"R@1={metrics['i2t_r@1'] * 100:.2f}%  "
            f"R@5={metrics['i2t_r@5'] * 100:.2f}%  "
            f"R@10={metrics['i2t_r@10'] * 100:.2f}%"
        )
        print(
            f"  T→I  "
            f"R@1={metrics['t2i_r@1'] * 100:.2f}%  "
            f"R@5={metrics['t2i_r@5'] * 100:.2f}%  "
            f"R@10={metrics['t2i_r@10'] * 100:.2f}%"
        )
        print(
            f"  Mean "
            f"R@1={metrics['mean_r@1'] * 100:.2f}%  "
            f"R@5={metrics['mean_r@5'] * 100:.2f}%  "
            f"R@10={metrics['mean_r@10'] * 100:.2f}%"
        )
    if aggregate:
        print(
            "\n"
            + "-" * 80
        )
        print(
            "THREE-SEED LORA AGGREGATES"
        )
        print(
            "-" * 80
        )
        print(
            "\nAllocation      Mean R@1       Std"
        )
        print(
            "--------------------------------------"
        )
        for row in aggregate:
            metric = row[
                "metrics"
            ]["mean_r@1"]
            print(
                f"{row['allocation']:>10}    "
                f"{metric['mean'] * 100:8.2f}%    "
                f"{metric['std'] * 100:7.2f}%"
            )

# ============================================================================
# MAIN

# ============================================================================

def main() -> None:
    args = parse_args()

    # Preserve the previous output location for 294,912-budget results.
    # Keep 589,824-budget artifacts isolated from all earlier results.
    global RESULTS_DIR
    if args.budget != 294_912:
        RESULTS_DIR = RESULTS_DIR / f"budget_{args.budget}"
    if args.limit_images is not None:
        # Keep debug subsets completely separate from final benchmark results.
        RESULTS_DIR = RESULTS_DIR / f"debug_{args.limit_images}_images"
    if (
        args.image_batch_size <= 0
        or args.text_batch_size <= 0
    ):
        raise ValueError(
            "Batch sizes must be positive."
        )
    device = resolve_device(
        args.device
    )
    image_ids, captions_by_image = load_metadata(
        limit_images=args.limit_images
    )
    if args.limit_images is not None:
        print(
            "\nWARNING: --limit-images is enabled."
        )
        print(
            "These results are DEBUG ONLY and "
            "must not be used for the paper."
        )
    print(
        "\n"
        + "=" * 80
    )
    print(
        "COCO CROSS-DATASET EVALUATION"
    )
    print(
        "=" * 80
    )
    print(
        f"Model: {MODEL_NAME}"
    )
    print(
        f"Device: {device}"
    )
    print(
        f"Trainable LoRA budget: {args.budget:,}"
    )
    print(
        f"Images: {len(image_ids):,}"
    )
    print(
        f"Captions: "
        f"{len(image_ids) * CAPTIONS_PER_IMAGE:,}"
    )
    print(
        f"Image batch size: "
        f"{args.image_batch_size}"
    )
    print(
        f"Text batch size: "
        f"{args.text_batch_size}"
    )
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
    # ========================================================================
    # FROZEN CLIP
    # ========================================================================
    if not args.skip_frozen:
        print(
            "\n"
            + "=" * 80
        )
        print(
            "Evaluating frozen CLIP"
        )
        print(
            "=" * 80
        )
        clear_memory()
        model, processor = load_model(
            adapter_path=None,
            device=device,
        )
        result = evaluate_model(
            name="frozen",
            model=model,
            processor=processor,
            image_ids=image_ids,
            captions_by_image=captions_by_image,
            device=device,
            image_batch_size=args.image_batch_size,
            text_batch_size=args.text_batch_size,
        )
        result["type"] = "frozen"
        results.append(result)
        save_json(
            RESULTS_DIR
            / "raw"
            / "frozen.json",
            result,
        )
        print(
            f"\nFrozen Mean R@1: "
            f"{result['metrics']['mean_r@1'] * 100:.2f}%"
        )
        del model
        del processor
        clear_memory()
    # ========================================================================
    # LORA ADAPTERS
    # ========================================================================
    if not args.skip_lora:
        for allocation in allocations:
            for seed in seeds:
                print(
                    "\n"
                    + "=" * 80
                )
                print(
                    "Evaluating LoRA"
                )
                print(
                    f"Allocation: {allocation}"
                )
                print(
                    f"Seed: {seed}"
                )
                print(
                    "=" * 80
                )
                adapter_path = find_adapter_path(
                    allocation=allocation,
                    seed=seed,
                    budget=args.budget,
                )
                print(
                    "Adapter:"
                )
                print(
                    f"  {adapter_path}"
                )
                clear_memory()
                model, processor = load_model(
                    adapter_path=adapter_path,
                    device=device,
                )
                adapter_params = count_lora_parameters(model)
                print(f"LoRA adapter parameters loaded: {adapter_params:,}")
                if adapter_params != args.budget:
                    raise RuntimeError(
                        f"Loaded adapter has {adapter_params:,} LoRA parameters, "
                        f"but budget {args.budget:,} was requested."
                    )
                result = evaluate_model(
                    name=(
                        f"lora_{allocation}_"
                        f"seed{seed}"
                    ),
                    model=model,
                    processor=processor,
                    image_ids=image_ids,
                    captions_by_image=captions_by_image,
                    device=device,
                    image_batch_size=args.image_batch_size,
                    text_batch_size=args.text_batch_size,
                )
                result["type"] = "lora"
                result["budget"] = args.budget
                result["allocation"] = allocation
                result["seed"] = seed
                result["adapter_path"] = str(
                    adapter_path
                )
                results.append(result)
                save_json(
                    RESULTS_DIR
                    / "raw"
                    / (
                        f"lora_{allocation}_"
                        f"seed{seed}.json"
                    ),
                    result,
                )
                metrics = result["metrics"]
                print(
                    f"\nMean R@1: "
                    f"{metrics['mean_r@1'] * 100:.2f}%"
                )
                del model
                del processor
                clear_memory()
    # ========================================================================
    # AGGREGATION
    # ========================================================================
    aggregate = aggregate_lora_results(
        results
    )
    payload = {
        "budget": args.budget,
        "model": MODEL_NAME,
        "wds_dataset": WDS_DATASET,
        "metadata_path": str(
            METADATA_PATH
        ),
        "num_images": len(
            image_ids
        ),
        "captions_per_image": (
            CAPTIONS_PER_IMAGE
        ),
        "num_captions": (
            len(image_ids)
            * CAPTIONS_PER_IMAGE
        ),
        "device": str(device),
        "results": results,
        "lora_aggregate": aggregate,
    }
    save_json(
        RESULTS_DIR
        / "coco_results.json",
        payload,
    )
    csv_path = save_summary_csv(
        results
    )
    aggregate_csv_path = save_aggregate_csv(
        aggregate
    )
    print_results(
        results,
        aggregate,
    )
    print(
        "\n"
        + "=" * 80
    )
    print(
        "COCO EVALUATION COMPLETE ✅"
    )
    print(
        "=" * 80
    )
    print(
        "\nSaved:"
    )
    print(
        f"  {RESULTS_DIR / 'coco_results.json'}"
    )
    print(
        f"  {csv_path}"
    )
    print(
        f"  {aggregate_csv_path}"
    )

if __name__ == "__main__":
    main()
