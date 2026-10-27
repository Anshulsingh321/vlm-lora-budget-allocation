# scripts/train.py
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from peft import LoraConfig, get_peft_model
from transformers import CLIPModel, CLIPProcessor


# ============================================================
# EXPERIMENT DEFINITIONS
# ============================================================

BASE_BUDGET = 294_912

SUPPORTED_BUDGETS = (294_912, 589_824)

# The 589K experiment doubles the ranks of the original
# 295K experiment while preserving the five allocation ratios.

BASE_ALLOCATIONS = {
    "100_0": {
        "vision_percent": 100,
        "language_percent": 0,
        "vision_rank": 8,
        "language_rank": 0,
    },
    "75_25": {
        "vision_percent": 75,
        "language_percent": 25,
        "vision_rank": 6,
        "language_rank": 3,
    },
    "50_50": {
        "vision_percent": 50,
        "language_percent": 50,
        "vision_rank": 4,
        "language_rank": 6,
    },
    "25_75": {
        "vision_percent": 25,
        "language_percent": 75,
        "vision_rank": 2,
        "language_rank": 9,
    },
    "0_100": {
        "vision_percent": 0,
        "language_percent": 100,
        "vision_rank": 0,
        "language_rank": 12,
    },
}


def get_allocation_spec(allocation: str, budget: int) -> dict:
    """Return a validated allocation for a supported exact budget."""
    if budget not in SUPPORTED_BUDGETS:
        supported = ", ".join(
            f"{value:,}" for value in SUPPORTED_BUDGETS
        )
        raise ValueError(
            f"Unsupported budget {budget:,}. "
            f"Supported budgets: {supported}"
        )

    if allocation not in BASE_ALLOCATIONS:
        raise ValueError(f"Unknown allocation: {allocation}")

    multiplier = budget // BASE_BUDGET
    spec = dict(BASE_ALLOCATIONS[allocation])
    spec["vision_rank"] *= multiplier
    spec["language_rank"] *= multiplier
    spec["budget"] = budget

    return spec


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train CLIP with modality-specific LoRA allocation."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to training config.",
    )

    parser.add_argument(
        "--allocation",
        type=str,
        required=True,
        choices=list(BASE_ALLOCATIONS.keys()),
        help="LoRA budget allocation.",
    )

    parser.add_argument(
        "--budget",
        type=int,
        choices=SUPPORTED_BUDGETS,
        default=None,
        help=(
            "Total trainable LoRA parameter budget. "
            "Defaults to configs/train.yaml. "
            "Use 589824 for the second-budget study."
        ),
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "Optional maximum number of optimizer steps per epoch."
        ),
    )

    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help=(
            "Delete and recreate this exact experiment directory. "
            "Existing runs are protected by default."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the seed from the config.",
    )

    return parser.parse_args()


# ============================================================
# CONFIG / SEED / DEVICE
# ============================================================

def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# OUTPUT HELPERS
# ============================================================

def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ============================================================
# DATASET
# ============================================================

class Flickr30kTrainDataset(Dataset):
    """
    One image = one training sample.

    Caption selection:
        caption_index = (image_index + epoch) % 5

    This ensures that every allocation and seed uses the
    same deterministic caption schedule.
    """

    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        split: str,
    ):
        self.image_dir = Path(image_dir)

        df = pd.read_csv(csv_path)
        df = df[
            df["split"] == split
        ].reset_index(drop=True)

        self.rows = []

        for _, row in df.iterrows():
            captions = json.loads(row["raw"])

            if len(captions) != 5:
                raise ValueError(
                    f"Expected 5 captions for "
                    f"{row['filename']}, got {len(captions)}"
                )

            self.rows.append(
                {
                    "filename": row["filename"],
                    "img_id": int(row["img_id"]),
                    "captions": captions,
                }
            )

        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image_path = self.image_dir / row["filename"]

        image = Image.open(image_path).convert("RGB")

        caption_index = (index + self.epoch) % 5
        caption = row["captions"][caption_index]

        return {
            "image": image,
            "text": caption,
            "filename": row["filename"],
            "img_id": row["img_id"],
        }


class Flickr30kEvalDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        split: str,
    ):
        self.image_dir = Path(image_dir)

        df = pd.read_csv(csv_path)
        df = df[
            df["split"] == split
        ].reset_index(drop=True)

        self.rows = []

        for _, row in df.iterrows():
            captions = json.loads(row["raw"])

            if len(captions) != 5:
                raise ValueError(
                    f"Expected 5 captions for "
                    f"{row['filename']}, got {len(captions)}"
                )

            self.rows.append(
                {
                    "filename": row["filename"],
                    "img_id": int(row["img_id"]),
                    "captions": captions,
                }
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image_path = self.image_dir / row["filename"]

        image = Image.open(image_path).convert("RGB")

        return {
            "image": image,
            "captions": row["captions"],
            "filename": row["filename"],
            "img_id": row["img_id"],
        }


# ============================================================
# COLLATE
# ============================================================

def collate_train(batch, processor):
    images = [
        item["image"] for item in batch
    ]

    texts = [
        item["text"] for item in batch
    ]

    encoded = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    return encoded


# ============================================================
# LORA CONFIGURATION
# ============================================================

def build_lora_patterns(
    model,
    vision_rank: int,
    language_rank: int,
):
    """
    Build targets only for modalities receiving nonzero
    LoRA capacity. PEFT does not permit r=0.
    """

    target_modules = []
    rank_pattern = {}
    alpha_pattern = {}

    vision_layers = (
        model.config.vision_config.num_hidden_layers
    )

    language_layers = (
        model.config.text_config.num_hidden_layers
    )

    # Vision encoder
    if vision_rank > 0:
        for layer in range(vision_layers):
            for module in ("q_proj", "v_proj"):
                module_name = (
                    f"vision_model.encoder.layers."
                    f"{layer}.self_attn.{module}"
                )

                target_modules.append(module_name)
                rank_pattern[module_name] = vision_rank
                alpha_pattern[module_name] = vision_rank * 2

    # Language encoder
    if language_rank > 0:
        for layer in range(language_layers):
            for module in ("q_proj", "v_proj"):
                module_name = (
                    f"text_model.encoder.layers."
                    f"{layer}.self_attn.{module}"
                )

                target_modules.append(module_name)
                rank_pattern[module_name] = language_rank
                alpha_pattern[module_name] = language_rank * 2

    if not target_modules:
        raise ValueError("No LoRA target modules were created.")

    return target_modules, rank_pattern, alpha_pattern


def build_lora_model(
    model_name: str,
    allocation: str,
    budget: int,
):
    base_model = CLIPModel.from_pretrained(model_name)

    spec = get_allocation_spec(allocation, budget)

    (
        target_modules,
        rank_pattern,
        alpha_pattern,
    ) = build_lora_patterns(
        base_model,
        vision_rank=spec["vision_rank"],
        language_rank=spec["language_rank"],
    )

    lora_config = LoraConfig(
        # PEFT needs a positive default rank.
        # Per-module ranks are supplied via rank_pattern.
        r=1,
        lora_alpha=2,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type="FEATURE_EXTRACTION",
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
    )

    model = get_peft_model(base_model, lora_config)

    return model, spec


# ============================================================
# PARAMETER ACCOUNTING
# ============================================================

def count_parameters(model):
    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total, trainable


def count_modality_parameters(model):
    vision = 0
    language = 0

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if (
            "vision_model" in name
            and "lora_" in name
        ):
            vision += parameter.numel()

        elif (
            "text_model" in name
            and "lora_" in name
        ):
            language += parameter.numel()

    return vision, language


# ============================================================
# CLIP OUTPUT HANDLING
# ============================================================

def extract_embedding(output):
    if isinstance(output, torch.Tensor):
        return output

    if hasattr(output, "pooler_output"):
        return output.pooler_output

    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state

    raise TypeError(
        f"Unsupported embedding output: {type(output)}"
    )


def normalize_embeddings(x):
    return x / x.norm(
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-8)


# ============================================================
# CONTRASTIVE LOSS
# ============================================================

def clip_loss(
    model,
    pixel_values,
    input_ids,
    attention_mask,
):
    image_output = model.get_image_features(
        pixel_values=pixel_values
    )

    text_output = model.get_text_features(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    image_features = normalize_embeddings(
        extract_embedding(image_output)
    )

    text_features = normalize_embeddings(
        extract_embedding(text_output)
    )

    base_model = model.base_model.model

    logit_scale = (
        base_model.logit_scale
        .exp()
        .clamp(max=100)
    )

    logits_per_image = (
        logit_scale
        * image_features
        @ text_features.T
    )

    logits_per_text = logits_per_image.T

    labels = torch.arange(
        pixel_values.size(0),
        device=pixel_values.device,
    )

    loss_i2t = nn.functional.cross_entropy(
        logits_per_image,
        labels,
    )

    loss_t2i = nn.functional.cross_entropy(
        logits_per_text,
        labels,
    )

    return (loss_i2t + loss_t2i) / 2


# ============================================================
# EMBEDDING FUNCTIONS
# ============================================================

@torch.no_grad()
def encode_image_batch(
    model,
    processor,
    images,
    device,
):
    inputs = processor(
        images=images,
        return_tensors="pt",
    )

    pixel_values = inputs["pixel_values"].to(device)

    output = model.get_image_features(
        pixel_values=pixel_values
    )

    features = normalize_embeddings(
        extract_embedding(output)
    )

    return features.cpu()


@torch.no_grad()
def encode_text_batch(
    model,
    processor,
    texts,
    device,
):
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

    features = normalize_embeddings(
        extract_embedding(output)
    )

    return features.cpu()


# ============================================================
# RETRIEVAL EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_retrieval(
    model,
    processor,
    dataset,
    device,
    batch_size,
):
    model.eval()

    # Encode images
    image_features = []

    for start in tqdm(
        range(0, len(dataset), batch_size),
        desc="Encoding validation images",
    ):
        end = min(start + batch_size, len(dataset))

        batch = [
            dataset[i] for i in range(start, end)
        ]

        images = [
            item["image"] for item in batch
        ]

        features = encode_image_batch(
            model,
            processor,
            images,
            device,
        )

        image_features.append(features)

    image_features = torch.cat(image_features, dim=0)

    # Encode all five captions per image
    all_captions = []
    caption_to_image = []

    for image_index, row in enumerate(dataset.rows):
        for caption in row["captions"]:
            all_captions.append(caption)
            caption_to_image.append(image_index)

    caption_features = []

    for start in tqdm(
        range(0, len(all_captions), batch_size),
        desc="Encoding validation captions",
    ):
        end = min(start + batch_size, len(all_captions))
        texts = all_captions[start:end]

        features = encode_text_batch(
            model,
            processor,
            texts,
            device,
        )

        caption_features.append(features)

    caption_features = torch.cat(
        caption_features,
        dim=0,
    )

    caption_to_image = torch.tensor(
        caption_to_image,
        dtype=torch.long,
    )

    # Similarity matrix: images x captions
    similarity = image_features @ caption_features.T

    # Image -> Text retrieval
    image_to_text = {}

    for k in (1, 5, 10):
        hits = 0

        for image_index in range(similarity.size(0)):
            topk = similarity[image_index].topk(k).indices

            gt_start = image_index * 5
            gt_end = gt_start + 5

            ground_truth = torch.arange(
                gt_start,
                gt_end,
            )

            if torch.isin(topk, ground_truth).any():
                hits += 1

        image_to_text[k] = hits / similarity.size(0)

    # Text -> Image retrieval
    text_to_image_similarity = similarity.T
    text_to_image = {}

    for k in (1, 5, 10):
        hits = 0

        for text_index in range(
            text_to_image_similarity.size(0)
        ):
            topk = (
                text_to_image_similarity[text_index]
                .topk(k)
                .indices
            )

            ground_truth_image = caption_to_image[text_index]

            if torch.isin(
                topk,
                ground_truth_image.view(1),
            ).any():
                hits += 1

        text_to_image[k] = (
            hits / text_to_image_similarity.size(0)
        )

    mean_r1 = (
        image_to_text[1] + text_to_image[1]
    ) / 2

    return {
        "image_to_text": {
            "R@1": image_to_text[1],
            "R@5": image_to_text[5],
            "R@10": image_to_text[10],
        },
        "text_to_image": {
            "R@1": text_to_image[1],
            "R@5": text_to_image[5],
            "R@10": text_to_image[10],
        },
        "mean_R@1": mean_r1,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    config = load_config(args.config)

    seed = (
        args.seed
        if args.seed is not None
        else config["seed"]
    )

    set_seed(seed)
    device = select_device()

    expected_budget = int(
        args.budget
        if args.budget is not None
        else config["peft"]["total_trainable_budget"]
    )

    allocation_spec = get_allocation_spec(
        args.allocation,
        expected_budget,
    )

    print("=" * 70)
    print("VISION-LANGUAGE LoRA TRAINING")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Allocation: {args.allocation}")
    print(f"Trainable parameter budget: {expected_budget:,}")
    print(
        f"Vision allocation: "
        f"{allocation_spec['vision_percent']}%"
    )
    print(
        f"Language allocation: "
        f"{allocation_spec['language_percent']}%"
    )
    print(f"Vision rank: {allocation_spec['vision_rank']}")
    print(f"Language rank: {allocation_spec['language_rank']}")

    # Configuration
    model_name = config["model"]["name"]
    annotation_path = config["data"]["annotations"]
    image_dir = config["data"]["image_dir"]
    train_split = config["data"]["train_split"]
    val_split = config["data"]["validation_split"]

    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    learning_rate = config["training"]["learning_rate"]
    weight_decay = config["training"]["weight_decay"]
    gradient_clip_norm = config["training"]["gradient_clip_norm"]
    num_workers = config["training"]["num_workers"]

    # Output paths
    if expected_budget == BASE_BUDGET:
        # Preserve the existing 294,912-parameter output convention.
        output_name = (
            f"flickr30k_lora_{args.allocation}_seed{seed}"
        )
    else:
        # Use a separate directory for the 589,824-parameter runs.
        output_name = (
            f"flickr30k_lora_{args.allocation}_"
            f"budget{expected_budget}_seed{seed}"
        )

    output_dir = (
        Path("results")
        / "experiments"
        / output_name
    )

    if output_dir.exists() and not args.overwrite_output:
        raise FileExistsError(
            f"Experiment output already exists: {output_dir}\n"
            "Training has stopped to protect prior results. "
            "Use --overwrite-output only if you intentionally "
            "want to replace this exact run directory."
        )

    best_adapter_dir = output_dir / "best_adapter"
    last_adapter_dir = output_dir / "last_adapter"

    # Processor
    print("\nLoading CLIP processor...")

    processor = CLIPProcessor.from_pretrained(model_name)

    # Datasets
    print("\nLoading datasets...")

    train_dataset = Flickr30kTrainDataset(
        csv_path=annotation_path,
        image_dir=image_dir,
        split=train_split,
    )

    val_dataset = Flickr30kEvalDataset(
        csv_path=annotation_path,
        image_dir=image_dir,
        split=val_split,
    )

    print(f"Training images: {len(train_dataset):,}")
    print(f"Validation images: {len(val_dataset):,}")

    # DataLoader
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        generator=generator,
        collate_fn=lambda batch: collate_train(
            batch,
            processor,
        ),
    )

    # Model
    print("\nLoading CLIP + LoRA...")

    model, allocation_spec = build_lora_model(
        model_name,
        args.allocation,
        expected_budget,
    )

    model.to(device)

    total_params, trainable_params = count_parameters(model)

    vision_trainable, language_trainable = (
        count_modality_parameters(model)
    )

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Vision LoRA parameters: {vision_trainable:,}")
    print(f"Language LoRA parameters: {language_trainable:,}")
    print(f"Expected budget: {expected_budget:,}")

    if trainable_params != expected_budget:
        raise RuntimeError(
            "\nTRAINABLE PARAMETER BUDGET MISMATCH\n"
            f"Expected: {expected_budget:,}\n"
            f"Actual:   {trainable_params:,}"
        )

    # Create output directory only after parameter validation.
    if output_dir.exists():
        # This can only occur if the user explicitly enabled overwrite.
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=False)

    # Optimizer: preserve the original training configuration.
    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = AdamW(
        trainable_parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Metadata
    metadata = {
        "model": model_name,
        "dataset": "Flickr30k",
        "annotation_file": annotation_path,
        "image_directory": image_dir,
        "allocation": args.allocation,
        "trainable_parameter_budget": expected_budget,
        "budget_multiplier": expected_budget // BASE_BUDGET,
        "optimizer": "torch.optim.AdamW",
        "optimizer_learning_rate": learning_rate,
        "optimizer_weight_decay": weight_decay,
        "vision_percent": allocation_spec["vision_percent"],
        "language_percent": allocation_spec["language_percent"],
        "vision_rank": allocation_spec["vision_rank"],
        "language_rank": allocation_spec["language_rank"],
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "vision_trainable_parameters": vision_trainable,
        "language_trainable_parameters": language_trainable,
        "seed": seed,
        "device": str(device),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "gradient_clip_norm": gradient_clip_norm,
        "num_workers": num_workers,
        "target_modules": config["peft"]["target_modules"],
        "max_steps": args.max_steps,
    }

    save_json(output_dir / "metadata.json", metadata)

    # Training
    history = []
    best_mean_r1 = -1.0
    best_epoch = None

    for epoch in range(epochs):
        print("\n" + "=" * 70)
        print(f"Epoch {epoch + 1}/{epochs}")
        print("=" * 70)

        # Deterministic caption rotation
        train_dataset.set_epoch(epoch)

        model.train()
        running_loss = 0.0
        steps = 0

        progress = tqdm(
            train_loader,
            desc=f"Training epoch {epoch + 1}",
        )

        for batch in progress:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad(set_to_none=True)

            loss = clip_loss(
                model=model,
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                gradient_clip_norm,
            )

            optimizer.step()

            loss_value = loss.item()
            running_loss += loss_value
            steps += 1

            progress.set_postfix(
                loss=f"{loss_value:.5f}"
            )

            if (
                args.max_steps is not None
                and steps >= args.max_steps
            ):
                break

        if steps == 0:
            raise RuntimeError("No training steps were executed.")

        epoch_loss = running_loss / steps

        print(f"\nTraining loss: {epoch_loss:.6f}")

        # Validation
        print("\nRunning retrieval validation...")

        metrics = evaluate_retrieval(
            model=model,
            processor=processor,
            dataset=val_dataset,
            device=device,
            batch_size=batch_size,
        )

        print("\nValidation results:")
        print("Image -> Text:")
        print(f"  R@1:  {metrics['image_to_text']['R@1']:.4f}")
        print(f"  R@5:  {metrics['image_to_text']['R@5']:.4f}")
        print(f"  R@10: {metrics['image_to_text']['R@10']:.4f}")

        print("Text -> Image:")
        print(f"  R@1:  {metrics['text_to_image']['R@1']:.4f}")
        print(f"  R@5:  {metrics['text_to_image']['R@5']:.4f}")
        print(f"  R@10: {metrics['text_to_image']['R@10']:.4f}")

        print(f"\nMean R@1: {metrics['mean_R@1']:.4f}")

        epoch_record = {
            "epoch": epoch + 1,
            "train_steps": steps,
            "train_loss": epoch_loss,
            "metrics": metrics,
        }

        history.append(epoch_record)

        save_json(
            output_dir / "history.json",
            history,
        )

        # Save best checkpoint according to validation Mean R@1.
        if metrics["mean_R@1"] > best_mean_r1:
            best_mean_r1 = metrics["mean_R@1"]
            best_epoch = epoch + 1

            if best_adapter_dir.exists():
                shutil.rmtree(best_adapter_dir)

            model.save_pretrained(best_adapter_dir)

            print("\nNew best checkpoint saved.")
            print(f"Best mean R@1: {best_mean_r1:.4f}")

        # Save last checkpoint.
        if last_adapter_dir.exists():
            shutil.rmtree(last_adapter_dir)

        model.save_pretrained(last_adapter_dir)

        # Save current summary.
        summary = {
            "allocation": args.allocation,
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "metrics": metrics,
            "best_epoch": best_epoch,
            "best_mean_R@1": best_mean_r1,
        }

        save_json(
            output_dir / "latest_summary.json",
            summary,
        )

    # Final results
    final_results = {
        "allocation": args.allocation,
        "trainable_parameter_budget": expected_budget,
        "budget_multiplier": expected_budget // BASE_BUDGET,
        "vision_percent": allocation_spec["vision_percent"],
        "language_percent": allocation_spec["language_percent"],
        "vision_rank": allocation_spec["vision_rank"],
        "language_rank": allocation_spec["language_rank"],
        "trainable_parameters": trainable_params,
        "vision_trainable_parameters": vision_trainable,
        "language_trainable_parameters": language_trainable,
        "best_epoch": best_epoch,
        "best_mean_R@1": best_mean_r1,
        "history": history,
    }

    save_json(
        output_dir / "final_results.json",
        final_results,
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Allocation: {args.allocation}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best mean R@1: {best_mean_r1:.4f}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Results directory: {output_dir}")


if __name__ == "__main__":
    main()