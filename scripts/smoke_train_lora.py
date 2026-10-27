from __future__ import annotations

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor

from src.data.flickr30k import Flickr30kDataset


MODEL_NAME = "openai/clip-vit-base-patch32"

ANNOTATIONS = "data/raw/flickr30k/flickr_annotations_30k.csv"
IMAGE_DIR = "data/processed/flickr30k/images"

BATCH_SIZE = 15
MAX_STEPS = 1
LEARNING_RATE = 1e-4

VISION_RANK = 4
TEXT_RANK = 6

EXPECTED_TRAINABLE = 294_912


def collate_fn(batch: list[dict]) -> dict:
    # One caption per image for the smoke test only.
    return {
        "images": [item["image"] for item in batch],
        "texts": [item["captions"][0] for item in batch],
    }


def build_model() -> CLIPModel:
    model = CLIPModel.from_pretrained(MODEL_NAME)

    # Freeze the complete pretrained CLIP.
    model.requires_grad_(False)

    rank_pattern = {}
    alpha_pattern = {}

    for name, module in model.named_modules():
        if not name.endswith(("q_proj", "v_proj")):
            continue

        if name.startswith("vision_model."):
            rank = VISION_RANK
        elif name.startswith("text_model."):
            rank = TEXT_RANK
        else:
            continue

        rank_pattern[f"^{name}$"] = rank
        alpha_pattern[f"^{name}$"] = 2 * rank

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
        task_type="FEATURE_EXTRACTION",
    )

    # Apply PEFT once to the complete CLIP model.
    model = get_peft_model(model, config)

    return model


def count_trainable(model: torch.nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def get_trainable_parameter_names(model: torch.nn.Module) -> list[str]:
    return [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]


def main() -> None:
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print(f"Device: {device}")

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------
    dataset = Flickr30kDataset(
        annotation_path=ANNOTATIONS,
        image_dir=IMAGE_DIR,
        split="train",
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    print("\nLoading CLIP + LoRA...")

    model = build_model()
    model.to(device)
    model.train()

    trainable_params = count_trainable(model)

    print(f"Trainable parameters: {trainable_params:,}")

    assert trainable_params == EXPECTED_TRAINABLE, (
        f"Expected {EXPECTED_TRAINABLE:,}, "
        f"got {trainable_params:,}"
    )

    # --------------------------------------------------
    # Processor
    # --------------------------------------------------
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------
    optimizer = AdamW(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=LEARNING_RATE,
    )

    # --------------------------------------------------
    # Track one vision and one language LoRA parameter
    # --------------------------------------------------
    vision_name = None
    text_name = None

    vision_before = None
    text_before = None

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if vision_name is None and "vision_model" in name and "lora_B" in name:
            vision_name = name
            vision_before = parameter.detach().clone()

        if text_name is None and "text_model" in name and "lora_B" in name:
            text_name = name
            text_before = parameter.detach().clone()

        if vision_name is not None and text_name is not None:
            break

    assert vision_name is not None
    assert text_name is not None
    assert vision_before is not None
    assert text_before is not None

    print(f"Tracking vision parameter:   {vision_name}")
    print(f"Tracking language parameter: {text_name}")

    # --------------------------------------------------
    # Training smoke test
    # --------------------------------------------------
    print("\nStarting smoke training...\n")

    step = 0

    for batch in loader:
        if step >= MAX_STEPS:
            break

        images = batch["images"]
        texts = batch["texts"]

        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        optimizer.zero_grad(set_to_none=True)

        # --------------------------------------------------
        # Forward
        # --------------------------------------------------
        image_output = model.get_image_features(
            pixel_values=inputs["pixel_values"]
        )

        text_output = model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

        if hasattr(image_output, "pooler_output"):
            image_features = image_output.pooler_output
        else:
            image_features = image_output

        if hasattr(text_output, "pooler_output"):
            text_features = text_output.pooler_output
        else:
            text_features = text_output

        # --------------------------------------------------
        # Normalize
        # --------------------------------------------------
        image_features = F.normalize(
            image_features,
            p=2,
            dim=-1,
        )

        text_features = F.normalize(
            text_features,
            p=2,
            dim=-1,
        )

        # --------------------------------------------------
        # CLIP contrastive logits
        # --------------------------------------------------
        logit_scale = model.base_model.model.logit_scale.exp()

        logits_per_image = (
            logit_scale
            * image_features
            @ text_features.T
        )

        logits_per_text = logits_per_image.T

        labels = torch.arange(
            len(texts),
            device=device,
        )

        # Symmetric CLIP loss.
        loss_image = F.cross_entropy(
            logits_per_image,
            labels,
        )

        loss_text = F.cross_entropy(
            logits_per_text,
            labels,
        )

        loss = 0.5 * (
            loss_image + loss_text
        )

        # --------------------------------------------------
        # Backward
        # --------------------------------------------------
        loss.backward()

        assert torch.isfinite(loss).item(), (
            "Loss became non-finite."
        )

        gradient_count = 0

        for parameter in model.parameters():
            if parameter.requires_grad:
                if parameter.grad is not None:
                    gradient_count += 1

                    assert torch.isfinite(
                        parameter.grad
                    ).all(), (
                        "Non-finite gradient detected."
                    )

        assert gradient_count > 0, (
            "No trainable gradients were produced."
        )

        # --------------------------------------------------
        # Optimizer step
        # --------------------------------------------------
        optimizer.step()

        print(
            f"Step {step + 1}/{MAX_STEPS} "
            f"- loss: {loss.item():.6f}"
        )

        step += 1

    # --------------------------------------------------
    # Verify both modalities actually updated
    # --------------------------------------------------
    vision_after = None
    text_after = None

    for name, parameter in model.named_parameters():
        if name == vision_name:
            vision_after = parameter.detach().clone()

        if name == text_name:
            text_after = parameter.detach().clone()

    assert vision_after is not None
    assert text_after is not None

    vision_changed = not torch.equal(
        vision_before,
        vision_after,
    )

    text_changed = not torch.equal(
        text_before,
        text_after,
    )

    print(f"\nVision LoRA parameter changed:   {vision_changed}")
    print(f"Language LoRA parameter changed: {text_changed}")

    assert vision_changed, (
        "Vision LoRA parameters did not update."
    )

    assert text_changed, (
        "Language LoRA parameters did not update."
    )

    print("\n" + "=" * 60)
    print("LO RA TRAINING SMOKE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()