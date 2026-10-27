#!/usr/bin/env python3
"""Verify actual PEFT-injected trainable counts for the 589,824-parameter CLIP study.

Uses the same whole-CLIP PeftModel + rank_pattern approach as the working
training implementation. This intentionally avoids wrapping vision_model and
text_model separately, which previously caused CLIP forward compatibility issues.

Run from the repository root:
    python scripts/verify_budget_589824_model.py
    python scripts/verify_budget_589824_model.py --smoke
"""
from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass

import torch
from peft import LoraConfig, get_peft_model
from transformers import CLIPModel

MODEL_NAME = "openai/clip-vit-base-patch32"
EXPECTED_BUDGET = 589_824
TARGET_SUFFIXES = ("q_proj", "v_proj")

@dataclass(frozen=True)
class Allocation:
    name: str
    vision_rank: int
    text_rank: int
    vision_percent: int
    text_percent: int

ALLOCATIONS = (
    Allocation("100_0", 16, 0, 100, 0),
    Allocation("75_25", 12, 6, 75, 25),
    Allocation("50_50", 8, 12, 50, 50),
    Allocation("25_75", 4, 18, 25, 75),
    Allocation("0_100", 0, 24, 0, 100),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Also run a tiny forward/backward smoke test for each allocation.",
    )
    return parser.parse_args()


def build_patterns(model: CLIPModel, vision_rank: int, text_rank: int):
    target_modules: list[str] = []
    rank_pattern: dict[str, int] = {}
    alpha_pattern: dict[str, int] = {}

    vision_layers = model.config.vision_config.num_hidden_layers
    text_layers = model.config.text_config.num_hidden_layers

    for layer in range(vision_layers):
        for suffix in TARGET_SUFFIXES:
            name = f"vision_model.encoder.layers.{layer}.self_attn.{suffix}"
            if vision_rank > 0:
                target_modules.append(name)
                rank_pattern[name] = vision_rank
                alpha_pattern[name] = 2 * vision_rank

    for layer in range(text_layers):
        for suffix in TARGET_SUFFIXES:
            name = f"text_model.encoder.layers.{layer}.self_attn.{suffix}"
            if text_rank > 0:
                target_modules.append(name)
                rank_pattern[name] = text_rank
                alpha_pattern[name] = 2 * text_rank

    if not target_modules:
        raise ValueError("No LoRA target modules configured.")

    return target_modules, rank_pattern, alpha_pattern


def build_lora_model(allocation: Allocation) -> torch.nn.Module:
    base_model = CLIPModel.from_pretrained(MODEL_NAME)
    base_model.requires_grad_(False)
    targets, rank_pattern, alpha_pattern = build_patterns(
        base_model, allocation.vision_rank, allocation.text_rank
    )

    config = LoraConfig(
        r=1,  # positive default required by PEFT; per-module ranks override it
        lora_alpha=2,
        lora_dropout=0.0,
        bias="none",
        target_modules=targets,
        task_type="FEATURE_EXTRACTION",
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
    )
    return get_peft_model(base_model, config)


def count_params(model: torch.nn.Module) -> tuple[int, int, int]:
    total = 0
    vision = 0
    text = 0
    unexpected_trainables: list[str] = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        n = parameter.numel()
        total += n
        if "lora_" not in name:
            unexpected_trainables.append(name)
        elif "vision_model" in name:
            vision += n
        elif "text_model" in name:
            text += n
        else:
            unexpected_trainables.append(name)

    if unexpected_trainables:
        raise AssertionError(
            "Unexpected trainable parameters found (expected LoRA only):\n"
            + "\n".join(unexpected_trainables[:20])
        )
    return total, vision, text


def _extract_embedding(output: object) -> torch.Tensor:
    """Handle the tensor and ModelOutput forms returned by Transformers."""
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        return output.last_hidden_state
    raise TypeError(f"Unsupported embedding output: {type(output)!r}")


def _normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    return embeddings / embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def smoke_test(model: torch.nn.Module) -> None:
    """Test the same separate image/text feature path used by scripts.train.

    Do not call the full PeftModelForFeatureExtraction wrapper here: depending
    on PEFT/Transformers versions, that wrapper injects `inputs_embeds` into
    CLIPModel.forward, which then forwards it again into the CLIP encoders.
    The actual trainer calls get_image_features/get_text_features separately.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)
    model.train()
    batch_size, sequence_length = 2, 16
    base_model = model.get_base_model()
    vocab_size = base_model.config.text_config.vocab_size
    input_ids = torch.randint(
        1, vocab_size - 2, (batch_size, sequence_length), device=device
    )
    input_ids[:, 0] = vocab_size - 2  # CLIP start token for this tokenizer
    input_ids[:, -2] = vocab_size - 1  # CLIP end token for this tokenizer
    input_ids[:, -1] = 0  # padding
    attention_mask = torch.ones_like(input_ids)
    attention_mask[:, -1] = 0
    pixel_values = torch.randn(batch_size, 3, 224, 224, device=device)

    model.zero_grad(set_to_none=True)
    image_output = model.get_image_features(pixel_values=pixel_values)
    text_output = model.get_text_features(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    image_features = _normalize_embeddings(_extract_embedding(image_output))
    text_features = _normalize_embeddings(_extract_embedding(text_output))

    # Match the symmetric contrastive objective used by scripts.train.
    logit_scale = base_model.logit_scale.exp().clamp(max=100)
    logits_per_image = logit_scale * image_features @ text_features.T
    labels = torch.arange(batch_size, device=device)
    loss = (
        torch.nn.functional.cross_entropy(logits_per_image, labels)
        + torch.nn.functional.cross_entropy(logits_per_image.T, labels)
    ) / 2
    loss.backward()

    grad_params = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" in name
    ]
    if not any(
        gradient is not None
        and torch.isfinite(gradient).all().item()
        and torch.count_nonzero(gradient).item() > 0
        for gradient in grad_params
    ):
        raise AssertionError("Smoke test produced no finite nonzero LoRA gradients.")
    print(f"Forward/backward smoke test passed on {device}; loss={loss.item():.6f}")


def main() -> None:
    args = parse_args()
    print(f"Model: {MODEL_NAME}")
    print(f"Expected trainable LoRA budget: {EXPECTED_BUDGET:,}\n")

    for allocation in ALLOCATIONS:
        model = build_lora_model(allocation)
        total, vision, text = count_params(model)
        expected_vision = EXPECTED_BUDGET * allocation.vision_percent // 100
        expected_text = EXPECTED_BUDGET * allocation.text_percent // 100

        print("=" * 78)
        print(f"Allocation: {allocation.name}")
        print(f"Ranks (vision, text): ({allocation.vision_rank}, {allocation.text_rank})")
        print(f"Vision trainable:     {vision:,} (expected {expected_vision:,})")
        print(f"Text trainable:       {text:,} (expected {expected_text:,})")
        print(f"Total trainable:      {total:,} (expected {EXPECTED_BUDGET:,})")

        assert vision == expected_vision, (allocation.name, "vision", vision, expected_vision)
        assert text == expected_text, (allocation.name, "text", text, expected_text)
        assert total == EXPECTED_BUDGET, (allocation.name, total, EXPECTED_BUDGET)

        if args.smoke:
            smoke_test(model)

        del model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nPASS: all five instantiated PEFT configurations match the exact budget.")
    if args.smoke:
        print("PASS: all five configurations completed forward/backward smoke tests.")
    print("This verifies PEFT parameter counts and, with --smoke, basic gradient flow.")

if __name__ == "__main__":
    main()
