from __future__ import annotations

import torch
from peft import LoraConfig, get_peft_model
from transformers import CLIPModel


MODEL_NAME = "openai/clip-vit-base-patch32"

VISION_RANK = 4
TEXT_RANK = 6

EXPECTED_TRAINABLE = 294_912


def build_rank_patterns(
    model: torch.nn.Module,
) -> tuple[dict[str, int], dict[str, int]]:
    rank_pattern = {}
    alpha_pattern = {}

    for name, module in model.named_modules():
        is_target = name.endswith(("q_proj", "v_proj"))

        if not is_target:
            continue

        if name.startswith("vision_model."):
            rank = VISION_RANK
        elif name.startswith("text_model."):
            rank = TEXT_RANK
        else:
            continue

        # PEFT rank_pattern keys can be regex/full module names.
        rank_pattern[f"^{name}$"] = rank
        alpha_pattern[f"^{name}$"] = 2 * rank

    return rank_pattern, alpha_pattern


def count_trainable(
    model: torch.nn.Module,
) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def main() -> None:
    print("Loading CLIP...")

    model = CLIPModel.from_pretrained(MODEL_NAME)

    # Freeze the complete base model.
    model.requires_grad_(False)

    rank_pattern, alpha_pattern = build_rank_patterns(model)

    print(f"Rank-pattern entries: {len(rank_pattern)}")

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        target_modules=[
            "q_proj",
            "v_proj",
        ],
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
        task_type="FEATURE_EXTRACTION",
    )

    # IMPORTANT:
    # Apply PEFT to the COMPLETE CLIP model.
    model = get_peft_model(model, config)

    total_trainable = count_trainable(model)

    vision_trainable = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and name.startswith("base_model.model.vision_model.")
    )

    text_trainable = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and name.startswith("base_model.model.text_model.")
    )

    print("\nParameter summary")
    print("-" * 50)
    print(f"Vision rank:           {VISION_RANK}")
    print(f"Language rank:         {TEXT_RANK}")
    print(f"Vision trainable:      {vision_trainable:,}")
    print(f"Language trainable:    {text_trainable:,}")
    print(f"Total trainable:       {total_trainable:,}")

    assert total_trainable == EXPECTED_TRAINABLE, (
        f"Expected {EXPECTED_TRAINABLE:,}, "
        f"got {total_trainable:,}"
    )

    assert vision_trainable == 147_456
    assert text_trainable == 147_456

    # Verify PEFT actually targeted the expected modules.
    targeted = getattr(model, "targeted_module_names", None)

    if targeted is not None:
        print(f"\nTargeted modules: {len(targeted)}")

    # --------------------------------------------------
    # Forward-pass sanity test
    # --------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    print(f"\nDevice: {device}")

    # One synthetic image and one caption.
    pixel_values = torch.randn(
        1,
        3,
        224,
        224,
        device=device,
    )

    input_ids = torch.randint(
        low=0,
        high=model.config.text_config.vocab_size,
        size=(1, 16),
        device=device,
    )

    attention_mask = torch.ones(
        1,
        16,
        dtype=torch.long,
        device=device,
    )

    with torch.inference_mode():
        image_output = model.get_image_features(
            pixel_values=pixel_values
        )

        text_output = model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    if hasattr(image_output, "pooler_output"):
        image_features = image_output.pooler_output
    else:
        image_features = image_output

    if hasattr(text_output, "pooler_output"):
        text_features = text_output.pooler_output
    else:
        text_features = text_output

    print(
        "Image embedding:",
        tuple(image_features.shape),
    )

    print(
        "Text embedding:",
        tuple(text_features.shape),
    )

    assert image_features.shape == (1, 512)
    assert text_features.shape == (1, 512)

    assert torch.isfinite(image_features).all()
    assert torch.isfinite(text_features).all()

    print("\nTop-level CLIP LoRA verification passed.")


if __name__ == "__main__":
    main()