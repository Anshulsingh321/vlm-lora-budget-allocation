from __future__ import annotations

import torch
from peft import LoraConfig, get_peft_model
from transformers import CLIPModel


MODEL_NAME = "openai/clip-vit-base-patch32"

# Fixed research budget.
BUDGET = 294_912

TARGET_MODULES = [
    "q_proj",
    "v_proj",
]

# Vision r=8 -> 294,912 params
# Text r=12  -> 294,912 params
#
# Exact equal-budget combinations:
# 100/0 -> vision r=8, text r=0
# 75/25  -> vision r=6, text r=3
# 50/50  -> vision r=4, text r=6
# 25/75  -> vision r=2, text r=9
# 0/100  -> vision r=0, text r=12
CONFIGS = {
    "vision_100_language_0": (8, 0),
    "vision_75_language_25": (6, 3),
    "vision_50_language_50": (4, 6),
    "vision_25_language_75": (2, 9),
    "vision_0_language_100": (0, 12),
}


def apply_lora(
    model: CLIPModel,
    module: torch.nn.Module,
    rank: int,
) -> torch.nn.Module:
    if rank == 0:
        return module

    config = LoraConfig(
        r=rank,
        lora_alpha=2 * rank,
        lora_dropout=0.0,
        bias="none",
        target_modules=TARGET_MODULES,
        task_type="FEATURE_EXTRACTION",
    )

    return get_peft_model(module, config)


def count_trainable(module: torch.nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def main() -> None:
    print(f"Target PEFT budget: {BUDGET:,}\n")

    for name, (vision_rank, language_rank) in CONFIGS.items():

        # Fresh model for every configuration.
        model = CLIPModel.from_pretrained(MODEL_NAME)
        model.requires_grad_(False)

        # Apply vision LoRA.
        model.vision_model = apply_lora(
            model,
            model.vision_model,
            vision_rank,
        )

        # Apply language LoRA.
        model.text_model = apply_lora(
            model,
            model.text_model,
            language_rank,
        )

        vision_trainable = count_trainable(
            model.vision_model
        )

        language_trainable = count_trainable(
            model.text_model
        )

        total_trainable = count_trainable(model)

        print("=" * 65)
        print(name)
        print(f"Vision rank:       {vision_rank}")
        print(f"Language rank:     {language_rank}")
        print(
            f"Vision trainable:  {vision_trainable:,}"
        )
        print(
            f"Language trainable:{language_trainable:,}"
        )
        print(
            f"Total trainable:   {total_trainable:,}"
        )

        assert (
            total_trainable == BUDGET
        ), (
            f"{name}: expected {BUDGET:,}, "
            f"got {total_trainable:,}"
        )

    print("\n" + "=" * 65)
    print("ALL FIXED-BUDGET CONFIGURATIONS VERIFIED")
    print("=" * 65)


if __name__ == "__main__":
    main()