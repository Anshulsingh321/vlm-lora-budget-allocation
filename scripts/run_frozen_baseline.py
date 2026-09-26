import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.data.flickr30k import Flickr30kDataset
from src.models.clip import FrozenCLIP
from src.evaluation.retrieval import recall_at_k


ANNOTATIONS = "data/raw/flickr30k/flickr_annotations_30k.csv"
IMAGE_DIR = "data/processed/flickr30k/images"

OUTPUT_DIR = Path("results/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_BATCH_SIZE = 32
TEXT_BATCH_SIZE = 64


def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    dataset = Flickr30kDataset(
        annotation_path=ANNOTATIONS,
        image_dir=IMAGE_DIR,
        split="test",
    )

    model = FrozenCLIP(device=device)

    # --------------------------------------------------
    # 1. Encode images
    # --------------------------------------------------

    image_embeddings = []

    print("\nEncoding images...")

    for start in tqdm(
        range(0, len(dataset), IMAGE_BATCH_SIZE)
    ):
        end = min(
            start + IMAGE_BATCH_SIZE,
            len(dataset),
        )

        samples = [
            dataset[i]
            for i in range(start, end)
        ]

        images = [
            sample["image"]
            for sample in samples
        ]

        embeddings = model.encode_images(images)

        image_embeddings.append(
            embeddings.cpu()
        )

    image_embeddings = torch.cat(
        image_embeddings,
        dim=0,
    )

    print(
        "Image embeddings:",
        tuple(image_embeddings.shape),
    )

    # --------------------------------------------------
    # 2. Encode captions
    # --------------------------------------------------

    captions = []
    caption_to_image = []

    for image_idx in range(len(dataset)):
        image_captions = dataset[image_idx]["captions"]

        for caption in image_captions:
            captions.append(caption)
            caption_to_image.append(image_idx)

    text_embeddings = []

    print("\nEncoding captions...")

    for start in tqdm(
        range(0, len(captions), TEXT_BATCH_SIZE)
    ):
        end = min(
            start + TEXT_BATCH_SIZE,
            len(captions),
        )

        batch = captions[start:end]

        embeddings = model.encode_texts(batch)

        text_embeddings.append(
            embeddings.cpu()
        )

    text_embeddings = torch.cat(
        text_embeddings,
        dim=0,
    )

    print(
        "Text embeddings:",
        tuple(text_embeddings.shape),
    )

    # --------------------------------------------------
    # 3. Similarity matrix
    # --------------------------------------------------

    print("\nComputing similarity matrix...")

    similarity = image_embeddings @ text_embeddings.T

    print(
        "Similarity matrix:",
        tuple(similarity.shape),
    )

    # --------------------------------------------------
    # 4. Image → Text ground truth
    # --------------------------------------------------

    image_to_text_ground_truth = []

    for image_idx in range(len(dataset)):
        correct_caption_indices = [
            caption_idx
            for caption_idx, image_id
            in enumerate(caption_to_image)
            if image_id == image_idx
        ]

        image_to_text_ground_truth.append(
            correct_caption_indices
        )

    # --------------------------------------------------
    # 5. Text → Image ground truth
    # --------------------------------------------------

    text_to_image_ground_truth = [
        [image_id]
        for image_id in caption_to_image
    ]

    # --------------------------------------------------
    # 6. Evaluate Image → Text
    # --------------------------------------------------

    image_to_text = recall_at_k(
        similarity,
        image_to_text_ground_truth,
    )

    # --------------------------------------------------
    # 7. Evaluate Text → Image
    # --------------------------------------------------

    text_to_image = recall_at_k(
        similarity.T,
        text_to_image_ground_truth,
    )

    # --------------------------------------------------
    # 8. Save
    # --------------------------------------------------

    results = {
        "model": "openai/clip-vit-base-patch32",
        "dataset": "flickr30k",
        "split": "test",
        "fine_tuned": False,
        "image_count": len(dataset),
        "caption_count": len(captions),
        "image_to_text": image_to_text,
        "text_to_image": text_to_image,
    }

    output_path = (
        OUTPUT_DIR / "frozen_clip_flickr30k.json"
    )

    output_path.write_text(
        json.dumps(results, indent=2)
    )

    # --------------------------------------------------
    # 9. Print results
    # --------------------------------------------------

    print("\n" + "=" * 50)
    print("FROZEN CLIP BASELINE")
    print("=" * 50)

    print("\nImage → Text")
    for metric, value in image_to_text.items():
        print(f"{metric}: {value:.4f}")

    print("\nText → Image")
    for metric, value in text_to_image.items():
        print(f"{metric}: {value:.4f}")

    print("\nSaved to:")
    print(output_path)


if __name__ == "__main__":
    main()