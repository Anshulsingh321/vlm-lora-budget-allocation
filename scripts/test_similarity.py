import torch

from src.data.flickr30k import Flickr30kDataset
from src.models.clip import FrozenCLIP


ANNOTATIONS = "data/raw/flickr30k/flickr_annotations_30k.csv"
IMAGE_DIR = "data/processed/flickr30k/images"


def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = Flickr30kDataset(
        annotation_path=ANNOTATIONS,
        image_dir=IMAGE_DIR,
        split="test",
    )

    sample = dataset[0]

    model = FrozenCLIP(device=device)

    image_embedding = model.encode_images(
        [sample["image"]]
    )

    text_embeddings = model.encode_texts(
        sample["captions"]
    )

    # Cosine similarity because embeddings are normalized.
    similarities = image_embedding @ text_embeddings.T

    similarities = similarities.squeeze(0)

    ranked_indices = torch.argsort(
        similarities,
        descending=True,
    )

    print("Image:", sample["filename"])
    print("\nCaption ranking:\n")

    for rank, index in enumerate(ranked_indices, start=1):
        print(
            f"{rank}. "
            f"similarity={similarities[index].item():.4f}"
        )
        print(f"   {sample['captions'][index]}\n")


if __name__ == "__main__":
    main()