from pathlib import Path

from torch.utils.data import DataLoader

from src.data.flickr30k import Flickr30kDataset


ANNOTATIONS = Path(
    "data/raw/flickr30k/flickr_annotations_30k.csv"
)

IMAGE_DIR = Path(
    "data/processed/flickr30k/images"
)


def collate_fn(batch: list[dict]) -> dict:
    return {
        "images": [item["image"] for item in batch],
        "captions": [item["captions"] for item in batch],
        "filenames": [item["filename"] for item in batch],
        "img_ids": [item["img_id"] for item in batch],
    }


def main() -> None:
    dataset = Flickr30kDataset(
        annotation_path=ANNOTATIONS,
        image_dir=IMAGE_DIR,
        split="train",
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    batch = next(iter(loader))

    print("Dataset size:", len(dataset))
    print("Batch size:", len(batch["images"]))

    print("\nFirst image:")
    print("Filename:", batch["filenames"][0])
    print("Image size:", batch["images"][0].size)

    print("\nCaptions:")
    for caption in batch["captions"][0]:
        print("-", caption)

    print("\nDataLoader smoke test passed.")


if __name__ == "__main__":
    main()