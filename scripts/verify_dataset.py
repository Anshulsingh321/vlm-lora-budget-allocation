import json
from pathlib import Path

import pandas as pd


CSV_PATH = Path("data/raw/flickr30k/flickr_annotations_30k.csv")


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"File not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    print("Columns:")
    print(df.columns.tolist())

    print("\nRows:", len(df))

    # Parse JSON-encoded caption and sentence-id lists.
    df["captions"] = df["raw"].apply(json.loads)
    df["sentids_parsed"] = df["sentids"].apply(json.loads)

    print("\nSplit distribution:")
    print(df["split"].value_counts())

    print("\nUnique images:", df["img_id"].nunique())

    captions_per_image = df["captions"].apply(len)

    print("\nCaptions per image:")
    print(captions_per_image.value_counts().sort_index())

    sentids_per_image = df["sentids_parsed"].apply(len)

    print("\nSentence IDs per image:")
    print(sentids_per_image.value_counts().sort_index())

    print("\nImages per split:")
    images_per_split = df.groupby("split")["img_id"].nunique()
    print(images_per_split)

    # ---------- Validation ----------

    print("\nChecking 5 captions per image...")
    assert captions_per_image.eq(5).all()

    print("Checking 5 sentence IDs per image...")
    assert sentids_per_image.eq(5).all()

    print("Checking unique image IDs...")
    assert df["img_id"].is_unique

    print("Checking split isolation...")
    split_counts = df.groupby("img_id")["split"].nunique()
    assert split_counts.max() == 1

    expected = {
        "train": 29_000,
        "val": 1_014,
        "test": 1_000,
    }

    print("\nChecking split sizes...")

    for split, expected_count in expected.items():
        actual = images_per_split.get(split, 0)

        print(
            f"{split}: expected={expected_count}, actual={actual}"
        )

        assert actual == expected_count

    print("\nDataset annotation verification passed.")


if __name__ == "__main__":
    main()