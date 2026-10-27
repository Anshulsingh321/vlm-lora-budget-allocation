# scripts/verify_coco.py

from __future__ import annotations

import json
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

from datasets import load_dataset


# ============================================================================
# CONFIGURATION
# ============================================================================

WDS_DATASET = "undefined443/coco-karpathy-wds"

# Andrej Karpathy's COCO split archive.
KARPATHY_ARCHIVE_URL = (
    "https://cs.stanford.edu/"
    "people/karpathy/deepimagesent/"
    "caption_datasets.zip"
)

# COCO 2014 validation caption annotations.
#
# This repository hosts the COCO 2014 caption annotation file used by
# the standard COCO caption tooling.
COCO_CAPTIONS_VAL_URL = (
    "https://raw.githubusercontent.com/"
    "tylin/coco-caption/master/"
    "annotations/"
    "captions_val2014.json"
)

# Local paths.
COCO_DIR = Path("data") / "raw" / "coco"

KARPATHY_DIR = COCO_DIR / "karpathy"
ANNOTATIONS_DIR = COCO_DIR / "annotations"

KARPATHY_ARCHIVE = (
    KARPATHY_DIR / "caption_datasets.zip"
)

KARPATHY_JSON = (
    KARPATHY_DIR / "dataset_coco.json"
)

COCO_CAPTIONS_VAL_JSON = (
    ANNOTATIONS_DIR / "captions_val2014.json"
)

CLEAN_METADATA_JSON = (
    KARPATHY_DIR / "coco_test_metadata.json"
)


# Expected Karpathy COCO retrieval benchmark size.
EXPECTED_TEST_IMAGES = 5000

# Standard retrieval setup:
# 5 captions per image.
EXPECTED_CAPTIONS_PER_IMAGE = 5

DOWNLOAD_RETRIES = 3
DOWNLOAD_TIMEOUT = 60

USER_AGENT = (
    "vlp-research/1.0 "
    "(CLIP cross-dataset evaluation)"
)


# ============================================================================
# DOWNLOAD UTILITIES
# ============================================================================

def download_file(
    url: str,
    output_path: Path,
) -> None:
    """
    Download a file with retries.

    Existing non-empty files are reused.
    Partial files are deleted on failure.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Reuse an existing valid-looking cached file.
    if output_path.exists():
        if output_path.stat().st_size > 0:
            print(
                f"Using cached file:\n"
                f"{output_path}"
            )
            return

        # Delete zero-byte cache.
        output_path.unlink()

    temp_path = output_path.with_suffix(
        output_path.suffix + ".part"
    )

    if temp_path.exists():
        temp_path.unlink()

    last_error = None

    print(f"\nDownloading:\n{url}")

    for attempt in range(
        1,
        DOWNLOAD_RETRIES + 1,
    ):
        print(
            f"Attempt {attempt}/"
            f"{DOWNLOAD_RETRIES}..."
        )

        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": USER_AGENT
                },
            )

            with urllib.request.urlopen(
                request,
                timeout=DOWNLOAD_TIMEOUT,
            ) as response:

                with open(
                    temp_path,
                    "wb",
                ) as destination:

                    shutil.copyfileobj(
                        response,
                        destination,
                    )

            if (
                not temp_path.exists()
                or temp_path.stat().st_size == 0
            ):
                raise RuntimeError(
                    "Downloaded file is empty."
                )

            temp_path.replace(output_path)

            print(
                f"Downloaded successfully:\n"
                f"{output_path}"
            )

            return

        except Exception as exc:
            last_error = exc

            print(
                f"Download failed: {exc}"
            )

            if temp_path.exists():
                temp_path.unlink()

            if attempt < DOWNLOAD_RETRIES:
                print("Retrying...")
                time.sleep(2)

    raise RuntimeError(
        f"Failed to download:\n"
        f"{url}\n"
        f"Last error: {last_error}"
    )


# ============================================================================
# KARPATHY SPLIT
# ============================================================================

def extract_karpathy_json() -> None:
    """
    Extract dataset_coco.json from Karpathy's archive.
    """

    if (
        KARPATHY_JSON.exists()
        and KARPATHY_JSON.stat().st_size > 0
    ):
        print(
            f"Using cached extracted file:\n"
            f"{KARPATHY_JSON}"
        )
        return

    print(
        "\nExtracting dataset_coco.json..."
    )

    with zipfile.ZipFile(
        KARPATHY_ARCHIVE,
        "r",
    ) as archive:

        candidates = [
            name
            for name in archive.namelist()
            if name.endswith(
                "dataset_coco.json"
            )
        ]

        if not candidates:
            raise RuntimeError(
                "dataset_coco.json was not found "
                "in the Karpathy archive."
            )

        # There should be exactly one.
        member = candidates[0]

        print(
            f"Archive member:\n"
            f"{member}"
        )

        with (
            archive.open(member) as source,
            open(
                KARPATHY_JSON,
                "wb",
            ) as destination,
        ):
            shutil.copyfileobj(
                source,
                destination,
            )

    if (
        not KARPATHY_JSON.exists()
        or KARPATHY_JSON.stat().st_size == 0
    ):
        raise RuntimeError(
            "Failed to extract dataset_coco.json."
        )

    print(
        f"Extracted to:\n"
        f"{KARPATHY_JSON}"
    )


def load_karpathy_test() -> dict[int, dict]:
    """
    Load the Karpathy COCO test split.

    Important:
    We select the first five Karpathy sentids for every image.

    This matches the standard VSE++/Karpathy retrieval protocol.
    """

    download_file(
        KARPATHY_ARCHIVE_URL,
        KARPATHY_ARCHIVE,
    )

    extract_karpathy_json()

    print(
        "\nLoading Karpathy split..."
    )

    try:
        with open(
            KARPATHY_JSON,
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Karpathy JSON is not valid JSON:\n"
            f"{KARPATHY_JSON}\n"
            f"Error: {exc}"
        ) from exc

    images = data.get("images")

    if not isinstance(
        images,
        list,
    ):
        raise RuntimeError(
            "Karpathy JSON does not contain "
            "a valid 'images' list."
        )

    test_items = [
        item
        for item in images
        if item.get("split") == "test"
    ]

    if len(test_items) != EXPECTED_TEST_IMAGES:
        raise RuntimeError(
            f"Unexpected Karpathy test size: "
            f"{len(test_items)} "
            f"(expected "
            f"{EXPECTED_TEST_IMAGES})"
        )

    test_metadata: dict[int, dict] = {}

    # Sentence IDs should uniquely identify caption annotations.
    globally_selected_sentids: set[int] = set()

    for item in test_items:

        if "cocoid" not in item:
            raise RuntimeError(
                "Karpathy test item is missing "
                "'cocoid'."
            )

        coco_id = int(
            item["cocoid"]
        )

        if coco_id in test_metadata:
            raise RuntimeError(
                "Duplicate COCO ID in Karpathy "
                f"test split: {coco_id}"
            )

        raw_sentids = item.get(
            "sentids"
        )

        if not isinstance(
            raw_sentids,
            list,
        ):
            raise RuntimeError(
                f"COCO ID {coco_id} does not "
                "contain a valid 'sentids' list."
            )

        if len(raw_sentids) < (
            EXPECTED_CAPTIONS_PER_IMAGE
        ):
            raise RuntimeError(
                f"COCO ID {coco_id} has only "
                f"{len(raw_sentids)} Karpathy "
                "sentence IDs; expected at "
                f"least {EXPECTED_CAPTIONS_PER_IMAGE}."
            )

        # ---------------------------------------------------------------
        # IMPORTANT:
        # Standard Karpathy/VSE++ protocol.
        # Use FIRST FIVE sentence IDs.
        # ---------------------------------------------------------------

        selected_sentids = [
            int(x)
            for x in raw_sentids[
                :EXPECTED_CAPTIONS_PER_IMAGE
            ]
        ]

        if len(
            set(selected_sentids)
        ) != EXPECTED_CAPTIONS_PER_IMAGE:

            raise RuntimeError(
                f"COCO ID {coco_id} contains "
                "duplicate selected sentence IDs:\n"
                f"{selected_sentids}"
            )

        overlap = (
            globally_selected_sentids
            .intersection(selected_sentids)
        )

        if overlap:
            raise RuntimeError(
                "Karpathy sentence IDs are "
                "reused across multiple images.\n"
                f"COCO ID: {coco_id}\n"
                f"Duplicate IDs: "
                f"{sorted(overlap)[:10]}"
            )

        globally_selected_sentids.update(
            selected_sentids
        )

        test_metadata[coco_id] = {
            "filename": item.get(
                "filename"
            ),
            "filepath": item.get(
                "filepath"
            ),
            "sentids": selected_sentids,
        }

    if len(test_metadata) != (
        EXPECTED_TEST_IMAGES
    ):
        raise RuntimeError(
            "Final Karpathy metadata count "
            "does not match expected test size."
        )

    total_selected_sentids = sum(
        len(item["sentids"])
        for item in test_metadata.values()
    )

    expected_total_sentids = (
        EXPECTED_TEST_IMAGES
        * EXPECTED_CAPTIONS_PER_IMAGE
    )

    if total_selected_sentids != (
        expected_total_sentids
    ):
        raise RuntimeError(
            "Unexpected selected sentence ID "
            f"count: {total_selected_sentids:,} "
            f"(expected "
            f"{expected_total_sentids:,})"
        )

    print(
        f"Karpathy test images: "
        f"{len(test_metadata):,}"
    )

    print(
        f"Selected captions: "
        f"{total_selected_sentids:,}"
    )

    print(
        "Karpathy test split verified ✅"
    )

    return test_metadata


# ============================================================================
# COCO CAPTION ANNOTATIONS
# ============================================================================

def load_coco_captions(
    test_metadata: dict[int, dict],
) -> dict[int, list[str]]:
    """
    Resolve the selected Karpathy sentence IDs against
    the COCO 2014 caption annotations.

    We DO NOT assume that the raw COCO annotation file
    has exactly five captions per image.

    Instead:
        Karpathy sentids[:5]
              ↓
        COCO annotation ID
              ↓
        caption text

    This gives the exact five captions used by the
    Karpathy/VSE++ retrieval protocol.
    """

    download_file(
        COCO_CAPTIONS_VAL_URL,
        COCO_CAPTIONS_VAL_JSON,
    )

    print(
        "\nLoading COCO 2014 validation captions..."
    )

    try:
        with open(
            COCO_CAPTIONS_VAL_JSON,
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"COCO caption JSON is not valid JSON:\n"
            f"{COCO_CAPTIONS_VAL_JSON}\n"
            f"Error: {exc}"
        ) from exc

    annotation_type = data.get(
        "type"
    )

    if (
        annotation_type is not None
        and annotation_type != "captions"
    ):
        raise RuntimeError(
            "Unexpected COCO annotation type: "
            f"{annotation_type}"
        )

    annotations = data.get(
        "annotations"
    )

    if not isinstance(
        annotations,
        list,
    ):
        raise RuntimeError(
            "COCO caption JSON does not "
            "contain a valid 'annotations' list."
        )

    # ---------------------------------------------------------------------
    # Build:
    #
    # COCO caption annotation ID
    #              ->
    # annotation
    # ---------------------------------------------------------------------

    annotations_by_id: dict[int, dict] = {}

    for annotation in annotations:

        if "id" not in annotation:
            raise RuntimeError(
                "COCO caption annotation is "
                "missing 'id'."
            )

        annotation_id = int(
            annotation["id"]
        )

        if annotation_id in annotations_by_id:
            raise RuntimeError(
                "Duplicate COCO caption "
                f"annotation ID: "
                f"{annotation_id}"
            )

        annotations_by_id[
            annotation_id
        ] = annotation

    captions_by_image: dict[
        int,
        list[str],
    ] = {}

    missing_sentids = []
    wrong_image_ids = []

    selected_annotation_ids: set[int] = set()

    # ---------------------------------------------------------------------
    # Resolve the exact five captions for each test image.
    # ---------------------------------------------------------------------

    for coco_id, metadata in (
        test_metadata.items()
    ):

        captions: list[str] = []

        for sentid in metadata[
            "sentids"
        ]:

            if sentid in selected_annotation_ids:
                raise RuntimeError(
                    "Selected annotation ID "
                    "was reused:\n"
                    f"{sentid}"
                )

            selected_annotation_ids.add(
                sentid
            )

            annotation = annotations_by_id.get(
                sentid
            )

            if annotation is None:
                missing_sentids.append(
                    (
                        coco_id,
                        sentid,
                    )
                )
                continue

            annotation_image_id = int(
                annotation["image_id"]
            )

            if annotation_image_id != coco_id:
                wrong_image_ids.append(
                    (
                        coco_id,
                        sentid,
                        annotation_image_id,
                    )
                )
                continue

            caption = annotation.get(
                "caption"
            )

            if (
                not isinstance(
                    caption,
                    str,
                )
                or not caption.strip()
            ):
                raise RuntimeError(
                    "Invalid caption for "
                    f"COCO ID {coco_id}, "
                    f"annotation ID {sentid}"
                )

            captions.append(
                caption.strip()
            )

        if len(captions) != (
            EXPECTED_CAPTIONS_PER_IMAGE
        ):
            raise RuntimeError(
                f"COCO ID {coco_id} resolved "
                f"to {len(captions)} captions; "
                f"expected "
                f"{EXPECTED_CAPTIONS_PER_IMAGE}."
            )

        captions_by_image[
            coco_id
        ] = captions

    # ---------------------------------------------------------------------
    # Validation failures.
    # ---------------------------------------------------------------------

    if missing_sentids:
        raise RuntimeError(
            f"{len(missing_sentids)} Karpathy "
            "caption IDs were not found in "
            "the COCO annotations.\n"
            f"Examples: "
            f"{missing_sentids[:10]}"
        )

    if wrong_image_ids:
        raise RuntimeError(
            "Karpathy sentence IDs mapped "
            "to the wrong COCO image.\n"
            f"Examples: "
            f"{wrong_image_ids[:10]}"
        )

    # ---------------------------------------------------------------------
    # Final cardinality validation.
    # ---------------------------------------------------------------------

    number_of_images = len(
        captions_by_image
    )

    total_captions = sum(
        len(captions)
        for captions
        in captions_by_image.values()
    )

    expected_total_captions = (
        EXPECTED_TEST_IMAGES
        * EXPECTED_CAPTIONS_PER_IMAGE
    )

    if number_of_images != (
        EXPECTED_TEST_IMAGES
    ):
        raise RuntimeError(
            "Unexpected number of captioned "
            f"test images: "
            f"{number_of_images:,} "
            f"(expected "
            f"{EXPECTED_TEST_IMAGES:,})"
        )

    if total_captions != (
        expected_total_captions
    ):
        raise RuntimeError(
            "Unexpected total caption count: "
            f"{total_captions:,} "
            f"(expected "
            f"{expected_total_captions:,})"
        )

    # Every image must have exactly five.
    incorrect_counts = [
        (
            coco_id,
            len(captions),
        )
        for coco_id, captions
        in captions_by_image.items()
        if len(captions)
        != EXPECTED_CAPTIONS_PER_IMAGE
    ]

    if incorrect_counts:
        raise RuntimeError(
            "Some test images do not have "
            "exactly five selected captions.\n"
            f"Examples: "
            f"{incorrect_counts[:10]}"
        )

    print(
        f"Captioned test images: "
        f"{number_of_images:,}"
    )

    print(
        f"Selected COCO captions: "
        f"{total_captions:,}"
    )

    print(
        "Exactly 5 selected captions/image ✅"
    )

    return captions_by_image


# ============================================================================
# WDS IMAGE STREAM
# ============================================================================

def verify_wds(
    test_ids: set[int],
) -> None:
    """
    Verify that the WDS test stream contains exactly
    the 5,000 Karpathy test image IDs.

    Important:
    WDS captions are NOT used.

    Only:
        - image bytes
        - cocoid
    are used.

    This avoids the caption-count inconsistency seen in
    some WDS metadata entries.
    """

    print(
        "\nLoading WDS image stream..."
    )

    try:
        dataset = load_dataset(
            WDS_DATASET,
            split="test",
            streaming=True,
        )

    except Exception as exc:
        raise RuntimeError(
            "Failed to load the COCO WDS dataset:\n"
            f"{WDS_DATASET}\n"
            f"Error: {exc}"
        ) from exc

    wds_ids: set[int] = set()

    sample_count = 0

    examples = []

    for sample in dataset:

        if "jpg" not in sample:
            raise RuntimeError(
                "WDS sample does not contain "
                "'jpg'."
            )

        if "json" not in sample:
            raise RuntimeError(
                "WDS sample does not contain "
                "'json'."
            )

        image = sample["jpg"]
        metadata = sample["json"]

        if "cocoid" not in metadata:
            raise RuntimeError(
                "WDS sample metadata does not "
                "contain 'cocoid'."
            )

        coco_id = int(
            metadata["cocoid"]
        )

        wds_ids.add(
            coco_id
        )

        if len(examples) < 3:
            examples.append(
                {
                    "cocoid": coco_id,
                    "image_size": image.size,
                    "key": sample.get(
                        "__key__"
                    ),
                    "wds_caption_count": len(
                        metadata.get(
                            "captions",
                            [],
                        )
                    ),
                }
            )

        sample_count += 1

        # The COCO Karpathy WDS test stream is
        # expected to contain exactly 5,000 samples.
        #
        # Once we have 5,000 samples, we stop.
        #
        # The ID-set validation below ensures those
        # samples exactly correspond to the Karpathy
        # test split.
        if sample_count >= (
            EXPECTED_TEST_IMAGES
        ):
            break

    if sample_count != (
        EXPECTED_TEST_IMAGES
    ):
        raise RuntimeError(
            f"WDS returned "
            f"{sample_count:,} samples; "
            f"expected "
            f"{EXPECTED_TEST_IMAGES:,}."
        )

    if len(wds_ids) != (
        EXPECTED_TEST_IMAGES
    ):
        raise RuntimeError(
            "WDS contains duplicate COCO IDs "
            "within the evaluated test stream.\n"
            f"Samples: {sample_count:,}\n"
            f"Unique IDs: {len(wds_ids):,}"
        )

    print(
        f"\nWDS test samples: "
        f"{sample_count:,}"
    )

    print(
        "\nFirst WDS samples:"
    )

    for example in examples:
        print(
            f"  COCO ID: "
            f"{example['cocoid']}"
        )

        print(
            f"  Image size: "
            f"{example['image_size']}"
        )

        print(
            f"  WDS caption count: "
            f"{example['wds_caption_count']}"
        )

        print(
            f"  Key: "
            f"{example['key']}"
        )

    missing = (
        test_ids - wds_ids
    )

    extra = (
        wds_ids - test_ids
    )

    print(
        "\nWDS / Karpathy ID comparison:"
    )

    print(
        f"  Karpathy IDs: "
        f"{len(test_ids):,}"
    )

    print(
        f"  WDS IDs:      "
        f"{len(wds_ids):,}"
    )

    print(
        f"  Missing:      "
        f"{len(missing):,}"
    )

    print(
        f"  Extra:        "
        f"{len(extra):,}"
    )

    if missing:
        examples_missing = sorted(
            missing
        )[:20]

        raise RuntimeError(
            "WDS is missing COCO test IDs.\n"
            f"Examples: "
            f"{examples_missing}"
        )

    if extra:
        examples_extra = sorted(
            extra
        )[:20]

        raise RuntimeError(
            "WDS contains IDs outside "
            "the Karpathy test split.\n"
            f"Examples: "
            f"{examples_extra}"
        )

    print(
        "  ID sets match ✅"
    )


# ============================================================================
# SAVE CLEAN METADATA
# ============================================================================

def save_clean_metadata(
    test_metadata: dict[int, dict],
    captions_by_image: dict[int, list[str]],
) -> None:
    """
    Save a single clean metadata file used by evaluate_coco.py.

    This file contains:

        COCO ID
        filename
        filepath
        selected Karpathy sentids
        selected five caption strings
    """

    clean = {}

    for coco_id in sorted(
        test_metadata
    ):

        metadata = test_metadata[
            coco_id
        ]

        if coco_id not in captions_by_image:
            raise RuntimeError(
                "Missing captions for COCO ID "
                f"{coco_id}."
            )

        captions = captions_by_image[
            coco_id
        ]

        if len(captions) != (
            EXPECTED_CAPTIONS_PER_IMAGE
        ):
            raise RuntimeError(
                f"COCO ID {coco_id} has "
                f"{len(captions)} captions "
                "during metadata save."
            )

        clean[str(coco_id)] = {
            "cocoid": coco_id,
            "filename": metadata[
                "filename"
            ],
            "filepath": metadata[
                "filepath"
            ],
            "sentids": metadata[
                "sentids"
            ],
            "captions": captions,
        }

    if len(clean) != (
        EXPECTED_TEST_IMAGES
    ):
        raise RuntimeError(
            "Clean metadata does not contain "
            "exactly 5,000 images."
        )

    CLEAN_METADATA_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        CLEAN_METADATA_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            clean,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"\nClean metadata saved to:\n"
        f"{CLEAN_METADATA_JSON}"
    )


# ============================================================================
# FINAL METADATA VALIDATION
# ============================================================================

def validate_saved_metadata() -> None:
    """
    Re-open the generated metadata file and validate it again.

    This protects against accidentally saving malformed,
    incomplete, or inconsistent metadata.
    """

    print(
        "\nValidating saved metadata..."
    )

    if not CLEAN_METADATA_JSON.exists():
        raise RuntimeError(
            "Clean metadata file was not created."
        )

    try:
        with open(
            CLEAN_METADATA_JSON,
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Saved COCO metadata is not "
            f"valid JSON:\n"
            f"{CLEAN_METADATA_JSON}\n"
            f"Error: {exc}"
        ) from exc

    if not isinstance(
        data,
        dict,
    ):
        raise RuntimeError(
            "Saved COCO metadata is not "
            "a JSON object."
        )

    if len(data) != (
        EXPECTED_TEST_IMAGES
    ):
        raise RuntimeError(
            "Saved metadata contains "
            f"{len(data):,} images; "
            f"expected "
            f"{EXPECTED_TEST_IMAGES:,}."
        )

    seen_ids: set[int] = set()

    total_captions = 0

    for key, item in data.items():

        if not isinstance(
            item,
            dict,
        ):
            raise RuntimeError(
                f"Invalid metadata entry "
                f"for key {key}."
            )

        required_fields = {
            "cocoid",
            "filename",
            "filepath",
            "sentids",
            "captions",
        }

        missing_fields = (
            required_fields
            - item.keys()
        )

        if missing_fields:
            raise RuntimeError(
                f"COCO ID {key} is missing "
                f"fields: "
                f"{sorted(missing_fields)}"
            )

        coco_id = int(
            item["cocoid"]
        )

        if coco_id in seen_ids:
            raise RuntimeError(
                "Duplicate COCO ID in saved "
                f"metadata: {coco_id}"
            )

        seen_ids.add(
            coco_id
        )

        sentids = item[
            "sentids"
        ]

        captions = item[
            "captions"
        ]

        if not isinstance(
            sentids,
            list,
        ):
            raise RuntimeError(
                f"COCO ID {coco_id} "
                "has invalid sentids."
            )

        if len(sentids) != (
            EXPECTED_CAPTIONS_PER_IMAGE
        ):
            raise RuntimeError(
                f"COCO ID {coco_id} "
                f"has {len(sentids)} sentids; "
                f"expected "
                f"{EXPECTED_CAPTIONS_PER_IMAGE}."
            )

        if not isinstance(
            captions,
            list,
        ):
            raise RuntimeError(
                f"COCO ID {coco_id} "
                "has invalid captions."
            )

        if len(captions) != (
            EXPECTED_CAPTIONS_PER_IMAGE
        ):
            raise RuntimeError(
                f"COCO ID {coco_id} "
                f"has {len(captions)} captions; "
                f"expected "
                f"{EXPECTED_CAPTIONS_PER_IMAGE}."
            )

        for caption in captions:
            if not isinstance(
                caption,
                str,
            ):
                raise RuntimeError(
                    f"COCO ID {coco_id} "
                    "contains a non-string caption."
                )

            if not caption.strip():
                raise RuntimeError(
                    f"COCO ID {coco_id} "
                    "contains an empty caption."
                )

        total_captions += len(
            captions
        )

    expected_total = (
        EXPECTED_TEST_IMAGES
        * EXPECTED_CAPTIONS_PER_IMAGE
    )

    if total_captions != expected_total:
        raise RuntimeError(
            "Saved metadata contains "
            f"{total_captions:,} captions; "
            f"expected "
            f"{expected_total:,}."
        )

    print(
        f"Saved images: "
        f"{len(data):,}"
    )

    print(
        f"Saved captions: "
        f"{total_captions:,}"
    )

    print(
        "Saved metadata validated ✅"
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print("=" * 70)
    print(
        "COCO KARPATHY DATASET INTEGRITY VERIFICATION"
    )
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load exact 5,000 Karpathy test images.
    # ------------------------------------------------------------------

    test_metadata = (
        load_karpathy_test()
    )

    test_ids = set(
        test_metadata.keys()
    )

    # ------------------------------------------------------------------
    # 2. Resolve exactly five captions per image.
    #
    #    Karpathy sentids[:5]
    #         ->
    #    official COCO caption annotation IDs
    #         ->
    #    caption text
    # ------------------------------------------------------------------

    captions_by_image = (
        load_coco_captions(
            test_metadata
        )
    )

    # ------------------------------------------------------------------
    # 3. Verify WDS contains exactly the same image IDs.
    # ------------------------------------------------------------------

    verify_wds(
        test_ids
    )

    # ------------------------------------------------------------------
    # 4. Save clean evaluation metadata.
    # ------------------------------------------------------------------

    save_clean_metadata(
        test_metadata,
        captions_by_image,
    )

    # ------------------------------------------------------------------
    # 5. Re-open and validate what we just saved.
    # ------------------------------------------------------------------

    validate_saved_metadata()

    # ------------------------------------------------------------------
    # 6. Final summary.
    # ------------------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "COCO VERIFICATION PASSED ✅"
    )

    print(
        "=" * 70
    )

    print(
        "\nFinal evaluation sources:"
    )

    print(
        "  Test membership : "
        "Karpathy COCO test split"
    )

    print(
        "  Caption IDs     : "
        "Karpathy sentids[:5]"
    )

    print(
        "  Caption text    : "
        "COCO 2014 caption annotations"
    )

    print(
        "  Images          : "
        "COCO Karpathy WDS stream"
    )

    print(
        "  Test images     : "
        f"{EXPECTED_TEST_IMAGES:,}"
    )

    print(
        "  Captions        : "
        f"{EXPECTED_TEST_IMAGES * EXPECTED_CAPTIONS_PER_IMAGE:,}"
    )

    print(
        "  Captions/image  : "
        f"{EXPECTED_CAPTIONS_PER_IMAGE}"
    )

    print(
        "\nWDS captions are NOT used."
    )

    print(
        "Only WDS image bytes and COCO IDs "
        "are used for evaluation."
    )

    # ------------------------------------------------------------------
    # Print one example.
    # ------------------------------------------------------------------

    first_id = sorted(
        test_ids
    )[0]

    print(
        "\nExample test item:"
    )

    print(
        f"  COCO ID: "
        f"{first_id}"
    )

    print(
        f"  Filename: "
        f"{test_metadata[first_id]['filename']}"
    )

    print(
        f"  Sentence IDs: "
        f"{test_metadata[first_id]['sentids']}"
    )

    print(
        "  Captions:"
    )

    for caption in captions_by_image[
        first_id
    ]:

        print(
            f"    - {caption}"
        )

    print(
        "\nMetadata file:"
    )

    print(
        f"  {CLEAN_METADATA_JSON}"
    )


if __name__ == "__main__":
    main()