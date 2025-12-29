#!/usr/bin/env python3
"""Export labeled passages and split into train/validation/test sets."""

import json
from collections import Counter
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
LABELER_DATA = BASE_DIR / "labeler" / "data"
OUTPUT_DIR = BASE_DIR / "data"

LABELS_FILE = LABELER_DATA / "labels.json"
PASSAGES_FILE = LABELER_DATA / "passages.json"


def main():
    # Load data
    with open(LABELS_FILE) as f:
        labels_data = json.load(f)

    with open(PASSAGES_FILE) as f:
        passages = {p["id"]: p for p in json.load(f)}

    labeled_entries = labels_data.get("labeled", [])
    skipped = labels_data.get("skipped", [])

    # Build labeled passages with full text
    labeled_passages = []
    for entry in labeled_entries:
        pid = entry["id"]
        passage = passages.get(pid)
        if not passage:
            print(f"Warning: passage {pid} not found in passages.json")
            continue

        labeled_passages.append({
            "id": pid,
            "text": passage["text"],
            "themes": entry["themes"],
            "split": passage.get("split", "train")
        })

    # Split by set
    train = [p for p in labeled_passages if p["split"] == "train"]
    validation = [p for p in labeled_passages if p["split"] == "val"]
    test = [p for p in labeled_passages if p["split"] == "test"]

    # Validate no duplicates across splits
    all_ids = [p["id"] for p in labeled_passages]
    if len(all_ids) != len(set(all_ids)):
        print("Warning: duplicate passage IDs found!")

    # Write output files
    OUTPUT_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_DIR / "labeled_passages.json", "w") as f:
        json.dump(labeled_passages, f, indent=2)

    with open(OUTPUT_DIR / "train.json", "w") as f:
        json.dump(train, f, indent=2)

    with open(OUTPUT_DIR / "validation.json", "w") as f:
        json.dump(validation, f, indent=2)

    with open(OUTPUT_DIR / "test.json", "w") as f:
        json.dump(test, f, indent=2)

    # Theme coverage in train set
    train_themes = Counter()
    for p in train:
        for theme in p["themes"]:
            train_themes[theme] += 1

    # Print summary
    print("Export complete:")
    print(f"  Total labeled: {len(labeled_passages)}")
    print(f"  Skipped: {len(skipped)}")
    print()
    print("Split distribution:")
    print(f"  Train:      {len(train):3} passages")
    print(f"  Validation: {len(validation):3} passages")
    print(f"  Test:       {len(test):3} passages")
    print()
    print("Theme coverage (train set):")
    for theme, count in train_themes.most_common():
        print(f"  {theme:15} {count:3}")


if __name__ == "__main__":
    main()
