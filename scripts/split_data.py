#!/usr/bin/env python3
"""
Split chunked novels into train/validation/test sets.
Splits by novel (not by chunk) to prevent data leakage.
"""

import json
import random
from pathlib import Path

# Configuration - paths relative to project root
PROJECT_ROOT = Path(__file__).parent.parent
CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks"
OUTPUT_DIR = PROJECT_ROOT / "data" / "passages"
RANDOM_SEED = 42

# Target proportions
TRAIN_RATIO = 0.75
VAL_RATIO = 0.125
TEST_RATIO = 0.125


def load_all_chunks():
    """Load all chunks and group by novel."""
    with open(CHUNKS_DIR / "all_chunks.json", 'r', encoding='utf-8') as f:
        all_chunks = json.load(f)

    # Group chunks by novel (using author_title as key)
    novels = {}
    for chunk in all_chunks:
        key = f"{chunk['author']}_{chunk['title']}"
        if key not in novels:
            novels[key] = {
                "author": chunk["author"],
                "title": chunk["title"],
                "chunks": []
            }
        novels[key]["chunks"].append(chunk)

    return novels


def split_novels(novels: dict, seed: int = RANDOM_SEED):
    """Randomly assign novels to train/val/test splits."""
    random.seed(seed)

    novel_keys = list(novels.keys())
    random.shuffle(novel_keys)

    n = len(novel_keys)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    train_novels = novel_keys[:train_end]
    val_novels = novel_keys[train_end:val_end]
    test_novels = novel_keys[val_end:]

    return train_novels, val_novels, test_novels


def collect_chunks(novels: dict, novel_keys: list):
    """Collect all chunks for given novel keys."""
    chunks = []
    for key in novel_keys:
        chunks.extend(novels[key]["chunks"])
    return chunks


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Loading chunks...")
    novels = load_all_chunks()
    print(f"  Found {len(novels)} novels")

    print("\nSplitting by novel...")
    train_novels, val_novels, test_novels = split_novels(novels)

    # Collect chunks for each split
    train_chunks = collect_chunks(novels, train_novels)
    val_chunks = collect_chunks(novels, val_novels)
    test_chunks = collect_chunks(novels, test_novels)

    # Save splits
    splits = {
        "train": train_chunks,
        "val": val_chunks,
        "test": test_chunks
    }

    for split_name, chunks in splits.items():
        output_file = OUTPUT_DIR / f"{split_name}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        print(f"  Saved {output_file}: {len(chunks)} chunks")

    # Save split metadata
    metadata = {
        "random_seed": RANDOM_SEED,
        "ratios": {
            "train": TRAIN_RATIO,
            "val": VAL_RATIO,
            "test": TEST_RATIO
        },
        "splits": {
            "train": {
                "novels": sorted(train_novels),
                "novel_count": len(train_novels),
                "chunk_count": len(train_chunks)
            },
            "val": {
                "novels": sorted(val_novels),
                "novel_count": len(val_novels),
                "chunk_count": len(val_chunks)
            },
            "test": {
                "novels": sorted(test_novels),
                "novel_count": len(test_novels),
                "chunk_count": len(test_chunks)
            }
        },
        "totals": {
            "novels": len(novels),
            "chunks": len(train_chunks) + len(val_chunks) + len(test_chunks)
        }
    }

    metadata_file = OUTPUT_DIR / "split_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Print summary
    total_chunks = len(train_chunks) + len(val_chunks) + len(test_chunks)
    print(f"\n{'='*50}")
    print("SPLIT SUMMARY")
    print(f"{'='*50}")
    print(f"{'Split':<10} {'Novels':>8} {'Chunks':>10} {'Percent':>10}")
    print(f"{'-'*50}")
    print(f"{'Train':<10} {len(train_novels):>8} {len(train_chunks):>10} {100*len(train_chunks)/total_chunks:>9.1f}%")
    print(f"{'Val':<10} {len(val_novels):>8} {len(val_chunks):>10} {100*len(val_chunks)/total_chunks:>9.1f}%")
    print(f"{'Test':<10} {len(test_novels):>8} {len(test_chunks):>10} {100*len(test_chunks)/total_chunks:>9.1f}%")
    print(f"{'-'*50}")
    print(f"{'Total':<10} {len(novels):>8} {total_chunks:>10} {'100.0%':>10}")
    print(f"\nNovel assignments saved to: {metadata_file}")


if __name__ == "__main__":
    main()
