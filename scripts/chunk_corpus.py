#!/usr/bin/env python3
"""
Chunk Gothic novels into passages for classifier training.
Uses paragraph-based chunking with token limits.
"""

import os
import json
import re
from pathlib import Path

# Configuration - paths relative to project root
PROJECT_ROOT = Path(__file__).parent.parent
SOURCE_DIR = PROJECT_ROOT / "data" / "gothic_central_novels"
OUTPUT_DIR = PROJECT_ROOT / "data" / "chunks"
MIN_TOKENS = 100
MAX_TOKENS = 500
TARGET_TOKENS = 300


def count_tokens(text: str) -> int:
    """Simple whitespace tokenizer for counting."""
    return len(text.split())


def split_into_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs based on blank lines."""
    # Normalize line endings and split on blank lines
    text = text.replace('\r\n', '\n')
    paragraphs = re.split(r'\n\s*\n', text)
    # Clean up and filter empty paragraphs
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    return paragraphs


def chunk_text(text: str, min_tokens: int = MIN_TOKENS,
               max_tokens: int = MAX_TOKENS) -> list[str]:
    """
    Chunk text into passages of appropriate length.
    Combines short paragraphs, splits long ones.
    """
    paragraphs = split_into_paragraphs(text)
    chunks = []
    current_chunk = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)

        # If single paragraph exceeds max, split it by sentences
        if para_tokens > max_tokens:
            # Flush current chunk first
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_tokens = 0

            # Split long paragraph by sentences
            sentences = re.split(r'(?<=[.!?])\s+', para)
            sent_chunk = []
            sent_tokens = 0

            for sent in sentences:
                sent_token_count = count_tokens(sent)
                if sent_tokens + sent_token_count > max_tokens and sent_chunk:
                    chunks.append(' '.join(sent_chunk))
                    sent_chunk = [sent]
                    sent_tokens = sent_token_count
                else:
                    sent_chunk.append(sent)
                    sent_tokens += sent_token_count

            if sent_chunk:
                # Add remaining sentences to current_chunk for potential merging
                current_chunk = [' '.join(sent_chunk)]
                current_tokens = sent_tokens

        # If adding this paragraph would exceed max, flush current chunk
        elif current_tokens + para_tokens > max_tokens and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
            current_tokens = para_tokens

        # Otherwise, add to current chunk
        else:
            current_chunk.append(para)
            current_tokens += para_tokens

    # Flush remaining content
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))

    # Filter out chunks that are too short (unless it's the only content)
    filtered_chunks = [c for c in chunks if count_tokens(c) >= min_tokens]

    # If all chunks were filtered out, return original chunks
    if not filtered_chunks and chunks:
        return chunks

    return filtered_chunks


def process_novel(filepath: Path) -> dict:
    """Process a single novel file and return chunked data."""
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()

    chunks = chunk_text(text)

    # Extract metadata from filename: Author_Title_Source.txt
    filename = filepath.stem
    parts = filename.split('_')
    author = parts[0] if parts else "Unknown"
    title = '_'.join(parts[1:-1]) if len(parts) > 2 else parts[1] if len(parts) > 1 else "Unknown"
    source = parts[-1] if len(parts) > 1 else "Unknown"

    return {
        "filename": filepath.name,
        "author": author,
        "title": title,
        "source": source,
        "total_chunks": len(chunks),
        "chunks": [
            {
                "chunk_id": f"{filename}_{i:04d}",
                "text": chunk,
                "token_count": count_tokens(chunk)
            }
            for i, chunk in enumerate(chunks)
        ]
    }


def main():
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Get all novel files
    novel_files = sorted(SOURCE_DIR.glob("*.txt"))

    print(f"Processing {len(novel_files)} novels...")

    all_chunks = []
    summary = []

    for filepath in novel_files:
        print(f"  Processing: {filepath.name}")
        novel_data = process_novel(filepath)

        # Save individual novel chunks
        output_file = OUTPUT_DIR / f"{filepath.stem}_chunks.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(novel_data, f, indent=2, ensure_ascii=False)

        # Collect all chunks for combined file
        for chunk in novel_data["chunks"]:
            all_chunks.append({
                "chunk_id": chunk["chunk_id"],
                "author": novel_data["author"],
                "title": novel_data["title"],
                "text": chunk["text"],
                "token_count": chunk["token_count"]
            })

        summary.append({
            "filename": novel_data["filename"],
            "author": novel_data["author"],
            "title": novel_data["title"],
            "total_chunks": novel_data["total_chunks"],
            "avg_tokens": sum(c["token_count"] for c in novel_data["chunks"]) // max(len(novel_data["chunks"]), 1)
        })

    # Save combined chunks file
    combined_file = OUTPUT_DIR / "all_chunks.json"
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    # Save summary
    summary_file = OUTPUT_DIR / "summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            "total_novels": len(novel_files),
            "total_chunks": len(all_chunks),
            "novels": summary
        }, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\nComplete!")
    print(f"  Total novels: {len(novel_files)}")
    print(f"  Total chunks: {len(all_chunks)}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Combined file: {combined_file}")


if __name__ == "__main__":
    main()
