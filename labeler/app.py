#!/usr/bin/env python3
"""Gothic Theme Labeling Tool - FastAPI Server"""

import json
import random
from datetime import datetime
from pathlib import Path

import yaml
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title="Gothic Theme Labeler")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
LABELS_FILE = DATA_DIR / "labels.json"
PASSAGES_FILE = DATA_DIR / "passages.json"
TAXONOMY_FILE = BASE_DIR.parent / "theme_taxonomy.yaml"
TARGET_PASSAGES_FILE = DATA_DIR / "target_passages.json"
SUMMARIES_FILE = DATA_DIR / "summaries.json"

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# Target counts per split
TARGETS = {"train": 400, "val": 75, "test": 75}
SKIP_BUDGET_PERCENT = 0.10  # 10% of total target allowed as skips


def load_passages() -> list[dict]:
    """Load all passages from passages.json."""
    if not PASSAGES_FILE.exists():
        return []
    with open(PASSAGES_FILE) as f:
        return json.load(f)


def load_labels() -> dict:
    """Load labeling progress from labels.json."""
    if not LABELS_FILE.exists():
        return {"labeled": [], "skipped": [], "last_updated": None}
    with open(LABELS_FILE) as f:
        return json.load(f)


def save_labels(labels: dict):
    """Save labeling progress to labels.json."""
    labels["last_updated"] = datetime.utcnow().isoformat() + "Z"
    with open(LABELS_FILE, "w") as f:
        json.dump(labels, f, indent=2)


def load_themes() -> list[dict]:
    """Load theme definitions from taxonomy file."""
    with open(TAXONOMY_FILE) as f:
        data = yaml.safe_load(f)
    return data.get("themes", [])


def load_target_passages() -> list[str]:
    """Load target passage IDs."""
    if not TARGET_PASSAGES_FILE.exists():
        return []
    with open(TARGET_PASSAGES_FILE) as f:
        return json.load(f)


def load_summaries() -> dict:
    """Load passage summaries."""
    if not SUMMARIES_FILE.exists():
        return {}
    with open(SUMMARIES_FILE) as f:
        return json.load(f)


def get_labeled_ids(labels: dict) -> set[str]:
    """Get set of all labeled and skipped passage IDs."""
    labeled_ids = {item["id"] for item in labels.get("labeled", [])}
    skipped_ids = set(labels.get("skipped", []))
    return labeled_ids | skipped_ids


def get_progress(labels: dict, passages: list[dict]) -> dict:
    """Calculate progress stats per split."""
    labeled_by_split = {"train": 0, "val": 0, "test": 0}
    for item in labels.get("labeled", []):
        # Find the passage to get its split
        for p in passages:
            if p["id"] == item["id"]:
                split = p.get("split", "train")
                labeled_by_split[split] = labeled_by_split.get(split, 0) + 1
                break

    return {
        split: {"current": labeled_by_split.get(split, 0), "target": target}
        for split, target in TARGETS.items()
    }


def get_next_passage(passages: list[dict], labels: dict) -> dict | None:
    """Get next unlabeled passage, prioritizing by split target completion."""
    labeled_ids = get_labeled_ids(labels)
    progress = get_progress(labels, passages)
    target_ids = set(load_target_passages())

    # Determine which split to prioritize (least complete first)
    split_order = sorted(
        TARGETS.keys(),
        key=lambda s: progress[s]["current"] / progress[s]["target"] if progress[s]["target"] > 0 else 1
    )

    for split in split_order:
        if progress[split]["current"] >= progress[split]["target"]:
            continue

        # Get unlabeled passages from this split (only from target list)
        unlabeled = [
            p for p in passages
            if p["id"] not in labeled_ids
            and p.get("split") == split
            and (not target_ids or p["id"] in target_ids)  # Filter by target if available
        ]
        if unlabeled:
            return random.choice(unlabeled)

    # All targets met or no passages left
    return None


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the labeling UI."""
    themes = load_themes()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "themes": themes
    })


@app.get("/api/next")
async def next_passage():
    """Get the next unlabeled passage."""
    passages = load_passages()
    labels = load_labels()
    summaries = load_summaries()

    passage = get_next_passage(passages, labels)
    progress = get_progress(labels, passages)
    total_skipped = len(labels.get("skipped", []))
    skip_budget = int(sum(TARGETS.values()) * SKIP_BUDGET_PERCENT)

    if passage is None:
        return {"done": True, "progress": progress, "skipped": total_skipped, "skip_budget": skip_budget}

    return {
        "done": False,
        "passage": passage,
        "progress": progress,
        "current_split": passage.get("split", "train"),
        "skipped": total_skipped,
        "skip_budget": skip_budget,
        "summary": summaries.get(passage["id"])
    }


@app.post("/api/label")
async def label_passage(request: Request):
    """Submit a label for a passage."""
    data = await request.json()
    passage_id = data.get("id")
    themes = data.get("themes", [])
    split = data.get("split", "train")

    if not passage_id:
        return {"error": "Missing passage ID"}

    labels = load_labels()

    # Remove from skipped if it was there
    if passage_id in labels["skipped"]:
        labels["skipped"].remove(passage_id)

    # Remove existing label for this ID if any (for undo/redo)
    labels["labeled"] = [l for l in labels["labeled"] if l["id"] != passage_id]

    # Add new label
    labels["labeled"].append({
        "id": passage_id,
        "split": split,
        "themes": themes,
        "labeled_at": datetime.utcnow().isoformat() + "Z"
    })

    save_labels(labels)
    return {"success": True}


@app.post("/api/skip")
async def skip_passage(request: Request):
    """Mark a passage as skipped."""
    data = await request.json()
    passage_id = data.get("id")

    if not passage_id:
        return {"error": "Missing passage ID"}

    labels = load_labels()

    if passage_id not in labels["skipped"]:
        labels["skipped"].append(passage_id)

    save_labels(labels)
    return {"success": True}


@app.post("/api/undo")
async def undo_label():
    """Undo the last label and return that passage."""
    labels = load_labels()
    summaries = load_summaries()

    if not labels["labeled"]:
        return {"error": "Nothing to undo"}

    # Remove last labeled item
    last = labels["labeled"].pop()
    save_labels(labels)

    # Find and return that passage
    passages = load_passages()
    passage = next((p for p in passages if p["id"] == last["id"]), None)
    progress = get_progress(labels, passages)
    total_skipped = len(labels.get("skipped", []))
    skip_budget = int(sum(TARGETS.values()) * SKIP_BUDGET_PERCENT)

    return {
        "success": True,
        "passage": passage,
        "previous_themes": last["themes"],
        "progress": progress,
        "skipped": total_skipped,
        "skip_budget": skip_budget,
        "summary": summaries.get(passage["id"]) if passage else None
    }


@app.get("/api/review-skipped")
async def review_skipped():
    """Get a random skipped passage for review."""
    passages = load_passages()
    labels = load_labels()
    summaries = load_summaries()

    skipped_ids = labels.get("skipped", [])
    if not skipped_ids:
        return {"done": True, "message": "No skipped passages to review"}

    # Find the passage
    passage = next((p for p in passages if p["id"] in skipped_ids), None)
    if not passage:
        return {"done": True, "message": "No skipped passages found"}

    progress = get_progress(labels, passages)
    total_skipped = len(skipped_ids)
    skip_budget = int(sum(TARGETS.values()) * SKIP_BUDGET_PERCENT)

    return {
        "done": False,
        "passage": passage,
        "progress": progress,
        "current_split": passage.get("split", "train"),
        "skipped": total_skipped,
        "skip_budget": skip_budget,
        "reviewing_skipped": True,
        "summary": summaries.get(passage["id"])
    }


@app.get("/api/progress")
async def get_progress_stats():
    """Get current progress statistics."""
    passages = load_passages()
    labels = load_labels()
    progress = get_progress(labels, passages)

    total_labeled = len(labels.get("labeled", []))
    total_skipped = len(labels.get("skipped", []))
    total_target = sum(TARGETS.values())
    skip_budget = int(total_target * SKIP_BUDGET_PERCENT)

    return {
        "by_split": progress,
        "total_labeled": total_labeled,
        "total_skipped": total_skipped,
        "total_target": total_target,
        "skip_budget": skip_budget,
        "over_skip_budget": total_skipped > skip_budget
    }
