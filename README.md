# Gothic Thematic Analysis Engine

An NLP system for analyzing Gothic literature for thematic content and generating new passages with controllable themes.

## Status

**Phase 1 (Foundation):** Complete
**Phase 2 (Analyzer):** Complete - classifier trained, corpus auto-tagged
**Phase 3 (Generator):** In progress - fine-tuning

## Data

- **Corpus:** 38 Gothic novels (1764-1911)
- **Passages:** ~11,700 chunks (100-500 tokens each)
- **Splits:** Train 8,906 / Val 1,619 / Test 1,182

## Theme Taxonomy

10 themes derived from corpus sampling:

| Theme        | Description                                                      |
| ------------ | ---------------------------------------------------------------- |
| supernatural | Vampires, ghosts, apparitions, the undead                        |
| captivity    | Physical captivity, confinement, being chased, escape attempts   |
| secrecy      | Hidden information, concealed identities                         |
| love         | Romantic attachment, courtship, passion                          |
| harm         | Death, dying, murder, violence, blood, physical brutality        |
| villainy     | Scheming, treachery, persecution, antagonistic behavior          |
| family       | Parent-child bonds, inheritance, lineage                         |
| sorrow       | Sadness, despair, loneliness, isolation                          |
| anguish      | Terror, horror, dread, guilt, insanity, psychological torment    |
| setting      | Atmospheric environments—storms, wilderness, Gothic architecture |

## Labeling Tool

Web UI for manually tagging passages with themes.

### Setup

```bash
cd gothic-thematic-analysis-engine
python -m venv venv
source venv/bin/activate
pip install -e ".[labeler]"
```

### Run

```bash
uvicorn labeler.app:app --reload
# Open http://localhost:8000
```

### Keyboard Shortcuts

- `1-9`, `0`, `-`, `=`, `Q` - Toggle themes
- `Enter` - Submit and next
- `S` - Skip passage
- `U` - Undo last label

## Project Structure

```
gothic-thematic-analysis-engine/
├── data/
│   ├── gothic_central_novels/   # Source texts (38 novels)
│   ├── chunks/                  # Per-novel chunk files
│   └── passages/                # Train/val/test splits
├── labeler/
│   ├── app.py                   # FastAPI server
│   ├── data/
│   │   ├── passages.json        # Passages for labeling
│   │   └── labels.json          # Labeling progress
│   ├── templates/
│   └── static/
├── scripts/
│   ├── chunk_corpus.py          # Novel chunking
│   └── split_data.py            # Train/val/test splitting
├── theme_taxonomy.yaml
└── pyproject.toml
```

## Design Decisions

### Training Data Size

I labeled 353 passages manually instead of the initially planned 550. At this point, the classifier achieved ~60% recall at a 0.25 threshold, with all 10 themes represented in the training data.

The classifier's purpose is auto-tagging the corpus for generator training, not production classification. With 11,000+ passages being tagged, patterns emerge from volume—minor labeling noise gets averaged out during fine-tuning.

This was a deliberate "good enough to test the pipeline" decision. If generator output quality is poor, I can return to labeling, retrain the classifier, and regenerate the tagged corpus without losing progress.

### Multi-Machine Training

Generator fine-tuning was attempted across multiple machines based on available hardware:

| Machine       | Specs                          | Use Case                                                |
| ------------- | ------------------------------ | ------------------------------------------------------- |
| M1 Mac Mini   | 8GB RAM, MPS                   | Initial attempts—ran out of GPU memory on larger models |
| ThinkPad T14s | Snapdragon X, 32GB RAM         | CPU training—worked but slow (30+ hours estimated)      |
| Desktop       | Ryzen 7 1700X, GTX 1050 Ti 4GB | Final training—GPU acceleration, ~18 hours              |

The GTX 1050 Ti is old but sufficient for Pythia-160M with small batch sizes. Training completed overnight.

---

## Multi-Machine Workflow

### Keeping Code in Sync

Use Git. Push before switching machines, pull after.

```bash
# Before leaving a machine
git add .
git commit -m "WIP: training in progress"
git push

# On the new machine
git pull
```

### Handling Large Files

Models and datasets shouldn't live in Git. Use `.gitignore`:

```
models/
data/chunks
data/generator_training.txt
data/tagged_corpus.json
*.pkl
```

Transfer large files manually:

- USB drive
- Cloud storage (Google Drive, Dropbox)

### Environment Consistency

Each machine needs a different setup:

| Machine    | PyTorch Install                                                                    |
| ---------- | ---------------------------------------------------------------------------------- |
| Mac M1     | `pip install torch` (MPS auto-detected)                                            |
| CPU-only   | `pip install torch` (defaults to CPU)                                              |
| NVIDIA GPU | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126` |

Keep a `pyproject.toml` for non-torch dependencies. Install torch manually per machine.
