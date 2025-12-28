# Gothic Thematic Analysis Engine

An NLP system for analyzing Gothic literature for thematic content and generating new passages with controllable themes.

## Status

**Phase 1 (Foundation):** Complete
**Phase 2 (Analyzer):** In progress - manual labeling

## Data

- **Corpus:** 38 Gothic novels (1764-1911)
- **Passages:** ~11,700 chunks (100-500 tokens each)
- **Splits:** Train 8,906 / Val 1,619 / Test 1,182

## Theme Taxonomy

12 themes derived from corpus sampling:

| Theme | Description |
|-------|-------------|
| supernatural | Vampires, ghosts, apparitions, the undead |
| imprisonment | Physical captivity, confinement, escape attempts |
| secrecy | Hidden information, concealed identities |
| love | Romantic attachment, courtship, passion |
| death | Dying, murder, grief, confronting mortality |
| persecution | Characters threatened, pursued, or victimized |
| family | Parent-child bonds, inheritance, lineage |
| melancholy | Sadness, regret, emotional suffering |
| religion | Clergy, churches, monasteries, spiritual matters |
| terror | Intense fear, horror, dread |
| isolation | Loneliness, solitude, separation |
| setting | Atmospheric environments—storms, wilderness, Gothic architecture |

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

- `1-9`, `0`, `-`, `=` - Toggle themes
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
