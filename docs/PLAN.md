# Gothic Theme Engine - Project Plan

## Overview

A two-part NLP system that analyzes Gothic literature for thematic content and generates new passages with controllable themes.

---

## Phase 1: Foundation

### 1.1 Project Setup

- [x] Initialize repository
- [x] Set up Python environment (pyproject.toml)
- [x] Define folder structure
- [x] Add .gitignore

### 1.2 Data Preparation

- [x] Inventory the 38 novels in `gothic_central_novels/`
- [x] Write script to chunk novels into passages (paragraph or fixed token length)
- [x] Create train/validation/test splits

### 1.3 Theme Taxonomy

- [x] Use Claude CLI to sample corpus and propose themes
- [x] Review and refine taxonomy (8-12 themes)
- [x] Save as `theme_taxonomy.yaml`

---

## Phase 2: Analyzer Module

### 2.1 Manual Labeling

- [ ] Build simple labeling interface (CLI or basic web UI)
- [ ] Label 300-500 passages with themes
- [ ] Save labeled data as JSON/CSV

### 2.2 Feature Engineering

- [ ] TF-IDF baseline
- [ ] Explore embeddings (sentence-transformers)
- [ ] Theme-specific keyword lists as features (optional)

### 2.3 Classifier Training

- [ ] Multi-label classifier (scikit-learn or PyTorch)
- [ ] Evaluate on held-out test set
- [ ] Iterate on features/model until accuracy is acceptable
- [ ] Save trained model

### 2.4 Auto-Tagging

- [ ] Run classifier on full corpus
- [ ] Output: passages with predicted theme labels
- [ ] Spot-check quality

---

## Phase 3: Generator Module

### 3.1 Data Formatting

- [ ] Format tagged corpus for fine-tuning: `[theme1][theme2] passage text...`
- [ ] Create train/validation splits

### 3.2 Model Fine-Tuning

- [ ] Select base model (GPT-2 or DistilGPT-2)
- [ ] Fine-tune on tagged Gothic corpus
- [ ] Experiment with generation parameters (temperature, top-p)

### 3.3 Conditional Generation

- [ ] Input: list of themes
- [ ] Output: generated passage in Gothic style with those themes
- [ ] Test and iterate on output quality

---

## Phase 4: Integration

### 4.1 CLI Interface

- [ ] `analyze` command: text → themes
- [ ] `generate` command: themes → text
- [ ] `transform` command: text → analyze → modify themes → regenerate

### 4.2 API (Optional)

- [ ] FastAPI wrapper
- [ ] Endpoints for analyze/generate

### 4.3 Demo

- [ ] Sample outputs for README
- [ ] Interactive demo (Gradio or Streamlit)

---

## Phase 5: Polish

### 5.1 Documentation

- [ ] README with project overview, setup, usage
- [ ] Explain theme taxonomy decisions
- [ ] Document model performance metrics

### 5.2 Testing

- [ ] Unit tests for parser, classifier, generator
- [ ] Integration tests for full pipeline

### 5.3 Portfolio Write-Up

- [ ] Case study / blog post explaining approach
- [ ] Visualizations (theme distributions, example outputs)

---

## Project Structure

```
gothic-theme-engine/
├── README.md
├── pyproject.toml
├── theme_taxonomy.yaml
├── data/
│   ├── gothic_central_novels/
│   ├── passages/
│   ├── labeled/
│   └── tagged_corpus/
├── src/
│   ├── __init__.py
│   ├── chunker.py
│   ├── labeler.py
│   ├── analyzer/
│   │   ├── features.py
│   │   └── classifier.py
│   └── generator/
│       ├── fine_tune.py
│       └── generate.py
├── models/
│   ├── analyzer.pkl
│   └── generator/
├── scripts/
│   ├── chunk_corpus.py
│   ├── train_analyzer.py
│   ├── tag_corpus.py
│   └── train_generator.py
├── cli.py
└── tests/
```

---

## Open Questions

- [x] Passage length: paragraphs vs. fixed token count?

  - Hybrid approach - paragraph-based with token constraints:
    1. Split on blank lines (paragraphs)
    2. Combine short paragraphs until reaching ~100-500 tokens
    3. Split overly long paragraphs by sentence if >500 tokens
  - The result is mostly natural paragraph boundaries, but normalized to a consistent size range. Average ended up ~420 tokens per chunk.

- [ ] Labeling tool: CLI, spreadsheet, or simple web UI?
- [ ] Base model: GPT-2 vs. DistilGPT-2 vs. something else?
- [ ] Multi-label threshold: what confidence = "has theme"?

---

## Timeline

| Phase       | Estimated Time                        |
| ----------- | ------------------------------------- |
| Foundation  | 1-2 days                              |
| Analyzer    | 3-5 days (labeling is the bottleneck) |
| Generator   | 2-3 days                              |
| Integration | 1-2 days                              |
| Polish      | 1-2 days                              |
| **Total**   | **8-14 days**                         |
