import json
import joblib
from pathlib import Path


THRESHOLD = 0.25

vectorizer = joblib.load("models/vectorizer.pkl")
mlb = joblib.load("models/mlb.pkl")
classifier = joblib.load("models/classifier.pkl")


print(f"Loaded model with {len(mlb.classes_)} themes: {list(mlb.classes_)}")


with open("data/passages.json", "r") as f:
    all_passages = json.load(f)


labeled_ids = set()
for split_file in ["train.json", "validation.json", "test.json"]:
    path = Path("data") / split_file
    if path.exists():
        with open(path, "r") as f:
            for p in json.load(f):
                labeled_ids.add(p["id"])


labeled = [p for p in all_passages if p["id"] in labeled_ids]
unlabeled = [p for p in all_passages if p["id"] not in labeled_ids]

print(f"Total passages: {len(all_passages)}")
print(f"Already labeled: {len(labeled)}")
print(f"To auto-tag: {len(unlabeled)}")


texts = [p["text"] for p in unlabeled]
X = vectorizer.transform(texts)
probs = classifier.predict_proba(X)
predictions = (probs >= THRESHOLD).astype(int)


for passage, pred in zip(unlabeled, predictions):
    themes = list(mlb.inverse_transform(pred.reshape(1, -1))[0])
    passage["themes"] = themes


for split_file in ["train.json", "validation.json", "test.json"]:
    path = Path("data") / split_file
    if path.exists():
        with open(path, "r") as f:
            for p in json.load(f):
                # Find matching passage and add themes
                for orig in labeled:
                    if orig["id"] == p["id"]:
                        orig["themes"] = p["themes"]
                        break


tagged_corpus = labeled + unlabeled


print(f"\n=== Auto-tagging complete ===")
print(f"Total tagged: {len(tagged_corpus)}")


theme_counts = {theme: 0 for theme in mlb.classes_}
empty_count = 0
for p in tagged_corpus:
    if not p.get("themes"):
        empty_count += 1
    for theme in p.get("themes", []):
        if theme in theme_counts:
            theme_counts[theme] += 1


print(f"\nTheme distribution:")
for theme, count in sorted(theme_counts.items(), key=lambda x: -x[1]):
    pct = count / len(tagged_corpus) * 100
    print(f"  {theme:15} {count:5} ({pct:.1f}%)")


print(f"\nPassages with no themes: {empty_count}")


with open("data/tagged_corpus.json", "w") as f:
    json.dump(tagged_corpus, f, indent=2)


print(f"\nSaved to data/tagged_corpus.json")
