import json
import joblib
from sklearn.metrics import classification_report, hamming_loss


vectorizer = joblib.load("models/vectorizer.pkl")
mlb = joblib.load("models/mlb.pkl")
classifier = joblib.load("models/classifier.pkl")


with open("data/validation.json", "r") as f:
    val_data = json.load(f)


texts = [p["text"] for p in val_data]
labels = [p["themes"] for p in val_data]


print(f"Evaluating on {len(texts)} validation passages\n")


X_val = vectorizer.transform(texts)
y_true = mlb.transform(labels)
probs = classifier.predict_proba(X_val)
threshold = 0.25  # Lower from default 0.5
y_pred = (probs >= threshold).astype(int)


print("=== Overall Metrics ===")
print(f"Hamming Loss: {hamming_loss(y_true, y_pred):.3f}")
print("(Lower is better. 0 = perfect, 1 = worst)\n")


print("=== Per-Theme Performance ===")
print(classification_report(
    y_true,
    y_pred,
    target_names=mlb.classes_,
    zero_division=0
))


print("=== Sample Predictions ===")
for i in range(min(5, len(texts))):
    true_themes = labels[i]
    pred_themes = list(mlb.inverse_transform(y_pred[i:i+1])[0])

    print(f"\nPassage {i+1}:")
    print(f"  Text: {texts[i][:100]}...")
    print(f"  Actual:    {true_themes}")
    print(f"  Predicted: {pred_themes}")
