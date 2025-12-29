import json
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

with open("data/train.json", "r") as f:
    train_data = json.load(f)


texts = [p["text"] for p in train_data]
labels = [p["themes"] for p in train_data]


print(f"Loaded {len(texts)} training passages")


vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(texts)


print(f"TF-IDF matrix shape: {X_train.shape}")


mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(labels)


print(f"Labels matrix shape: {y_train.shape}")
print(f"Themes: {mlb.classes_}")


classifier = OneVsRestClassifier(LogisticRegression(max_iter=1000))
classifier.fit(X_train, y_train)


print("Classifier trained")


Path("models").mkdir(exist_ok=True)


joblib.dump(vectorizer, "models/vectorizer.pkl")
joblib.dump(mlb, "models/mlb.pkl")
joblib.dump(classifier, "models/classifier.pkl")


print("Saved to models/")
