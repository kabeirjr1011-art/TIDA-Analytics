"""Train a news-category classifier.
Usage: python train.py path/to/india-news-headlines.csv
"""
import re, sys, json
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

CSV = sys.argv[1] if len(sys.argv) > 1 else "india-news-headlines.csv"
PER_CLASS = 15000
SEED = 42

# 1. Load + map the 1,024 raw categories to 7 broad news topics
df = pd.read_csv(CSV)
top = df["headline_category"].str.split(".").str[0]
MAP = {"india": "National", "business": "Business", "entertainment": "Entertainment",
       "sports": "Sports", "world": "World", "tech": "Technology", "life-style": "Lifestyle"}
df["label"] = top.map(MAP)
df = df.dropna(subset=["label"])[["headline_text", "label"]].drop_duplicates()

# 2. Balance classes
df = pd.concat([g.sample(min(len(g), PER_CLASS), random_state=SEED)
                for _, g in df.groupby("label")], ignore_index=True)

# 3. Text preprocessing (lowercase, strip URLs/non-letters, collapse spaces)
def clean(t: str) -> str:
    t = t.lower()
    t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

df["clean"] = df["headline_text"].astype(str).map(clean)
df = df[df["clean"].str.len() > 2]

X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["label"], test_size=0.2, stratify=df["label"], random_state=SEED)

# 4. Feature representation + model comparison
def make(model):
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                  min_df=2, sublinear_tf=True)),
        ("clf", model)])

candidates = {
    "Naive Bayes": make(MultinomialNB(alpha=0.3)),
    "Logistic Regression": make(LogisticRegression(max_iter=1000, C=5)),
    "Linear SVM": make(CalibratedClassifierCV(LinearSVC(C=0.5), cv=3)),  # calibrated => real probabilities
}
results, fitted = {}, {}
for name, pipe in candidates.items():
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    results[name] = {"accuracy": accuracy_score(y_test, pred),
                     "macro_f1": f1_score(y_test, pred, average="macro")}
    fitted[name] = pipe
    print(f"{name:22s} acc={results[name]['accuracy']:.4f}  macroF1={results[name]['macro_f1']:.4f}")

# 5. Pick best, report in detail
best_name = max(results, key=lambda k: results[k]["macro_f1"])
best = fitted[best_name]
pred = best.predict(X_test)
print("\nBest model:", best_name)
print(classification_report(y_test, pred, digits=3))
labels = sorted(df["label"].unique())
print(pd.DataFrame(confusion_matrix(y_test, pred, labels=labels), index=labels, columns=labels))

# 6. Save artefacts for Streamlit
joblib.dump(best, "model.joblib", compress=3)
json.dump({"best_model": best_name, "results": results, "labels": labels,
           "n_train": len(X_train), "n_test": len(X_test)}, open("metrics.json", "w"), indent=2)
print("\nSaved model.joblib + metrics.json")
