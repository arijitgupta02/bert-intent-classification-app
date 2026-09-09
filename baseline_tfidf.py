"""
STEP 2: Train a TF-IDF + Logistic Regression baseline.

This is the "classical ML" baseline we will compare our fine-tuned BERT
model against later. It is fast to train (seconds/minutes on a laptop CPU).

Run this after data_prep.py:
    python baseline_tfidf.py

It creates:
    models/tfidf_baseline/vectorizer.pkl
    models/tfidf_baseline/classifier.pkl
    models/tfidf_baseline/metrics.json
    models/tfidf_baseline/classification_report.txt
"""

import os
import json
import time
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report


def train_baseline(data_dir="data", output_dir="models/tfidf_baseline"):
    os.makedirs(output_dir, exist_ok=True)

    train_df = pd.read_csv(f"{data_dir}/train.csv")
    test_df = pd.read_csv(f"{data_dir}/test.csv")

    print("Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_df["text"])
    X_test = vectorizer.transform(test_df["text"])

    y_train = train_df["intent"]
    y_test = test_df["intent"]

    print("Training Logistic Regression classifier...")
    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(X_train, y_train)

    # Measure inference latency on the test set
    start = time.time()
    y_pred = clf.predict(X_test)
    total_time = time.time() - start
    avg_latency_ms = (total_time / len(test_df)) * 1000

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"Baseline Accuracy      : {acc:.4f}")
    print(f"Baseline Weighted F1   : {f1:.4f}")
    print(f"Avg inference latency  : {avg_latency_ms:.3f} ms/example")

    joblib.dump(vectorizer, f"{output_dir}/vectorizer.pkl")
    joblib.dump(clf, f"{output_dir}/classifier.pkl")

    metrics = {
        "model": "TF-IDF + Logistic Regression",
        "accuracy": acc,
        "f1_weighted": f1,
        "avg_latency_ms": avg_latency_ms,
    }
    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    report = classification_report(y_test, y_pred)
    with open(f"{output_dir}/classification_report.txt", "w") as f:
        f.write(report)

    print(f"Saved baseline model + metrics to {output_dir}/")
    return metrics


if __name__ == "__main__":
    train_baseline()
