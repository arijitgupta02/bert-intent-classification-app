"""
STEP 4: Compare inference latency between the TF-IDF baseline and the
fine-tuned BERT model.

Run this AFTER both baseline_tfidf.py and train_bert.py have finished:
    python evaluate_latency.py

It creates:
    models/latency_comparison.json
"""

import time
import json
import joblib
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def measure_bert_latency(model_dir="models/bert_intent_model",
                          data_dir="data", n_samples=100):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    test_df = pd.read_csv(f"{data_dir}/test.csv")
    n_samples = min(n_samples, len(test_df))
    sample = test_df.sample(n=n_samples, random_state=42)

    times = []
    with torch.no_grad():
        for text in sample["text"]:
            start = time.time()
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
            model(**inputs)
            times.append((time.time() - start) * 1000)

    avg_latency = sum(times) / len(times)
    print(f"BERT average latency: {avg_latency:.3f} ms/example")
    return avg_latency


def measure_tfidf_latency(baseline_dir="models/tfidf_baseline",
                           data_dir="data", n_samples=100):
    vectorizer = joblib.load(f"{baseline_dir}/vectorizer.pkl")
    clf = joblib.load(f"{baseline_dir}/classifier.pkl")

    test_df = pd.read_csv(f"{data_dir}/test.csv")
    n_samples = min(n_samples, len(test_df))
    sample = test_df.sample(n=n_samples, random_state=42)

    times = []
    for text in sample["text"]:
        start = time.time()
        X = vectorizer.transform([text])
        clf.predict(X)
        times.append((time.time() - start) * 1000)

    avg_latency = sum(times) / len(times)
    print(f"TF-IDF average latency: {avg_latency:.3f} ms/example")
    return avg_latency


def compare_latency():
    tfidf_latency = measure_tfidf_latency()
    bert_latency = measure_bert_latency()

    comparison = {
        "tfidf_avg_latency_ms": tfidf_latency,
        "bert_avg_latency_ms": bert_latency,
        "bert_slower_by_factor": bert_latency / tfidf_latency,
    }

    with open("models/latency_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print("\nLatency comparison:")
    print(json.dumps(comparison, indent=2))
    return comparison


if __name__ == "__main__":
    compare_latency()
