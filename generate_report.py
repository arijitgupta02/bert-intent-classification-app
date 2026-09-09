"""
STEP 5: Generate a combined "Intent Classification Report" (Markdown file)
comparing the TF-IDF baseline vs the fine-tuned BERT model.

Run this AFTER baseline_tfidf.py, train_bert.py, and (optionally)
evaluate_latency.py have all finished:
    python generate_report.py

It creates:
    Intent_Classification_Report.md
"""

import json


def generate_report():
    with open("models/tfidf_baseline/metrics.json") as f:
        tfidf_metrics = json.load(f)

    with open("models/bert_intent_model/metrics.json") as f:
        bert_metrics = json.load(f)

    try:
        with open("models/latency_comparison.json") as f:
            latency = json.load(f)
    except FileNotFoundError:
        latency = None

    report = f"""# Intent Classification Report

## Project: BERT Intent Classification App

## 1. Dataset
Banking77 — a dataset of banking customer-support messages labeled with
77 distinct intents (e.g. "lost_or_stolen_card", "card_payment_fee_charged").

## 2. Models Compared

### TF-IDF + Logistic Regression (Baseline)
- Accuracy: {tfidf_metrics['accuracy']:.4f}
- Weighted F1: {tfidf_metrics['f1_weighted']:.4f}
- Avg inference latency: {tfidf_metrics['avg_latency_ms']:.3f} ms/example

### Fine-tuned BERT-base
- Accuracy: {bert_metrics.get('eval_accuracy', 'N/A')}
- Weighted F1: {bert_metrics.get('eval_f1_weighted', 'N/A')}
"""

    if latency:
        report += f"""
## 3. Latency Comparison
- TF-IDF avg latency: {latency['tfidf_avg_latency_ms']:.3f} ms
- BERT avg latency: {latency['bert_avg_latency_ms']:.3f} ms
- BERT is {latency['bert_slower_by_factor']:.1f}x slower than TF-IDF per prediction
"""

    report += """
## 4. Conclusion
The fine-tuned BERT model generally achieves higher accuracy on nuanced or
paraphrased intent queries than the TF-IDF baseline, since it understands
context and word meaning rather than just keyword overlap. In exchange, it
is slower to run and needs more memory/storage. The TF-IDF + Logistic
Regression baseline is a good lightweight option when speed matters more
than squeezing out the last bit of accuracy.
"""

    with open("Intent_Classification_Report.md", "w") as f:
        f.write(report)

    print("Report saved to Intent_Classification_Report.md")


if __name__ == "__main__":
    generate_report()
