# Intent Classification Report

## Project: BERT Intent Classification App

## 1. Dataset
Banking77 — a dataset of banking customer-support messages labeled with
77 distinct intents (e.g. "lost_or_stolen_card", "card_payment_fee_charged").

## 2. Models Compared

### TF-IDF + Logistic Regression (Baseline)
- Accuracy: 0.8562
- Weighted F1: 0.8556
- Avg inference latency: 0.001 ms/example

### Fine-tuned BERT-base
- Accuracy: 0.9279220779220779
- Weighted F1: 0.9278758605568839

## 3. Latency Comparison
- TF-IDF avg latency: 0.480 ms
- BERT avg latency: 62.177 ms
- BERT is 129.5x slower than TF-IDF per prediction

## 4. Conclusion
The fine-tuned BERT model generally achieves higher accuracy on nuanced or
paraphrased intent queries than the TF-IDF baseline, since it understands
context and word meaning rather than just keyword overlap. In exchange, it
is slower to run and needs more memory/storage. The TF-IDF + Logistic
Regression baseline is a good lightweight option when speed matters more
than squeezing out the last bit of accuracy.
