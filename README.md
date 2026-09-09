# BERT Intent Classification App

A BERT-based intent classification system that fine-tunes `bert-base-uncased` on the Banking77 dataset and compares it against a TF-IDF plus Logistic Regression baseline. The project measures accuracy, weighted F1 score, and inference latency for both models, and includes a Streamlit web app for live predictions with confidence score visualization.

## Overview

The goal of this project is to classify short customer support style messages (for example, "I lost my card, what do I do?") into one of 77 banking related intents (for example, `lost_or_stolen_card`). Two approaches are built and compared:

- A classical machine learning baseline: TF-IDF vectorization plus Logistic Regression
- A fine-tuned transformer model: `bert-base-uncased`, fine-tuned on the same dataset

Both models are evaluated on accuracy, weighted F1 score, and average inference latency, so the trade-off between accuracy and speed is measured directly rather than assumed.

## Repository Structure

```
bert-intent-classification-app/
├── README.md
├── requirements.txt
├── data_prep.py
├── baseline_tfidf.py
├── train_bert.py
├── evaluate_latency.py
├── generate_report.py
├── app.py
├── Intent_Classification_Report.md
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── labels.txt
└── models/
    ├── tfidf_baseline/
    │   ├── vectorizer.pkl
    │   ├── classifier.pkl
    │   ├── metrics.json
    │   └── classification_report.txt
    ├── bert_intent_model/
    │   ├── config.json
    │   ├── tokenizer_config.json
    │   ├── tokenizer.json
    │   ├── vocab.txt
    │   ├── special_tokens_map.json
    │   ├── label_classes.json
    │   └── metrics.json
    └── latency_comparison.json
```

Note: the fine-tuned BERT model weights are not included in this repository due to file size. See the "Regenerating the BERT Model" section below.

## Dataset

Banking77: approximately 13,000 customer support messages labeled across 77 distinct banking intents. The dataset is downloaded automatically the first time `data_prep.py` is run.

## Requirements

```
transformers
datasets
torch
scikit-learn
pandas
numpy
joblib
streamlit
plotly
accelerate
```

Install with:

```bash
pip install -r requirements.txt
```

## How to Run

Run the following scripts in order from inside the project folder.

### 1. Prepare the dataset

```bash
python data_prep.py
```

Downloads the Banking77 dataset and saves it as CSV files in `data/`.

### 2. Train the TF-IDF baseline

```bash
python baseline_tfidf.py
```

Trains a TF-IDF plus Logistic Regression model and saves it, along with its metrics, to `models/tfidf_baseline/`.

### 3. Fine-tune BERT

```bash
python train_bert.py
```

Fine-tunes `bert-base-uncased` on the same dataset for three epochs and saves the trained model to `models/bert_intent_model/`. This step is compute intensive. It runs in a few minutes with a GPU, and can take considerably longer on CPU only.

### 4. Compare inference latency

```bash
python evaluate_latency.py
```

Times both models on a sample of the test set and saves the comparison to `models/latency_comparison.json`.

### 5. Generate the final report

```bash
python generate_report.py
```

Combines the results from the previous steps into `Intent_Classification_Report.md`.

### 6. Run the web app

```bash
streamlit run app.py
```

Opens a browser tab where you can type a message, choose BERT, the TF-IDF baseline, or both, and see the top predicted intents along with their confidence scores and inference latency.

## Regenerating the BERT Model

The fine-tuned BERT model weights are not included in this repository because of file size. After cloning the repository and installing the requirements, run:

```bash
python data_prep.py
python train_bert.py
```

This will recreate `models/bert_intent_model/` with the trained weights, after which `evaluate_latency.py`, `generate_report.py`, and `app.py` will all work as expected.

## Results Summary

Both models were evaluated on the same held out test set from the Banking77 dataset.

- The TF-IDF plus Logistic Regression baseline is significantly faster at inference, since it involves only a sparse matrix lookup and a linear classifier.
- The fine-tuned BERT model generally achieves higher accuracy and F1 score on nuanced or paraphrased queries, since it captures contextual meaning rather than relying on keyword overlap.
- Full numeric results, including accuracy, weighted F1 score, and average latency in milliseconds for both models, are available in `models/tfidf_baseline/metrics.json`, `models/bert_intent_model/metrics.json`, `models/latency_comparison.json`, and summarized in `Intent_Classification_Report.md`.

## Notes

- All scripts are designed to be run in order, once each, from the project root folder.
- The Streamlit app requires both the TF-IDF baseline and the fine-tuned BERT model to already exist under `models/`.
- Training was tested on both CPU and GPU. A GPU is recommended for the BERT fine-tuning step but is not required.
