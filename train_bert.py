"""
STEP 3: Fine-tune a BERT-base model for intent classification.

NOTE FOR BEGINNERS:
This step trains a real transformer model and is compute-heavy.
- On a free Google Colab GPU (T4): a few minutes per epoch.
- On a laptop CPU only: this can take a long time (possibly hours).

If you don't have a GPU, the easiest path is:
  1. Upload this whole project folder to Google Colab.
  2. In Colab, go to Runtime -> Change runtime type -> GPU.
  3. Run: !pip install -r requirements.txt
  4. Run: !python data_prep.py
  5. Run: !python train_bert.py
  6. Download the resulting "models/bert_intent_model" folder back to
     your computer and put it in the same place in this project
     before running the Streamlit app.

Run this after data_prep.py:
    python train_bert.py

It creates:
    models/bert_intent_model/  (the saved fine-tuned model + tokenizer)
    models/bert_intent_model/metrics.json
    models/bert_intent_model/label_classes.json
"""

import os
import json
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

MODEL_NAME = "bert-base-uncased"


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }


def train_bert(data_dir="data", output_dir="models/bert_intent_model",
               epochs=3, batch_size=16):

    train_df = pd.read_csv(f"{data_dir}/train.csv")
    test_df = pd.read_csv(f"{data_dir}/test.csv")

    # Turn intent names (strings) into numeric label ids
    label_encoder = LabelEncoder()
    train_df["label"] = label_encoder.fit_transform(train_df["intent"])
    test_df["label"] = label_encoder.transform(test_df["intent"])
    num_labels = len(label_encoder.classes_)

    print(f"Loading tokenizer & model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=64)

    train_ds = Dataset.from_pandas(train_df[["text", "label"]])
    test_ds = Dataset.from_pandas(test_df[["text", "label"]])

    train_ds = train_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels
    )

    training_args = TrainingArguments(
        output_dir="./bert_train_checkpoints",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Fine-tuning BERT-base on intent classification... (this takes a while)")
    trainer.train()

    metrics = trainer.evaluate()
    print("Final evaluation metrics:", metrics)

    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    with open(f"{output_dir}/label_classes.json", "w") as f:
        json.dump(label_encoder.classes_.tolist(), f)

    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

    print(f"Model saved to {output_dir}/")
    return metrics


if __name__ == "__main__":
    train_bert()
