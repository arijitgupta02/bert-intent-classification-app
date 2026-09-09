"""
STEP 1: Download and prepare the intent classification dataset.

We use the "Banking77" dataset (77 banking customer-support intents,
e.g. "lost_or_stolen_card", "card_payment_fee_charged", etc.).
It is small enough to train on quickly but realistic enough to be a
proper intent classification problem.

Run this first:
    python data_prep.py

It creates a "data" folder with:
    data/train.csv
    data/test.csv
    data/labels.txt
"""

import os
import pandas as pd
from datasets import load_dataset


def prepare_data(output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)

    print("Downloading the Banking77 intent classification dataset...")
    dataset = load_dataset("banking77")

    train_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset["test"])

    label_names = dataset["train"].features["label"].names
    train_df["intent"] = train_df["label"].apply(lambda x: label_names[x])
    test_df["intent"] = test_df["label"].apply(lambda x: label_names[x])

    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    labels_path = os.path.join(output_dir, "labels.txt")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    with open(labels_path, "w") as f:
        for name in label_names:
            f.write(name + "\n")

    print(f"Saved {len(train_df)} training examples -> {train_path}")
    print(f"Saved {len(test_df)} test examples -> {test_path}")
    print(f"Number of intent classes: {len(label_names)}")

    return train_df, test_df, label_names


if __name__ == "__main__":
    prepare_data()
