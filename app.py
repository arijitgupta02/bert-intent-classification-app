"""
STEP 6: Streamlit web app for the Intent Classification project.

Run with:
    streamlit run app.py

Requires that you've already run (in order):
    1. data_prep.py
    2. baseline_tfidf.py
    3. train_bert.py
"""

import time
import json
import joblib
import torch
import pandas as pd
import streamlit as st
import plotly.express as px
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="Intent Classifier", page_icon="🤖", layout="centered")

BERT_MODEL_DIR = "models/bert_intent_model"
TFIDF_DIR = "models/tfidf_baseline"


@st.cache_resource
def load_bert():
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
    model.eval()
    with open(f"{BERT_MODEL_DIR}/label_classes.json") as f:
        labels = json.load(f)
    return tokenizer, model, labels


@st.cache_resource
def load_tfidf():
    vectorizer = joblib.load(f"{TFIDF_DIR}/vectorizer.pkl")
    clf = joblib.load(f"{TFIDF_DIR}/classifier.pkl")
    return vectorizer, clf


def predict_bert(text, tokenizer, model, labels, top_k=5):
    start = time.time()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0]
    latency = (time.time() - start) * 1000
    top_indices = torch.argsort(probs, descending=True)[:top_k]
    results = [(labels[i], float(probs[i])) for i in top_indices]
    return results, latency


def predict_tfidf(text, vectorizer, clf, top_k=5):
    start = time.time()
    X = vectorizer.transform([text])
    probs = clf.predict_proba(X)[0]
    latency = (time.time() - start) * 1000
    top_indices = probs.argsort()[::-1][:top_k]
    results = [(clf.classes_[i], float(probs[i])) for i in top_indices]
    return results, latency


def show_bar_chart(results, title):
    df = pd.DataFrame(results, columns=["Intent", "Confidence"])
    fig = px.bar(
        df, x="Confidence", y="Intent", orientation="h",
        range_x=[0, 1], title=title,
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)


st.title("🤖 BERT Intent Classification App")
st.write(
    "Type a customer-support style message and see which intent it matches, "
    "comparing a fine-tuned BERT model against a TF-IDF baseline."
)

user_input = st.text_area(
    "Enter your message:",
    placeholder="e.g. I lost my card, what do I do?",
)

model_choice = st.radio(
    "Choose model:",
    ["BERT (fine-tuned)", "TF-IDF baseline", "Both"],
    horizontal=True,
)

if st.button("Classify Intent") and user_input.strip():

    if model_choice in ("BERT (fine-tuned)", "Both"):
        try:
            tokenizer, model, labels = load_bert()
            results, latency = predict_bert(user_input, tokenizer, model, labels)
            st.subheader("BERT Prediction")
            st.write(f"⏱️ Latency: {latency:.2f} ms")
            show_bar_chart(results, "Top 5 Predicted Intents (BERT)")
        except OSError:
            st.error(
                "BERT model not found in 'models/bert_intent_model'. "
                "Please run train_bert.py first."
            )

    if model_choice in ("TF-IDF baseline", "Both"):
        try:
            vectorizer, clf = load_tfidf()
            results, latency = predict_tfidf(user_input, vectorizer, clf)
            st.subheader("TF-IDF Baseline Prediction")
            st.write(f"⏱️ Latency: {latency:.2f} ms")
            show_bar_chart(results, "Top 5 Predicted Intents (TF-IDF)")
        except FileNotFoundError:
            st.error(
                "TF-IDF baseline not found in 'models/tfidf_baseline'. "
                "Please run baseline_tfidf.py first."
            )

st.markdown("---")

