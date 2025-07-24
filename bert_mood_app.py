import streamlit as st
import pickle
from transformers import BertTokenizer, BertModel
import numpy as np
import torch
print("✅ PyTorch version:", torch.__version__)
# Load BERT & classifier
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased")

with open("bert_mood_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("bert_label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

st.title("🧠Mood Classifier from Journal Entry")
st.write("Enter a sentence describing your day. I'll predict your mood!")

text_input = st.text_area("✍️ How was your day?", height=150)

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = bert(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy().reshape(1, -1)


if st.button("Predict Mood") and text_input.strip():
    embedding = get_bert_embedding(text_input)
    pred_idx = model.predict(embedding)[0]
    mood = label_encoder.inverse_transform([pred_idx])[0]
    st.subheader("Predicted Mood:")
    st.write(f"🔹 {mood}")
