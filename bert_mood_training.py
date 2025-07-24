import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertModel
import torch
import pickle
from tqdm import tqdm

# 1. Load dataset
df = pd.read_csv("mood_dataset.csv")
texts = df['text'].tolist()
labels = df['label'].tolist()

# 2. Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# 3. Load BERT tokenizer & model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased")

# 4. Tokenize + BERT embedding
def get_bert_embeddings(text_list):
    embeddings = []
    for text in tqdm(text_list, desc="Embedding"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = bert(**inputs)
        # Use [CLS] token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        embeddings.append(cls_embedding)
    return np.array(embeddings)

X = get_bert_embeddings(texts)
y = np.array(encoded_labels)

# 5. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# 7. Save model and label encoder
with open("bert_mood_model.pkl", "wb") as f:
    pickle.dump(clf, f)

with open("bert_label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("✅ BERT-based mood model trained and saved!")
print("📊 Evaluation:\n", classification_report(y_test, clf.predict(X_test), target_names=label_encoder.classes_))
