import json
import math
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltkprocess import tokenization, stem
from model import NeuralNet


# --------------------------------------------------
# 1. LOAD INTENTS
# --------------------------------------------------

with open('/Users/akash/Documents/neural Nine/Bot/intent.json', 'r') as f:
    intents = json.load(f)

patterns = []
tags = []
labels = []

for intent in intents['intents']:
    tag = intent['tag']
    if tag not in tags:
        tags.append(tag)

    for pattern in intent['pattern']:
        patterns.append(pattern)
        labels.append(tag)

tags = sorted(tags)


# --------------------------------------------------
# 2. PREPROCESS (TOKENIZE + STEM)
# --------------------------------------------------

documents = []
for sentence in patterns:
    tokens = tokenization(sentence)
    stemmed = [stem(w.lower()) for w in tokens]
    documents.append(stemmed)


# --------------------------------------------------
# 3. BUILD VOCABULARY
# --------------------------------------------------

vocab = sorted(set(word for doc in documents for word in doc))


# --------------------------------------------------
# 4. TF
# --------------------------------------------------

def compute_tf(tokens):
    total = len(tokens)
    counts = Counter(tokens)
    return {w: c / total for w, c in counts.items()}


# --------------------------------------------------
# 5. IDF
# --------------------------------------------------

def compute_idf(documents, vocab):
    N = len(documents)
    idf = {}

    for word in vocab:
        df = sum(1 for doc in documents if word in doc)
        idf[word] = math.log(N / (1 + df))

    return idf


idf_dict = compute_idf(documents, vocab)


# --------------------------------------------------
# 6. TF-IDF VECTOR
# --------------------------------------------------

def tfidf_vector(tokens, vocab, idf_dict):
    tf = compute_tf(tokens)
    return [
        tf.get(word, 0.0) * idf_dict.get(word, 0.0)
        for word in vocab
    ]


X_train = []
y_train = []

for doc, tag in zip(documents, labels):
    X_train.append(tfidf_vector(doc, vocab, idf_dict))
    y_train.append(tags.index(tag))

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train)


# --------------------------------------------------
# 7. DATASET
# --------------------------------------------------

class ChatDataset(Dataset):
    def __init__(self):
        self.x_data = X_train
        self.y_data = y_train
        self.n_samples = len(self.x_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


# --------------------------------------------------
# 8. TRAINING SETUP (UNCHANGED)
# --------------------------------------------------

batch_size = 8
input_size = len(vocab)
hidden_size = 128
output_size = len(tags)
learning_rate = 0.001
num_epoch = 350

dataset = ChatDataset()
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print("Training Started...")

for epoch in range(num_epoch):
    for words, labels in loader:
        words = words.to(device)
        labels = labels.to(device)

        outputs = model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{num_epoch}], Loss: {loss.item():.4f}")

print("Training Finished")


# --------------------------------------------------
# 9. SAVE MODEL + TF-IDF METADATA
# --------------------------------------------------

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "vocab": vocab,
    "idf": idf_dict,
    "tags": tags
}

torch.save(data, "pytorch_modelBOT_tfidf.pt")
print("Model saved as pytorch_modelBOT_tfidf.pt")