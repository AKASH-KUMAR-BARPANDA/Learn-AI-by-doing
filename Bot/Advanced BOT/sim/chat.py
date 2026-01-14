import random
import json
import math
from collections import Counter

import torch
from model import NeuralNet
from nltkprocess import tokenization, stem


# --------------------------------------------------
# DEVICE
# --------------------------------------------------

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


# --------------------------------------------------
# LOAD INTENTS (FOR RESPONSES)
# --------------------------------------------------

with open('/Users/akash/Documents/neural Nine/Bot/intent.json', 'r') as f:
    intents = json.load(f)


# --------------------------------------------------
# LOAD TRAINED MODEL + TF-IDF DATA
# --------------------------------------------------

data = torch.load("pytorch_modelBOT_tfidf.pt")

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
vocab = data["vocab"]
idf_dict = data["idf"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()


# --------------------------------------------------
# TF-IDF FUNCTIONS (SAME AS TRAINING)
# --------------------------------------------------

def compute_tf(tokens):
    total = len(tokens)
    counts = Counter(tokens)
    return {w: c / total for w, c in counts.items()}


def tfidf_vector(tokens, vocab, idf_dict):
    tf = compute_tf(tokens)
    return [
        tf.get(word, 0.0) * idf_dict.get(word, 0.0)
        for word in vocab
    ]


# --------------------------------------------------
# CHAT LOOP
# --------------------------------------------------

bot_name = "Tweet"
print("Let's chat! type 'quit' to exit")

while True:
    sentence = input("You: ")
    if sentence.lower() == "quit":
        break

    # preprocess input
    tokens = tokenization(sentence)
    stemmed = [stem(w.lower()) for w in tokens]

    # TF-IDF vector
    vec = tfidf_vector(stemmed, vocab, idf_dict)
    X = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(device)

    # model inference
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    confidence = probs[0][predicted.item()]

    if confidence.item() > 0.75:
        for intent in intents["intents"]:
            if intent["tag"] == tag:
                print(f"{bot_name}: {random.choice(intent['response'])}")
                break
    else:
        print(f"{bot_name}: I do not understand...")