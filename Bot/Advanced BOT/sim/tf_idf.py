import json
import math
from collections import Counter
from nltkprocess import tokenization, stem


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

with open("/Users/akash/Documents/neural Nine/Bot/intent.json", "r") as f:
    data = json.load(f)

patterns = []
tags = []

for intent in data["intents"]:
    tags.append(intent["tag"])
    for p in intent["pattern"]:
        patterns.append(p)


# --------------------------------------------------
# 2. PREPROCESS SENTENCES
#    (tokenize + lowercase + stem)
# --------------------------------------------------

documents = []
for sentence in patterns:
    tokens = tokenization(sentence)
    stemmed = [stem(w.lower()) for w in tokens]
    documents.append(stemmed)


# --------------------------------------------------
# 3. BUILD VOCABULARY
# --------------------------------------------------

vocab = sorted(set(
    word for doc in documents for word in doc
))


# --------------------------------------------------
# 4. TERM FREQUENCY (TF)
# --------------------------------------------------

def compute_tf(sentence_tokens):
    total_words = len(sentence_tokens)
    word_counts = Counter(sentence_tokens)

    tf_dict = {}
    for word, count in word_counts.items():
        tf_dict[word] = count / total_words

    return tf_dict


tf_list = [compute_tf(doc) for doc in documents]


# --------------------------------------------------
# 5. INVERSE DOCUMENT FREQUENCY (IDF)
# --------------------------------------------------

def compute_idf(documents, vocab):
    idf_dict = {}
    N = len(documents)

    for word in vocab:
        df = 0
        for doc in documents:
            if word in doc:
                df += 1

        idf_dict[word] = math.log(N / (1 + df))

    return idf_dict


idf_dict = compute_idf(documents, vocab)


# --------------------------------------------------
# 6. TF-IDF VECTOR
# --------------------------------------------------

def tfidf_vector(tf_dict, idf_dict, vocab):
    vector = []
    for word in vocab:
        tf = tf_dict.get(word, 0.0)
        idf = idf_dict.get(word, 0.0)
        vector.append(tf * idf)
    return vector


tfidf_vectors = [
    tfidf_vector(tf, idf_dict, vocab)
    for tf in tf_list
]


# --------------------------------------------------
# 7. CHECK OUTPUT
# --------------------------------------------------

print("Vocabulary size:", len(vocab))
print("TF example:", tf_list[0])
print("IDF example:", list(idf_dict.items())[:5])
print("TF-IDF vector length:", len(tfidf_vectors[0]))







