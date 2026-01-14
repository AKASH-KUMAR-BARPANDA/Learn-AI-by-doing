import nltk
from nltk.stem.porter import PorterStemmer
import  numpy as np

stemmer = PorterStemmer()
# tokenization => converting sentence to word
def tokenization(sentence):
    return nltk.word_tokenize(sentence)

# stemming => removing basic grammar like ing.. etc
def stem(word):
    return stemmer.stem(word.lower())

def bag_of_word2(tokenized_sentence,all_word):
    """
    :param tokenized_sentence:['hello','how','are','you']
    :param all_word:['hi','hello','I','you','bye','thanks','cool']
    :return:[0,1,0,1,0,0,0]
    """
    tokenized_sentence = [stem(w) for w in tokenized_sentence]

    bag = np.zeros(len(all_word),dtype=np.float32)
    for idx,w in enumerate(all_word):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag


# tokenized_sentence2 = ['hello','how','are','you']
# all_word = ['hi','hello','I','you','bye','thanks','cool']
# print(bag_of_word2(tokenized_sentence2,all_word))

# a = "how long it will take for shipping ?"
# # print([stem(word) for word in tokenization(a)])
# list_of_token = tokenization(a)
# for i in list_of_token:
#     print(stem(i))
