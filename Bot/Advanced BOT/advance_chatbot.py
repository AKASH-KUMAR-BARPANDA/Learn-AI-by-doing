import os
import nltk
import json
import numpy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import dataloader,TensorDataset

from Bot.SimpleBOT.trainingBOT import documents


class ChatbotModel(nn.Module):

    def __int__(self,input_size,output_size):
        super(ChatbotModel,self).__init__()

        # fc -> fully connected layer
        self.fc1 = nn.Linear(in_features=input_size,out_features=128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,output_size)
        self.relu = nn.ReLU
        self.softmax = nn.Softmax
        self.dropout = nn.Dropout(0.5)

    ## how you are taking input and processing for the output
    def forward(self,x):
        # layer 1
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        #layer2
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        #layer2
        x = self.fc3(x)
        #x = self.softmax(x)  --> implicitly apply softmax

class ChatbotAssistant:

    def __init__(self,intents_path, function_mapping = None):
        self.model = None
        self.intents_path = intents_path

        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intent_response = {}

        self.function_mapping = function_mapping

        self.X = None
        self.y = None

    @staticmethod
    def tokenize_and_lemmatize(text):
        word_lemmatizer = nltk.WordNetLemmatizer()

        words = nltk.word_tokenize(text)
        words = [word_lemmatizer.lemmatize(word.lower()) for word in words]

        return words

    @staticmethod
    def bag_of_word(words,vocabulary):
        result = []
        for word in vocabulary:
            if word in words:
                result.append(1)
            else:
                result.append(0)
        return result

    def parse_intents(self):
        word_lemmatizer = nltk.WordNetLemmatizer()

        if os.path.exists(self.intents_path):
            with open(self.intents_path,'r') as f:
                intents_data = json.load(f)

            for intent in intents_data['intents']:
                if intent['tag'] not in self.intents:
                    self.intents.append(intent['tag'])
                    self.intent_response[intent['tag']] = intent['response']

                for pattern in intents_data['pattern']:
                    pattern_word = self.tokenize_and_lemmatize(pattern)
                    self.vocabulary.append(pattern_word)
                    self.documents.append((pattern_word,intent['tag']))

                self.vocabulary = sorted(set(self.vocabulary))



    def prepare_data(self):
        bags = []
        indices = []

        for document in self.documents:
            words = document[0]
            bags = self.bag_of_word(words)

            intent_index  = self.intents.index(document[1])
            bags.append(bags)
            indices.append(intent_index)

        self.X = np.array(bags)
        self.y = np.array(indices)

    def train_model(self,batch_size,lr , epochs):
        pass





















