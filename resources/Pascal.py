import random
import json
import pickle
import numpy as np

import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model



class Chatbot:

    #initialize and load necessary files
    def __init__(self, resource_path):
        self.lemmatizer = tf.lite.Interpreter(model_path=str(resource_path) + 'model.tflite')
        self.intents = json.loads(open(str(resource_path) + 'intents.json').read())
        self.words = pickle.load(open(str(resource_path) + 'words.pkl', 'rb'))
        self.classes = pickle.load(open(str(resource_path) + 'classes.pkl', 'rb'))
        self.lemmatizer_nltk = WordNetLemmatizer()

    #split text to tokens and lemmatize them
    def clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer_nltk.lemmatize(word) for word in sentence_words]
        return sentence_words

    #create bag of words
    def bag_of_Words(self, sentence):
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(self.words)
        for w in sentence_words:
            for i, word in enumerate(self.words):
                if word == w:
                    bag[i] = 1
        return np.array(bag)

    #predict class based on processed input
    def predict_class(self, sentence):
        bow = self.bag_of_Words(sentence)
        self.lemmatizer.allocate_tensors()
        input_details = self.lemmatizer.get_input_details()
        output_details = self.lemmatizer.get_output_details()

        self.lemmatizer.set_tensor(input_details[0]['index'], np.array([bow], dtype=np.float32))
        self.lemmatizer.invoke()
        res = self.lemmatizer.get_tensor(output_details[0]['index'])[0]

        ERROR_THRESHOLD = 0.7
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({'intent': self.classes[r[0]], 'probability': str(r[1])})
        return return_list

    #retrieve response based on predicted class.
    def get_response(self, intents_list, intents_json):
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = i['responses']
                break
        return result
