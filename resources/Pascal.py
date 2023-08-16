import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model







class Chatbot:

    def __init__(self,resource_path):
        
        self.lemmatizer = WordNetLemmatizer()
        self.intents = json.loads(open(str(resource_path)+'intents.json').read())
    
        self.words = pickle.load(open(str(resource_path)+'words.pkl','rb'))
        self.classes = pickle.load(open(str(resource_path)+'classes.pkl','rb'))

        self.model =load_model(str(resource_path)+'model.h5')

    def clean_up_sentence(self,sentence):
        sentence_words =nltk.word_tokenize(sentence)
        sentence_words = [ self.lemmatizer.lemmatize(word) for word in sentence_words]
        return sentence_words

    def bag_of_Words(self,sentence):
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0]*len(self.words)
        for w in sentence_words:
            for i,word in enumerate(self.words):
                if word ==w:
                    bag[i]=1
        return np.array(bag)

    def predict_class(self,sentence):
        bow = self.bag_of_Words(sentence)
        res = self.model.predict(np.array([bow]),verbose=0)[0]
        ERROR_THRESHOLD = 0.3
        results = [[i,r] for i,r in enumerate(res)if r > ERROR_THRESHOLD]

        results.sort(key=lambda x: x[1],reverse=True)
        return_list =[]
        for r in results:
            return_list.append({'intent':self.classes[r[0]],'propability':str(r[1])})
        return return_list

    def get_response(self,intents_list, intents_json):
        tag= intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
        return result

