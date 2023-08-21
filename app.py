from flask import Flask, render_template, request
from resources.Pascal import Chatbot
import nltk
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""



nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)

#initializeList for Chat display
chat_history = []
#initialize Chatbot Class and define subdirectory for resources
pascal = Chatbot('./resources/')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        #grab message from input box and append to chat list 
        user_message = request.form['user_input']
        chat_history.append(("user", user_message))

        #analyze input
        ints = pascal.predict_class(user_message)
        #retrieve response
        bot_response = pascal.get_response(ints,pascal.intents)
        #append response to chat list
        chat_history.append(("bot", bot_response))
    
    #clear chat when website is called anew
    if request.method=='GET':
        chat_history.clear()

    #update display from index.html with new chat list
    return render_template('index.html', chat_history=chat_history)

if __name__ == '__main__':
    app.run(debug=True)
