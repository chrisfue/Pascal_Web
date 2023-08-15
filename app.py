from flask import Flask, render_template, request
from resources.Pascal import Chatbot
import nltk

nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)

chat_history = []
pascal = Chatbot('./resources/')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_message = request.form['user_input']
        chat_history.append(("user", user_message))

        # Simulate bot response for demonstration purposes
        ints = pascal.predict_class(user_message)
        
        
        #bot_response = pascal.get_response(ints,pascal.intents)
        #chat_history.append(("bot", bot_response))

    return render_template('index.html', chat_history=chat_history)

if __name__ == '__main__':
    app.run(debug=True)
