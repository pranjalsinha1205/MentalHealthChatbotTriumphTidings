from flask import Flask, render_template, request

app = Flask(__name__)

# Import your chatbot code
import GUIChatbot

# Global list to store chat history
chat_history = []

# Create a function to generate responses from your chatbot
def generate_response(query):
    response = GUIChatbot.get_response(query)
    chat_history.append({'user': query, 'bot': response})
    return response

# Define the index route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        response = generate_response(query)
        return render_template('index.html', response=response, chat_history=chat_history)
    else:
        return render_template('index.html', chat_history=chat_history)

# Define the about route
@app.route('/about')
def about():
    return render_template('about.html')

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
