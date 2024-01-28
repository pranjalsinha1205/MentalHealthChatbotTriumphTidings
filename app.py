from flask import Flask, render_template, request
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import GUIChatbot

app = Flask(__name__)


# Global list to store chat history
chat_history = []

# Create a function to generate responses from your chatbot
# Create a Sentiment Intensity Analyzer object
sid = SentimentIntensityAnalyzer()

# Modify the generate_response function to include sentiment analysis and plot creation
def generate_response(query):
    response = GUIChatbot.get_response(query)
    
    # Analyze sentiment of the user's input
    user_sentiment = sid.polarity_scores(query)['compound']
    
    # Append user sentiment and chat history to a global list
    chat_history.append({'user': query, 'bot': response, 'sentiment': user_sentiment})
    
    # Extract message numbers and user polarities for plotting
    message_numbers = list(range(1, len(chat_history) + 1))
    user_polarities = [entry['sentiment'] for entry in chat_history]
    
    # Plot user's polarity against message number
    plt.figure(figsize=(8, 6))
    plt.plot(message_numbers, user_polarities, marker='o', color='b', label='User Polarity')
    plt.xlabel('Number of Observations(N)')
    plt.ylabel('User Polarity')
    plt.title('User Polarity Distribution')
    plt.legend()
    plt.savefig('static/sentiment_plot.png')  # Save the plot as an image file
    
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
