import tkinter as tk    #for gui
from tkinter import scrolledtext   #for scrollable text as in real chat
import nltk   #for language processing
import numpy as np  #for scientific calculations
import json  #for importing the intents json file
import random  #for generating a random response from a list of responses
from keras.models import Sequential 
from keras.layers import Dense, Embedding, LSTM, Dropout
from nltk.stem import WordNetLemmatizer

# Load intents file
with open('intents.json', 'r') as file:  #opening the json file in read-only mode
    intents = json.load(file)

# Extract words and classes from intents file
words = []
classes = []
documents = []
ignore_words = ['?', '!']
lemmatizer = WordNetLemmatizer()

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize words
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize words and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# Sort classes
classes = sorted(list(set(classes)))

# Create training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Pad the sublists to make them equal in length
max_length = len(max(training, key=lambda x: len(x[0]))[0])
for sublist in training:
    sublist[0] += [0] * (max_length - len(sublist[0]))

random.shuffle(training)
training = np.array(training, dtype=object)

# Split data into training and testing sets
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Build the model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model to the training data
model.fit(np.array(train_x), np.array(train_y), epochs=100, batch_size=5, verbose=1)

# Function to clean and tokenize user input
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Function to create a bag of words from user input
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"Found in bag: {w}")
    return np.array(bag)

# Function to predict the class of the user input and get a response
def predict_class(sentence):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# Function to handle user input and generate a response
def get_response(user_input):
    response = "I'm sorry, I don't understand."
    ints = predict_class(user_input)
    tag = ints[0]['intent']
    for intent in intents['intents']:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            break
    return response
