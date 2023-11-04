# MentalHealthChatbotTriumphTidings
A chatbot to help people suffering from depression to feel better by talking with the chatbot. 

# Chat with the bot
![proof](https://github.com/pranjalsinha1205/MentalHealthChatbotTriumphTidings/assets/112460531/3743abf9-e729-4371-a348-e39106b073a6)

# Polarity graph
![image](https://github.com/pranjalsinha1205/MentalHealthChatbotTriumphTidings/assets/112460531/074950b4-182d-4d44-94f9-143ed2027fcb)


# Abstract
Mental health is a major public health concern, with millions of people suffering from depression, anxiety, and other mental health conditions. Chatbots have the potential to provide a valuable tool for supporting mental health, by offering confidential and accessible support. This project presents the development of a mental health chatbot that can be used to provide emotional support and resources to people who are feeling lonely, depressed, or anxious.

The chatbot was developed using a combination of machine learning and natural language processing techniques. The chatbot is trained on a dataset of conversations between people who are struggling with mental health challenges and mental health professionals. This training allows the chatbot to learn how to respond to user input in a way that is both supportive and informative.

The chatbot can be used by people of all ages and backgrounds. It is easy to use and does not require any prior knowledge of mental health conditions. The chatbot can be accessed through a variety of platforms, including web chat, mobile app, and SMS.

The chatbot is designed to be used as a supplement to professional mental health care. It is not intended to diagnose or treat mental health conditions. However, the chatbot can provide valuable support and resources to people who are struggling with mental health challenges.

# Introduction
In the bustling corridors of higher education, where knowledge intertwines with ambition, a silent epidemic pervades the lives of countless students: loneliness. The college experience, often hailed as a time of intellectual growth and personal discovery, can be paradoxically isolating. For many, the transition from the familiarity of home to an unfamiliar campus in a different city or country can trigger profound feelings of solitude. This phenomenon is not limited by geographical boundaries; rather, it reverberates in dormitories, lecture halls, and communal spaces worldwide.

Loneliness, if left unaddressed, can metamorphose into a formidable adversary, gnawing at the core of mental well-being. It breeds feelings of alienation, anxiety, and melancholy, and in its darkest corners, it can pave the way for severe mental health issues. College life, with its academic pressures, social expectations, and newfound responsibilities, can magnify the impact of loneliness, rendering even the most resilient souls vulnerable.

# The Significance of the Problem:
The importance of addressing this issue cannot be overstated. Loneliness not only affects the individual's mental health but also seeps into the fabric of the community, creating a collective sense of disconnection. This estrangement can hinder academic performance, social integration, and overall happiness, making it imperative to find innovative solutions that transcend the barriers of physical distance.


# The Motivation Behind Triumph Tidings:
The genesis of Triumph Tidings lies in the shared experiences of countless students who have grappled with loneliness. As the project creator found themselves thousands of kilometers away from home, the pangs of isolation became palpable. Yet, amidst this struggle, a vision emerged — a vision of crafting a digital companion capable of mitigating the emotional turbulence faced by students worldwide.

# Triumph Tidings as a Supportive Companion:
Triumph Tidings is not a mere technological endeavor; it is a manifestation of empathy and understanding. It is a response to the silent cries for connection echoing in the dormitories and common rooms. By harnessing the power of artificial intelligence, natural language processing, and machine learning, Triumph Tidings aspires to become more than just a chatbot. It is conceived as a virtual confidante, a steadfast companion that listens without judgment, speaks with compassion, and offers unwavering support.

# The Ripple Effect of Triumph Tidings:
Beyond the individual impact, Triumph Tidings is poised to create a ripple effect. By alleviating the burden of loneliness, it nurtures a sense of belonging and camaraderie. It fosters resilience, empowers individuals to face challenges head-on, and provides a ray of hope in the face of despair. As students find solace in the digital realm, they are better equipped to navigate the complexities of college life, forging meaningful connections and embracing their academic journey with newfound vigour.


# RELATED WORK
The field of mental health chatbots has witnessed significant advancements in recent years. Several notable projects have emerged, aiming to address the pressing issue of mental health and provide accessible support to individuals. One such example is Woebot, an AI-powered chatbot developed by psychologists and AI experts. Woebot utilizes cognitive-behavioral therapy techniques to engage users in interactive conversations, offering personalized emotional support and coping strategies.
Wysa, another prominent mental health chatbot, integrates AI with evidence-based therapeutic techniques. It focuses on emotional well-being, guiding users through mindfulness exercises, mood tracking, and relaxation techniques. Replika, an AI chatbot designed as a personal AI companion, provides users with a virtual friend for open conversations, thereby reducing feelings of loneliness and providing a non-judgmental space for self-expression.
By analyzing these existing solutions, TriumphTidings aims to draw inspiration and incorporate the best practices into its design. TriumphTidings distinguishes itself through its focus on not only providing empathetic responses but also encouraging users to take an active role in managing their mental health, fostering a sense of empowerment and resilience.


# PROBLEM STATEMENT
The alarming rise in mental health issues, fueled by the challenges of modern life, necessitates innovative solutions. 
A significant problem lies in the lack of accessible and personalized mental health support. Traditional therapy methods face limitations due to high costs, stigma, and limited availability. Furthermore, many individuals, especially students and young adults, struggle with feelings of loneliness and anxiety, particularly due to being away from home and the constant fear of future. 
TriumphTidings addresses this critical gap by offering an AI-powered mental health chatbot. This chatbot provides an empathetic, understanding presence, guiding users through their emotions, promoting self-awareness, and delivering coping mechanisms. The project's primary goal is to offer immediate, stigma-free, and effective mental health support to those in need, ensuring no one faces their battles alone.

# APPROACH
TriumphTidings employs Natural Language Processing (NLP) and machine learning techniques to create an empathetic AI chatbot. Utilizing NLTK for language processing, the chatbot analyzes user inputs, identifying emotions and intents. 
This model is trained on a dataset containing categorized patterns and responses.(Dataset taken from Kaggle, link: https://www.kaggle.com/datasets/elvis23/mental-health-conversational-data) and 

this link from GitHub:
https://github.com/cherrywine-04/-MindfulMate/blob/main/intents1.json
The training enables the bot to understand and respond contextually. The graphical user interface (GUI) is built using Flask, enhancing user experience. TriumphTidings stands out for its personalized approach, tailoring responses to individual emotional states. 
Additionally, it incorporates features for recommending stress-buster exercises, fostering user engagement. The approach ensures a holistic, user-friendly, and effective mental health support system, promoting emotional well-being.
Also, it give a graph showing the polarity trends of the user which in turn will help the medical or psychiatric practitioners to further understand the user’s mental health at least upto some extent.

# Algorithms:
1) LSTM (Long Short-Term Memory): LSTM neural network is used for sequence modeling and prediction of user intents from their input.

2) Word Embedding: Words are represented as dense vectors to capture semantic relationships, enhancing the model's understanding of language nuances.

3) Tokenization: The process of breaking down text into words or subwords, essential for preparing text data for analysis.

4) Lemmatization: Lemmatization is the process of reducing words to their base or root form, ensuring different forms of words are treated as the same, aiding in better analysis.

5) Vader: Used to assign polarity between -1 and 1 to each and every sentence written by the user, if the polarity is >0.5, the user is +ve, if it is <0.5 the user is negative, and anything between that is neutral or partially positive or negative

These algorithms collectively contribute to the chatbot's ability to process user input, understand context, and generate appropriate responses.

# Algorithm Explanation
1) Tokenization and Lemmatization:

  i) Intent Data Preparation:
  The chatbot loads intent data from a JSON file, containing patterns and corresponding responses for various user intents.
  ii) Tokenization and Lemmatization:
    a)	User input, as well as patterns from intents, are tokenized into words to break down sentences into individual components.
    b)	The NLTK library is employed to lemmatize the words, reducing them to their base or root form, ensuring uniformity and enhancing analysis.
    c) These processed words are organized into a bag of words, forming    the basis for understanding user input patterns.

2) Neural Network Model Construction:

  i) Architecture Setup:
    a) A Neural Network model is constructed using the Keras library, comprising input, hidden, and output layers.
    b) Input layer: Accepts numerical vectors derived from the bag of words representing input patterns.
    c) Hidden layers: These layers process and learn complex patterns within the input data, enhancing the model's understanding.
    d) Output layer: Produces probabilities for each intent class, indicating the likelihood of the user's input corresponding to a specific intent.

  ii) Training Process:
    a)	Input patterns are converted into numerical vectors to be used for training the Neural Network model.
    b) The model is trained using categorical cross-entropy loss function and the Adam optimizer, iterating through epochs to refine its accuracy.

3) User Input Processing:

  i) Preprocessing User Input:
    a) User input is tokenized and lemmatized to ensure consistency and uniformity.
    b) The processed input is converted into a bag of words, aligning it with the format used during training.
  ii) Intent Prediction:
    a)	The trained Neural Network model predicts the intent class probabilities for the preprocessed user input.
    b) If the highest predicted probability exceeds a predefined threshold (ERROR_THRESHOLD), the corresponding intent is identified as the user's intent.

4) Generating Responses:

  i) Intent-Based Response Selection:
    a) If a valid intent is recognized based on the predicted probabilities exceeding the threshold:
    b) A relevant response is randomly selected from the pre-defined responses associated with that intent.

  ii) Fallback to Default Response:
    a)	If no valid intent is found (probabilities below the threshold):
      a.1) A default response, indicating a lack of understanding, is generated, ensuring a response even in ambiguous situations.

  This modular approach allows the chatbot to process user input effectively, predict intents accurately, and generate contextually appropriate responses, enhancing the user experience and interaction quality.

5) Sentiment Analysis: Sentiment analysis is performed using the VADER (Valence Aware Dictionary and Sentiment Reasoner) tool, tailored for social media text analysis. The process involves:
  i) Intent Data Preparation:
  Intent data is loaded from a JSON file, containing user patterns and responses.
  ii) Tokenization and Sentiment Analysis:
    a) User input is tokenized into sentences for analysis.
    b) The NLTK library is used for tokenization and lemmatization.
    c) Sentiment analysis with VADER assigns a compound score to assess the text's overall sentiment.
  iii) Data Visualization:
  Sentiment analysis results, including compound scores, are plotted against message numbers using Matplotlib. This graphical representation provides insights into the conversation's emotional tone over time.

# Conclusion
In conclusion, the development of this mental health chatbot represents a significant step toward providing accessible and empathetic support for individuals struggling with loneliness and depression. By leveraging natural language processing and machine learning techniques, the chatbot can engage users in meaningful conversations, offering a virtual companion that understands their emotions and responds with empathy. The integration of LSTM neural networks and word embedding enables the chatbot to comprehend the complexity of human language, making it capable of generating contextually relevant and sensitive responses.
While the chatbot demonstrates promising results, there is still room for improvement. Further research and development could focus on enhancing the chatbot's emotional intelligence, refining its ability to recognize subtle cues in user input, and expanding its repertoire of responses to cater to a wider range of situations. Additionally, user feedback and real-world testing are essential to iteratively improve the chatbot's effectiveness and user experience.
Overall, this project showcases the potential of technology in addressing mental health challenges and highlights the importance of continued innovation in the field of mental health support systems. Through thoughtful design, rigorous testing, and ongoing refinement, mental health chatbots can become valuable companions, offering solace and understanding to those in need.
