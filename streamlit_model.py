import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Input, Concatenate, Dropout, Flatten
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import emoji
import streamlit as st
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Load data from CSV file
data = pd.read_csv('reddit_posts_data.csv')

# Data Preprocessing
# Data Cleaning
def clean_text(text):
    # Remove unnecessary characters
    text = re.sub(r'[^\w\s]', '', text)
    # Replace repetitive line breaks and blank spaces with only one
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove emoticons and emojis
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    return text

data['post_text'] = data['post_text'].apply(clean_text)

# Lemmatization
lemmatizer = WordNetLemmatizer()
def lemmatize_text(text):
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

data['post_text'] = data['post_text'].apply(lemmatize_text)

# POS Tagging
def pos_tagging(text):
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    return tagged_tokens

# Apply POS tagging
data['post_text_pos'] = data['post_text'].apply(pos_tagging)

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['post_text'])
X_seq = tokenizer.texts_to_sequences(data['post_text'])
X_pad = pad_sequences(X_seq, maxlen=100)

# Stopwords removal
stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(filtered_tokens)

data['post_text'] = data['post_text'].apply(remove_stopwords)

# Define the function to extract structural features
def extract_structural_features(text):
    # Implement feature extraction logic
    message_length = len(text)
    num_tokens = len(word_tokenize(text))
    num_hashtags = text.count('#')
    num_emails = len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
    num_urls = text.count('http://') + text.count('https://')
    num_periods = text.count('.')
    num_commas = text.count(',')
    num_digits = sum(c.isdigit() for c in text)
    num_sentences = len(sent_tokenize(text))
    num_mentioned_users = text.count('@')
    num_uppercase = sum(c.isupper() for c in text)
    num_question_marks = text.count('?')
    num_exclamation_marks = text.count('!')
    emojis = set(re.findall(r'\:[\w]+\:', emoji.demojize(text)))
    num_emoticons = len(emojis)
    num_dollar_symbols = text.count('$')
    # Other symbols
    num_other_symbols = len([char for char in text if char not in '"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#@.://,?!' + ''.join(emojis)])
    # Return features as a list
    return [message_length, num_tokens, num_hashtags, num_emails, num_urls, num_periods, num_commas, num_digits, num_sentences, num_mentioned_users, num_uppercase, num_question_marks, num_exclamation_marks, num_emoticons, num_dollar_symbols, num_other_symbols]

# Apply the function to extract structural features and create new columns
data['structural_features'] = data['post_text'].apply(extract_structural_features)

# Convert labels to numerical values
label_dict = {'negative': 0, 'neutral': 1, 'positive': 2}
data['opinion'] = data['opinion'].map(label_dict)

# Normalize 'post_upvotes' column
data['post_upvotes'] = (data['post_upvotes'] - data['post_upvotes'].min()) / (data['post_upvotes'].max() - data['post_upvotes'].min())

# Define the model architecture
input_content = Input(shape=(100,), name='content_input')
input_upvotes = Input(shape=(1,), name='upvotes_input')

embedding = Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=100, input_length=100)(input_content)
conv_layer = Conv1D(128, 5, activation='relu')(embedding)
pooling_layer_content = GlobalMaxPooling1D()(conv_layer)

# Upvotes input
dense_upvotes = Dense(32, activation='relu')(input_upvotes)

# Concatenate content and upvotes
concatenated_inputs = Concatenate()([pooling_layer_content, dense_upvotes])

# Additional fully connected layers
dense1 = Dense(128, activation='relu')(concatenated_inputs)
output = Dense(3, activation='softmax')(dense1)  # Output

# Define the model
model = Model(inputs=[input_content, input_upvotes], outputs=output)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Sentiment Prediction Function
def predict_sentiment(user_input, user_upvotes):
    cleaned_input = clean_text(user_input)
    lemmatized_input = lemmatize_text(cleaned_input)
    tokenized_input = tokenizer.texts_to_sequences([lemmatized_input])
    padded_input = pad_sequences(tokenized_input, maxlen=100)

    user_input_pad = [padded_input, np.array([[user_upvotes]])]

    predictions = model.predict(user_input_pad)

    return predictions

# Streamlit App
st.title("Reddit Sentiment Analysis")

user_input = st.text_area("Enter a Reddit post text:")

user_upvotes = st.number_input("Enter the number of upvotes:", min_value=0, value=0)

if st.button("Predict Sentiment"):
    if user_input:
        predictions = predict_sentiment(user_input, user_upvotes)
        negative_prob = predictions[0][0]
        neutral_prob = predictions[0][1]
        positive_prob = predictions[0][2]

        st.write("Sentiment Probabilities:")
        # Plot pie chart
        labels = ['Negative', 'Neutral', 'Positive']
        sizes = [negative_prob, neutral_prob, positive_prob]
        colors = ['red', 'blue', 'green']

        # Create a pie chart
        fig, ax = plt.subplots()
        wedges, labels, percentages = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)

        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.axis('equal')

        # Add glowing effect
        for wedge in wedges:
            wedge.set_edgecolor('#000000')  # Black edge color
            wedge.set_linewidth(0.1)        # Edge width

        # Set transparent background
        fig.patch.set_alpha(0)

        # Set label text color to white
        for label in labels:
            label.set_color('white')

        # Set percentage text to bold
        for pct in percentages:
            pct.set_fontweight('bold')

        st.pyplot(fig)