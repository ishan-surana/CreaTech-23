import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten
import re

# Load data from CSV file
data = pd.read_csv('reddit_posts_data.csv')

# Define a function to clean the text
def clean_text(text):
    # Remove unnecessary characters
    text = re.sub(r'[^\w\s]', '', text)
    # Replace repetitive line breaks and blank spaces with only one
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove emoticons and emojis
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    return text.lower()

# Lemmatization
lemmatizer = WordNetLemmatizer()
def lemmatize_text(text):
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

data['post_text'] = data['post_text'].apply(clean_text)
data['post_text'] = data['post_text'].apply(lemmatize_text)

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['post_text'])
X_seq = tokenizer.texts_to_sequences(data['post_text'])
X_pad = pad_sequences(X_seq, maxlen=100)

# Tokenize and count words for each sentiment category
stop_words = set(stopwords.words('english'))

positive_words = Counter()
negative_words = Counter()
neutral_words = Counter()

for index, row in data.iterrows():
    sentiment = row['opinion']
    text = row['post_text']
    cleaned_text = clean_text(text)
    tokens = word_tokenize(cleaned_text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    if sentiment == 'positive':
        positive_words.update(filtered_tokens)
    elif sentiment == 'negative':
        negative_words.update(filtered_tokens)
    elif sentiment == 'neutral':
        neutral_words.update(filtered_tokens)

# Function to analyze a sentence and predict sentiment probabilities
def predict_sentiment(sentence):
    cleaned_sentence = clean_text(sentence)
    tokens = word_tokenize(cleaned_sentence)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    total_positive_score = sum(positive_words.get(word, 0) for word in filtered_tokens)
    total_negative_score = sum(negative_words.get(word, 0) for word in filtered_tokens)
    total_neutral_score = sum(neutral_words.get(word, 0) for word in filtered_tokens)
    total_score = total_positive_score + total_negative_score + total_neutral_score
    if total_score == 0:
        # If no words found, consider it neutral
        positive_prob = 1/3
        negative_prob = 1/3
        neutral_prob = 1/3
    else:
        positive_prob = total_positive_score / total_score
        negative_prob = total_negative_score / total_score
        neutral_prob = total_neutral_score / total_score
    return {'positive': positive_prob, 'negative': negative_prob, 'neutral': neutral_prob}

# Create the Streamlit app
def main():
    st.title('Sentiment Analysis App')
    st.write('Enter a sentence and click the button to analyze its sentiment.')

    # Text input for user
    user_input = st.text_area("Enter a Reddit post text:")

    # Button to analyze sentiment
    if st.button('Analyze Sentiment'):
        if user_input:
            # Predict sentiment
            predictions = predict_sentiment(user_input)
            negative_prob = predictions['negative']
            neutral_prob = predictions['neutral']
            positive_prob = predictions['positive']

            # Determine major sentiment
            major_sentiment = max(predictions, key=predictions.get)

            # Display analyzed sentiment
            st.markdown('---')
            color_dict = {'negative': 'red', 'neutral': 'blue', 'positive': 'green'}
            color = color_dict[major_sentiment]
            st.markdown('<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
            st.markdown('<h3 style="text-align:center">Analyzed Sentiment:- </h3>', unsafe_allow_html=True)
            st.markdown(f'<div style="background-color:{color};border-radius:5px;padding:10px;padding-left:15px;padding-right:15px;color:white;text-align:center;max-width:max-content;margin: auto;margin-bottom:3%;">{major_sentiment.capitalize()}</div>', unsafe_allow_html=True)
            st.markdown('---')
            # Plot pie chart
            labels = ['Negative', 'Neutral', 'Positive']
            sizes = [negative_prob, neutral_prob, positive_prob]
            colors = ['red', 'blue', 'green']
            # Create a pie chart
            fig, ax = plt.subplots()
            wedges, pie_labels, percentages = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
            # Equal aspect ratio ensures that pie is drawn as a circle
            ax.axis('equal')
            # Add glowing effect
            for wedge in wedges:
                wedge.set_edgecolor('#000000')  # Black edge color
                wedge.set_linewidth(0.1)        # Edge width
            # Set transparent background
            fig.patch.set_alpha(0)
            # Set label text color to white
            for label in pie_labels:
                label.set_color('white')
            # Set percentage text to bold
            for pct in percentages:
                pct.set_fontweight('bold')
            with st.expander('Show pie-chart of probability predictions:-'):
                st.pyplot(fig)
        else:
            st.warning('Please enter a sentence.')


if __name__ == '__main__':
    main()
