import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from collections import Counter
from model_new_trained import clean_text, word_tokenize, tokenizer, pad_sequences

# Load positive_words
with open("positive_words.txt", "r", encoding="utf-8") as f:
    positive_words_list = f.read().splitlines()
positive_words = Counter(positive_words_list)

# Load negative_words
with open("negative_words.txt", "r", encoding="utf-8") as f:
    negative_words_list = f.read().splitlines()
negative_words = Counter(negative_words_list)

# Load neutral_words
with open("neutral_words.txt", "r", encoding="utf-8") as f:
    neutral_words_list = f.read().splitlines()
neutral_words = Counter(neutral_words_list)

# Load model architecture
model = load_model('final_model.h5',compile=False)

def predict_sentiment(text):
    cleaned_input = clean_text(text)
    tokens = word_tokenize(cleaned_input)
    filtered_tokens = [word for word in tokens]
    final_text = ' '.join(filtered_tokens)
    tokenized_text = tokenizer.texts_to_sequences([cleaned_input])
    padded_text = pad_sequences(tokenized_text, maxlen=100)
    
    negation_window = 3
    negations = set(["no", "not", "never", "none", "nobody", "nothing", "nowhere", "neither", "nor", "bad", "worse","worst"])
    negate = False
    for i, word in enumerate(filtered_tokens):
        if word.lower() in negations:
            negate = not negate
            # Extend the negation window
            for j in range(1, min(negation_window + 1, len(filtered_tokens) - i)):
                if filtered_tokens[i + j] not in negations:
                    if negate:
                        # Reverse sentiment of the current word
                        if filtered_tokens[i + j] in positive_words:
                            filtered_tokens[i + j] = "not_" + filtered_tokens[i + j]
                            negative_words.update([filtered_tokens[i+j]])
                        elif filtered_tokens[i + j] in negative_words:
                            filtered_tokens[i + j] = "not_" + filtered_tokens[i + j]
                            positive_words.update([filtered_tokens[i+j]])
                        elif filtered_tokens[i + j] in neutral_words:
                            filtered_tokens[i + j] = "not_" + filtered_tokens[i + j]
                    else:
                        break
    
    total_positive_score = sum(positive_words.get(word, 0) for word in filtered_tokens)
    total_negative_score = sum(negative_words.get(word, 0) for word in filtered_tokens)
    total_neutral_score = sum(neutral_words.get(word, 0) for word in filtered_tokens)
    total_score = total_positive_score + total_negative_score + total_neutral_score
    if total_score == 0:
        positive_prob = 1/3
        negative_prob = 1/3
        neutral_prob = 1/3
    else:
        positive_prob = total_positive_score / total_score
        negative_prob = total_negative_score / total_score
        neutral_prob = total_neutral_score / total_score
    return [negative_prob, neutral_prob, positive_prob]

st.title('Sentiment Analysis App')
st.write('Enter a sentence and click the button to analyze its sentiment.')
# Text input for user
user_input = st.text_area("Enter a Reddit post text:")
# Button to analyze sentiment
if st.button('Analyze Sentiment'):
    if user_input:
        # Predict sentiment
        sentiment_list = ['negative', 'neutral','positive']
        predictions = predict_sentiment(user_input)
        negative_prob = predictions[0]
        neutral_prob = predictions[1]
        positive_prob = predictions[2]
        # Determine major sentiment
        major_sentiment = sentiment_list[np.argmax(predictions)]
        if neutral_prob == predictions[np.argmax(predictions)]:
            major_sentiment = 'neutral'
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