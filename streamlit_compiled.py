import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import cloudpickle
import matplotlib.pyplot as plt
import nltk

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
# Load model architecture
model = load_model('final_model.h5')

# Load custom prediction function
with open("predict_sentiment.pkl", "rb") as f:
    predict_sentiment = cloudpickle.load(f)

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