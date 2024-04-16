import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Input, Concatenate
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import cloudpickle

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

rd = pd.read_csv('reddit_posts_data.csv')
rd['post_text'] = rd['post_title'] + ' ' + rd['post_text']
td = pd.read_csv('twitter_data.csv')
data = pd.concat([rd[['post_text','opinion']], td[['post_text','opinion']]],ignore_index=True)

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

lemmatizer = WordNetLemmatizer()
def lemmatize_text(text):
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

data['post_text'] = data['post_text'].apply(clean_text)
data['post_text'] = data['post_text'].apply(lemmatize_text)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['post_text'])
X_seq = tokenizer.texts_to_sequences(data['post_text'])
X_pad = pad_sequences(X_seq, maxlen=100)

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
    if sentiment == 'positive' or sentiment == 2:
        positive_words.update(filtered_tokens)
    elif sentiment == 'negative' or sentiment == 0:
        negative_words.update(filtered_tokens)
    elif sentiment == 'neutral' or sentiment == 1:
        neutral_words.update(filtered_tokens)

def predict_sentiment(text):
    cleaned_input = clean_text(text)
    tokens = word_tokenize(cleaned_input)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    final_text = ' '.join(filtered_tokens)
    tokenized_text = tokenizer.texts_to_sequences([cleaned_input])
    padded_text = pad_sequences(tokenized_text, maxlen=100)

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

data['predicted_sentiments'] = data['post_text'].apply(predict_sentiment)

label_dict = {'negative': 0, 'neutral': 1, 'positive': 2}
data['opinion'] = data['opinion'].map(label_dict)

X_word_features = np.array(data['predicted_sentiments'].tolist())
X_train, X_test, y_train, y_test = train_test_split(X_pad, data['opinion'], test_size=0.2, random_state=42)

input_content = Input(shape=(100,), name='content_input')

embedding = Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=100, input_length=100)(input_content)
conv_layer = Conv1D(128, 5, activation='relu')(embedding)
pooling_layer_content = GlobalMaxPooling1D()(conv_layer)
dense = Dense(128, activation='relu')(pooling_layer_content)
output = Dense(3, activation='softmax')(dense)

model = Model(inputs=input_content, outputs=output)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(
    x=X_train,
    y=y_train,
    batch_size=32,
    epochs=5,
    validation_data=(X_test, y_test)
)

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

model.predict = predict_sentiment
save_model(model, "final_model.keras")

with open("predict_sentiment.pkl", "wb") as f:
    cloudpickle.dump(predict_sentiment, f)
'''
while True:
    user_input = input("Enter a post text (type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        print("Exiting...")
        break

    predictions = model.predict(user_input)
    predicted_class = np.argmax(predictions)
    predicted_sentiment = {v: k for k, v in label_dict.items()}[predicted_class]

    print(predictions)
    print(predicted_sentiment)

'''
'''
    # Print results
    print("\nSentiment Probabilities:")
    print(f"Negative: {negative_prob:.2f}")
    print(f"Neutral: {neutral_prob:.2f}")
    print(f"Positive: {positive_prob:.2f}")
    print()
'''