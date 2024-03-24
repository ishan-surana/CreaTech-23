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

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

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
dense = Dense(128, activation='relu')(concatenated_inputs)
output = Dense(3, activation='softmax')(dense)  # Output

# Define the model
model = Model(inputs=[input_content, input_upvotes], outputs=output)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Split the dataset into training and testing sets for X_pad
X_train_pad, X_test_pad, y_train, y_test = train_test_split(X_pad, data['opinion'], test_size=0.2, random_state=42)

# Split the dataset into training and testing sets for post_upvotes
X_train_upvotes, X_test_upvotes, _, _ = train_test_split(data['post_upvotes'], data['opinion'], test_size=0.2, random_state=42)

# Concatenate X_pad and post_upvotes for training and testing sets
X_train = [X_train_pad, X_train_upvotes]
X_test = [X_test_pad, X_test_upvotes]

# Train the model
model.fit(
    x=X_train,
    y=y_train,
    batch_size=32,
    epochs=5,
    validation_data=(X_test, y_test)
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

while True:
    # User input
    user_input = input("Enter a post text (type 'exit' to quit): ")

    # Check if user wants to exit
    if user_input.lower() == 'exit':
        print("Exiting...")
        break

    # Preprocess user input
    cleaned_input = clean_text(user_input)
    lemmatized_input = lemmatize_text(cleaned_input)
    tokenized_input = tokenizer.texts_to_sequences([lemmatized_input])
    padded_input = pad_sequences(tokenized_input, maxlen=100)

    # Get user input structural features
    structural_features = extract_structural_features(cleaned_input)
    structural_features = np.array(structural_features).reshape(1, -1)

    # Get user input upvotes (assuming it's normalized)
    user_upvotes = float(input("Enter the number of upvotes (normalized): "))

    # Concatenate user input pad and upvotes
    user_input_pad = [padded_input, np.array([[user_upvotes]])]

    # Predict sentiment
    predictions = model.predict(user_input_pad)

    # Get the sentiment probabilities
    negative_prob = predictions[0][0]
    neutral_prob = predictions[0][1]
    positive_prob = predictions[0][2]

    # Print results
    print("\nSentiment Probabilities:")
    print(f"Negative: {negative_prob:.2f}")
    print(f"Neutral: {neutral_prob:.2f}")
    print(f"Positive: {positive_prob:.2f}")
    print()