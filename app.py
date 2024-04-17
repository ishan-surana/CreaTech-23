from flask import Flask, jsonify, request, render_template
from tensorflow.keras.models import load_model
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

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']
    predictions = predict_sentiment(text)
    print(predictions)
    response = {
        'negative': predictions[0],
        'neutral': predictions[1],
        'positive': predictions[2]
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)