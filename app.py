from flask import Flask, jsonify, request, render_template
from tensorflow.keras.models import load_model
import cloudpickle

model = load_model('final_model.h5',compile=False)

# Load custom prediction function
with open("predict_sentiment.pkl", "rb") as f:
    predict_sentiment = cloudpickle.load(f)

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