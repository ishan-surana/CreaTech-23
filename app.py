from flask import Flask, jsonify, request, render_template
from tensorflow.keras.models import load_model
from model_new_trained import predict_sentiment

model = load_model('final_model.h5',compile=False)

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