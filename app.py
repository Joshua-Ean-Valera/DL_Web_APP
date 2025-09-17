from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load models and tokenizers
MODELS = {
    'CNN': {
        'model': tf.keras.models.load_model('cnn_model.keras'),
        'tokenizer': pickle.load(open('cnn_tokenizer.obj', 'rb'))
    },
    'LSTM': {
        'model': tf.keras.models.load_model('lstm_model.keras'),
        'tokenizer': pickle.load(open('lstm_tokenizer.obj', 'rb'))
    },
    'RNN': {
        'model': tf.keras.models.load_model('rnn_model.keras'),
        'tokenizer': pickle.load(open('rnn_tokenizer.obj', 'rb'))
    }
}

MAXLEN = 100  # Adjust as needed

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    import time
    start_time = time.time()
    text = request.json.get('text', '')
    results = {}
    for name, items in MODELS.items():
        tokenizer = items['tokenizer']
        model = items['model']
        seq = tokenizer.texts_to_sequences([text])
        padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=MAXLEN)
        pred = model.predict(padded)
        score = float(pred[0][0])
        # Calculate probabilities for each class
        positive = score
        negative = 1 - score
        neutral = 1 - abs(score - 0.5) * 2
        total = positive + negative + neutral
        positive_p = round(positive / total * 100, 2)
        negative_p = round(negative / total * 100, 2)
        neutral_p = round(neutral / total * 100, 2)
        if positive_p >= max(negative_p, neutral_p):
            label = 'Positive'
        elif negative_p >= max(positive_p, neutral_p):
            label = 'Negative'
        else:
            label = 'Neutral'
        results[name] = {
            'score': score,
            'label': label,
            'positive': positive_p,
            'negative': negative_p,
            'neutral': neutral_p
        }
    elapsed = round((time.time() - start_time) * 1000, 2)  # ms
    return jsonify({'results': results, 'elapsed_ms': elapsed})

if __name__ == '__main__':
    app.run(debug=True)
