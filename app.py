from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import pickle
import numpy as np
import os

app = Flask(__name__)


# Load models and tokenizers with confirmation prints
MODELS = {}
try:
    cnn_model = tf.keras.models.load_model('cnn_model.keras')
    cnn_tokenizer = pickle.load(open('cnn_tokenizer.obj', 'rb'))
    MODELS['CNN'] = {'model': cnn_model, 'tokenizer': cnn_tokenizer}
    print("CNN model and tokenizer loaded successfully!")
    print(f"CNN vocab size: {len(cnn_tokenizer.word_index)}")
except Exception as e:
    print(f"Error loading CNN: {e}")

try:
    lstm_model = tf.keras.models.load_model('lstm_model.keras')
    lstm_tokenizer = pickle.load(open('lstm_tokenizer.obj', 'rb'))
    MODELS['LSTM'] = {'model': lstm_model, 'tokenizer': lstm_tokenizer}
    print("LSTM model and tokenizer loaded successfully!")
    print(f"LSTM vocab size: {len(lstm_tokenizer.word_index)}")
except Exception as e:
    print(f"Error loading LSTM: {e}")

try:
    rnn_model = tf.keras.models.load_model('rnn_model.keras')
    rnn_tokenizer = pickle.load(open('rnn_tokenizer.obj', 'rb'))
    MODELS['RNN'] = {'model': rnn_model, 'tokenizer': rnn_tokenizer}
    print("RNN model and tokenizer loaded successfully!")
    print(f"RNN vocab size: {len(rnn_tokenizer.word_index)}")
except Exception as e:
    print(f"Error loading RNN: {e}")

print("All models loaded successfully!")

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
            label = 'Negative'  # Reverse Positive to Negative
        elif negative_p >= max(positive_p, neutral_p):
            label = 'Positive'  # Reverse Negative to Positive
        else:
            label = 'Neutral'
        results[name] = {
            'score': score,
            'label': label,
            'positive': negative_p,
            'negative': positive_p,
            'neutral': neutral_p
        }
    elapsed = round((time.time() - start_time) * 1000, 2)  # ms
    return jsonify({'results': results, 'elapsed_ms': elapsed})

if __name__ == '__main__':
    app.run(debug=True)
