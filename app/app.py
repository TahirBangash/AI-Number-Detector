from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import os

app = Flask(__name__)
model_path = os.path.join(os.path.dirname(__file__), '../saved_model/mnist_model.keras')
model = load_model(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    print(request.files)
    print(request.form)
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    file = request.files['image']
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28)) / 255.0
    img = img.reshape(1, 28, 28, 1)
    prediction = model.predict(img)
    predicted_number = np.argmax(prediction[0])
    return jsonify({'number': int(predicted_number)})

if __name__ == '__main__':
    app.run(debug=True)
