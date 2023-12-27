from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = load_model('emotion_detection_model.h5')
model.make_predict_function()

# Define the emotion classes
emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the uploaded image
        file = request.files['file']
        if file:
            # Save the image
            img_path = 'uploads/temp.jpg'
            file.save(img_path)

            # Preprocess the image
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # Make predictions
            predictions = model.predict(img_array)

            # Get the predicted emotion
            predicted_class = emotion_classes[np.argmax(predictions)]
            confidence = np.max(predictions)

            return jsonify({'result': predicted_class, 'confidence': float(confidence)})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
