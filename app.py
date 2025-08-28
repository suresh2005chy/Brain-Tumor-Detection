from flask import Flask, request, jsonify, render_template, url_for
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import os
import random
import sys
import pickle

import logging
import hashlib

# intialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# intialize flask app
app=Flask(__name__)

CNN = None  # global variable to hold the CNN model

def load_model():
    global CNN
    if CNN is not None:
        return  # model already loaded
    script_dir = os.path.dirname(__file__)
    model_json_path = os.path.join(script_dir, 'models', 'CNN_structure.json')

    with open(model_json_path, 'r') as json_file:
        model_json = json_file.read()
    try:
        # load model
        CNN = tf.keras.models.model_from_json(model_json)

        # load and set model weights
        weights_path = os.path.join(script_dir, 'models', 'CNN_weights.pkl')
        with open(weights_path, 'rb') as weights_file:
            weights = pickle.load(weights_file)
            CNN.set_weights(weights)

        # compile model
        CNN.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001), 
                    loss='categorical_crossentropy', 
                    metrics=['accuracy'])
    except Exception as e:
        logger.error(f"Error loading model: {e}")

# function for retrieving prediction from model given an image path
def get_model_prediction(image_path):
    load_model()
    try:
        # load and preprocess the image
        img = Image.open(image_path).resize((224, 224))
        # convert grayscale images to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_array = np.expand_dims(np.array(img), axis=0)
        
        # predict using the CNN model
        prediction = CNN.predict(img_array)
        
        # interpret the prediction
        predicted_index = np.argmax(prediction[0])
        class_labels = ['glioma', 'meningioma', 'no tumor', 'pituitary']
        predicted_class = class_labels[predicted_index]
        return predicted_class
    except Exception as e:
        logger.error(f"Error in get_model_prediction: {e}")
        return None

# load html template
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET', 'POST'])
def predict_fire():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        preds = get_model_prediction(file_path)
        return preds
    return None

if __name__ == '__main__':
    app.run(debug=False)
