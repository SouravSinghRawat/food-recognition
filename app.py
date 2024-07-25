# app.py
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
import pandas as pd

app = Flask(__name__)
model = load_model('food_recognition_model.h5')

calorie_info = pd.read_csv('D:/Calories/dataset/calorie_info.csv')
food_categories = list(calorie_info['FoodItems'])

@app.route('/')
def index():
    return render_template('index.html', food_categories=food_categories)

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            raise ValueError('No file part')

        file = request.files['file']

        if file.filename == '':
            raise ValueError('No selected file')

        if file:
            img_content = file.read()
            img = image.load_img(io.BytesIO(img_content), target_size=(224, 224))

            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions[0])
            predicted_food_name = food_categories[predicted_class_index]
            predicted_calories = int(calorie_info.iloc[predicted_class_index]['Calories'])

            return jsonify({'predicted_food_name': predicted_food_name, 'predicted_calories': predicted_calories})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
