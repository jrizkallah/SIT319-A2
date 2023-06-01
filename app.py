from flask import Flask, request, render_template, jsonify
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

model = load_model('C:/Users/Jamil/S308/2023T1/SIT319/Assignment2/HD task/model.h5')

@app.route('/')
def home():
    return '''
        <html>
        <body>
            <h1>Welcome to the Garbage Classification App</h1>
            <form action="/predict" method="post" enctype="multipart/form-data">
                <input type="file" name="file">
                <input type="submit" value="Predict">
            </form>
        </body>
        </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = Image.open(file.stream)
    img = img.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    pred = model.predict(img)
    classes = ['metal', 'paper', 'cardboard', 'glass', 'plastic']
    predicted_class = classes[np.argmax(pred)]
    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
