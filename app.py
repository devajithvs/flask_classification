import os
from flask import Flask, render_template, request
from flask import send_from_directory

import numpy as np
from keras.preprocessing.image import load_img
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input

app = Flask(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))
# UPLOAD_FOLDER = dir_path + '/uploads'
# STATIC_FOLDER = dir_path + '/static'
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
model = load_model(STATIC_FOLDER + '/' + 'model_design.h5')
# call model to predict an image

def api(full_path):
    img = load_img(full_path, target_size=(224, 224, 3))
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return model.predict(img)

# home page
@app.route('/')
def home():
   return render_template('index.html')


# procesing uploaded file and predict it
@app.route('/upload', methods=['POST','GET'])
def upload_file():

    if request.method == 'GET':
        return render_template('index.html')
    else:
        file = request.files['image']
        full_name = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(full_name)

        indices = {0: 'With Retinoic Acid', 1: 'Without Retinoic Acid'}
        result = api(full_name)

        predicted_class = np.asscalar(np.argmax(result, axis=1))
        accuracy = round(result[predicted_class] * 100, 2)
        label = indices[predicted_class]

    return render_template('predict.html', image_file_name = file.filename, label = label, accuracy = accuracy)


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
    app.debug = True
    app.run(debug=True)
    app.debug = True