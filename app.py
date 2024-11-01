from flask import Flask, render_template, request, redirect, url_for, send_file
from tensorflow.keras.models import load_model
from translatepy import Translate
from gtts import gTTS
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = 'C:/PyCharmProjects/RiceDisease1/model/model.h5'
model = load_model(MODEL_PATH)

UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class_labels = [
    'Bacterial Leaf Blight', 'Bacterial Leaf Streak', 'Bacterial Panicle Blight', 'Blast',
    'Brown Spot', 'Dead Heart', 'Downy Mildew', 'Hispa', 'Normal', 'Tungro'
]

translator = Translate()

supported_languages = {
    'en': 'English',
    'hi': 'Hindi',
    'bn': 'Bengali',
    'ta': 'Tamil',
    'te': 'Telugu',
    'mr': 'Marathi',
    'gu': 'Gujarati',
    'kn': 'Kannada',
    'pa': 'Punjabi',
    'ml': 'Malayalam',
    'or': 'Odia'
}

gtts_supported_languages = ['en', 'hi', 'bn', 'ta', 'te', 'mr', 'gu', 'kn', 'pa', 'ml']

def preprocess_image(image_path):
    img = Image.open(image_path).resize((256, 256))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/')
def home():
    return render_template('index.html', languages=supported_languages)

@app.route('/upload', methods=['POST'])
def upload_image():
    if request.method == 'POST':
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.mp3')
        if os.path.exists(audio_path):
            os.remove(audio_path)

        file = request.files['file']
        language = request.form.get('language')

        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            image = preprocess_image(file_path)

            predictions = model.predict(image)
            predicted_class = np.argmax(predictions)
            result = class_labels[predicted_class]

            if language != 'en':
                result = translator.translate(result, destination_language=language).result

            if language not in gtts_supported_languages:
                language = 'en'

            tts = gTTS(text=result, lang=language)
            tts.save(audio_path)

            return render_template('index.html', result=result, image_file=file.filename, audio_file='result.mp3', languages=supported_languages)
    return redirect(url_for('home'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return send_file(str(file_path))

if __name__ == '__main__':
    app.run(debug=True)