from tensorflow.keras.models import load_model
from flask import Flask, render_template, request
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image

import librosa
import librosa.display

app = Flask(__name__)


model = load_model('modelpkm.h5')
# Parameters
input_size = (150, 150)

# define input shape
channel = (3,)
input_shape = input_size + channel

# define labels
labels = ['Negatif', 'Positif']

model.make_predict_function()


def preprocess(img_path, input_size):
    nimg = img_path.convert('RGB').resize(input_size, resample=0)
    img_arr = (np.array(nimg))/255
    return img_arr


def reshape(imgs_arr):
    return np.stack(imgs_arr, axis=0)


def predict_label(img_path):
    i = image.load_img(img_path, target_size=(150, 150))
    x = preprocess(i, input_size)
    x = reshape([x])
    y = model.predict(x)
    return labels[np.argmax(y)], round(np.max(y*100))

# routes


@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")


@app.route("/about")
def about_page():
    return "Web masih dalam pengembangan"


@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        sound = request.files['my_image']
        sound_path = "static/sound/" + sound.filename
        sound.save(sound_path)

        f = sound_path
        y, _ = librosa.load(f, sr=44100)
        signal = y[0:int(0.9 * _)]
    # get mel-spectogram
        S = librosa.feature.melspectrogram(signal)
        S_DB = librosa.power_to_db(S, ref=np.max)
        plt.figure(figsize=(5, 4))
        librosa.display.specshow(S_DB)
        plt.tight_layout()
        img_path = "static/image/" + sound.filename + ".png"
        plt.savefig(img_path, bbox_inches='tight', pad_inches=0)

        p = predict_label(img_path)

    return render_template("index.html", prediction=p, img_path=img_path)


if __name__ == '__main__':
    #app.debug = True
    app.run(debug=True)
