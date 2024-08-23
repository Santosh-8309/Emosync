from __future__ import division, print_function
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
from flask import Flask, request, render_template
import statistics as st

class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)  # Remove 'groups' if it exists
        super().__init__(*args, **kwargs)

# Define custom object mapping
custom_objects = {'DepthwiseConv2D': CustomDepthwiseConv2D}

# Load the model with custom objects once
model = load_model('final_model.h5', custom_objects=custom_objects)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index1.html")

@app.route('/camera', methods=['GET', 'POST'])
def camera():
    GR_dict = {0: (0, 255, 0), 1: (0, 0, 255)}
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    output = []
    cap = cv2.VideoCapture(0)
    
    i = 0
    while i <= 30:
        ret, img = cap.read()
        if not ret:
            break

        faces = face_cascade.detectMultiScale(img, 1.05, 5)
        
        for x, y, w, h in faces:
            face_img = img[y:y+h, x:x+w]
            resized = cv2.resize(face_img, (224, 224))
            reshaped = resized.reshape(1, 224, 224, 3) / 255
            predictions = model.predict(reshaped)

            max_index = np.argmax(predictions[0])
            emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'neutral', 'surprise')
            predicted_emotion = emotions[max_index]
            output.append(predicted_emotion)
            
            cv2.rectangle(img, (x, y), (x+w, y+h), GR_dict[1], 2)
            cv2.rectangle(img, (x, y-40), (x+w, y), GR_dict[1], -1)
            cv2.putText(img, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imshow('LIVE', img)
        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            break
        
        i += 1

    cap.release()
    cv2.destroyAllWindows()

    if output:
        final_output1 = st.mode(output)
    else:
        final_output1 = 'unknown'
    
    return render_template("buttons.html", final_output=final_output1)

@app.route('/templates/buttons', methods=['GET', 'POST'])
def buttons():
    return render_template("buttons.html")

@app.route('/movies/<emotion>', methods=['GET', 'POST'])
def movies(emotion):
    valid_emotions = {'surprise', 'angry', 'sad', 'disgust', 'happy', 'fear', 'neutral'}
    if emotion in valid_emotions:
        return render_template(f"movies{emotion.capitalize()}.html")
    else:
        return "Invalid emotion", 404

@app.route('/songs/<emotion>', methods=['GET', 'POST'])
def songs(emotion):
    valid_emotions = {'surprise', 'angry', 'sad', 'disgust', 'happy', 'fear', 'neutral'}
    if emotion in valid_emotions:
        return render_template(f"songs{emotion.capitalize()}.html")
    else:
        return "Invalid emotion", 404

@app.route('/templates/join_page.html', methods=['GET', 'POST'])
def join():
    return render_template("join_page.html")

if __name__ == "__main__":
    app.run(debug=True)
