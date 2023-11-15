from flask import Flask, render_template, request,flash, redirect, url_for, send_file,flash, make_response,Response
import os
import datetime
import requests
import io
import pandas as pd
import numpy as np
import moviepy.editor as mp
import speech_recognition as sr


app = Flask(__name__)
app.secret_key = 'dc9a0187c5297e94c26cdd32ffb3266eda502bc49a2874cb77961707ecda021f'
app.debug = True

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route('/')
def index():
    return render_template('index.html')

def convert_video_to_text(video_path):
    video = mp.VideoFileClip(video_path)
    audio_file = video.audio
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'audio.wav')
    audio_file.write_audiofile(audio_path)
    r = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        data = r.record(source)
    text_result = r.recognize_google(data)
    return text_result

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['video']

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file:
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(video_path)
        text_result = convert_video_to_text(video_path)
    return render_template('index.html', text_result=text_result)


if __name__ == '__main__':
  app.run(port=5000)