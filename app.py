from flask import Flask, render_template, request,flash, redirect, url_for, send_file,flash, make_response,Response
import os
import datetime
import requests
import io
import pandas as pd
import numpy as np
import moviepy.editor as mp
import speech_recognition as sr
from summarizer import Summarizer,TransformerSummarizer
from transformers import T5Tokenizer, TFT5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util



app = Flask(__name__)
app.secret_key = 'dc9a0187c5297e94c26cdd32ffb3266eda502bc49a2874cb77961707ecda021f'
app.debug = True
modelsummary = TransformerSummarizer(transformer_type="XLNet",transformer_model_key="xlnet-base-cased")

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
tokenizer = T5Tokenizer.from_pretrained('SJ-Ray/Re-Punctuate')
model = TFT5ForConditionalGeneration.from_pretrained('SJ-Ray/Re-Punctuate')
model2 = SentenceTransformer('all-MiniLM-L6-v2')

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

def merge_paragraphs(paragraph1, paragraph2):
    sentences1 = [sentence.strip() for sentence in paragraph1.split('.') if sentence.strip()]
    sentences2 = [sentence.strip() for sentence in paragraph2.split('.') if sentence.strip()]

    embeddings1 = model2.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model2.encode(sentences2, convert_to_tensor=True)
    similarity_matrix = util.pytorch_cos_sim(embeddings1, embeddings2)
    merged_paragraph = ""
    for i in range(len(sentences1)):
        max_sim_idx = similarity_matrix[i].argmax().item()
        merged_paragraph += f"{sentences1[i]}. {sentences2[max_sim_idx]}. "

    return merged_paragraph.strip()

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
        inputs = tokenizer.encode("punctuate: " + text_result, return_tensors="tf")
        result = model.generate(inputs)
        decoded_output = tokenizer.decode(result[0], skip_special_tokens=True)
        para2="Good Morning."
        merged_paragraph = merge_paragraphs(decoded_output, para2)
        me=merged_paragraph
        summary = ''.join(modelsummary(me, min_length=5))
    return render_template('index.html', text_result=text_result,corrected=decoded_output,merged=merged_paragraph,summary=summary)


if __name__ == '__main__':
  app.run(port=5000)