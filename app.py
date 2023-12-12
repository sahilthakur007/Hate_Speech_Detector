
import numpy as np
import spacy
import mahaNLP
import  fasttext
import pickle
from simpletransformers.classification import ClassificationModel
import pandas as pd 
import re
import emoji
import torch
from mahaNLP.preprocess import Preprocess
from mahaNLP.tokenizer import Tokenize
# from pydub import AudioSegment 
from werkzeug.utils import secure_filename
import os
import speech_recognition as sr
r = sr.Recognizer()
from flask import Flask,request, url_for, redirect, render_template

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'

train_args={
        'overwrite_output_dir': True,
        'max_seq_length' : 512,
        'sliding_window' : True,
        'num_train_epochs' : 3,
        'labels_list' : ['NOT','HOF'],
        'train_batch_size': 16

    }
# loaded_model = None
# if __name__ == '__main__':
def startConvertion(path="",lang = 'mr-IN'):
    with sr.AudioFile(path) as source:
        print('Fetching File')
        audio_text = r.listen(source)
        # recoginize_() method will throw a request error if the API is unreachable, hence using exception handling
        try:
        
            # using google speech recognition
            print('Converting audio transcripts into text ...')
            text = r.recognize_google(audio_text, language=lang)
            # text_file = open("D:\Sem7\Mega Project\Output.txt", "a")
            # #write string to file
            # text_file.write(text)
            # #close file
            # text_file.close()
            print(text)
            return text 
    
        except:
            print('Sorry.. run again...')


def is_marathi_word(token):
    # A simple heuristic to check if a word is Marathi or not
    # You can customize this function for more accuracy
    return all(char >= 'ऀ' and char <= 'ॿ' for char in token.text)

def strip_sent(x_labels):
    for i in range (len(x_labels)) :
        text = x_labels[i]
        if isinstance(text, str) and not pd.isna(text):
            words = text.strip().split()  # Tokenize the text
            stripped = [word for word in words if "#" not in word and "@" not in word and "pic" not in word and "com" not in word and "you" not in word and re.match(u'[^\u0900-\u097F]+[^\u0020-\u0039]+', word) is None and not emoji.is_emoji(word)]
            stripped = [word for word in stripped if len(word) > 1]
            x_labels[i] = " ".join(stripped)  # Join the tokens back into text
        else:
            x_labels[i] = ""  # Replace NaN or non-string values with an empty string
        return x_labels

nlp = spacy.blank("mr")

def preprocessing(text):
    tokenizer = Tokenize()
    tokenized = tokenizer.word_tokenize(text)
    tokenized_string = " ".join(tokenized)
    tokenized = nlp(tokenized_string)
    marathi_text = []
    for word in tokenized:
        if is_marathi_word(word)==True:
            marathi_text.append(word)
    stopword_removed = []
    for word in marathi_text:
        if word.is_stop==False:
            stopword_removed.append(word.text)
    return " ".join(stopword_removed)
@app.route('/')
def hello_world():
    return render_template('home.html')

@app.route("/text-detector")
def getTextDetector():
    return render_template("text_detector.html")

@app.route("/audio-detector")
def getAudioDetector():
    return render_template("audio_detector.html")

@app.route('/detect-text',methods=["POST"])
def get_text():
    if request.method == "POST":
        text = request.form.get("text")
        processesText = strip_sent([text])[0]
        loaded_model = ClassificationModel('bert', './models/content/outputs', num_labels=2,use_cuda=False, args={'fp16': False})
        predictions, raw_outputs = loaded_model.predict([
        [processesText],
        ])

    # Map prediction index to labels
        label_mapping = {0: 'HOF', 1: 'NOT'}
        predicted_label = predictions[0]
        result = ""
        if predicted_label=='NOT':
            result = "No Hate"
        else:
            result = "Hate"
        return render_template('result.html',result=result,text=text)
      

    return render_template('text_detector.html')



@app.route('/detect-audio',methods=["POST"])
def get_audio():
    if request.method == "POST":
       audio = request.files['audio']
       audio = request.files.get('audio')
       if audio:
           print(audio)
        #    print("hello")
           audio.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(audio.filename))) # 
           
           text = startConvertion("./static/"+audio.filename)
           print(text)
           processesText = strip_sent([text])[0]
           loaded_model = ClassificationModel('bert', './models/content/outputs', num_labels=2,use_cuda=False, args={'fp16': False})

    #  loaded_model = ClassificationModel('bert', './models/content/outputs', num_labels=2,use_cuda=False, args={'fp16': False})
           predictions, raw_outputs = loaded_model.predict([
           [processesText],])

    # Map prediction index to labels
           label_mapping = {0: 'HOF', 1: 'NOT'}
           predicted_label = predictions[0]
           result = ""
           if predicted_label=='NOT':
             result = "No Hate"
           else:
             result = "Hate"
           return render_template('result.html',result=result,text=text)

    return render_template('audio_detector.html')
if __name__ == '__main__':
    app.run(debug=True,use_reloader=False)
