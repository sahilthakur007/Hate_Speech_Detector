
import numpy as np
import spacy
import mahaNLP
import  fasttext
import pickle
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
    #    audio = request.files['audio']
    #    audio = request.files.get('audio')
    #    if audio:
    #        print(audio)
    #        print("hello")
    #        audio.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(audio.filename))) # 
    #        startConvertion("./static/"+audio.filename)

         
           
    #    audio.export("output.mp3", format="mp3")
    #    print(audio)
       processedText  = preprocessing(text)

       model = fasttext.load_model('./models/fasttext.bin')
       hatedetecter=pickle.load(open('./models/hate_detector.pkl','rb'))
       output = hatedetecter.predict([model.get_sentence_vector
       (processedText)])[0]
    #    print(output)
    #    print(text)
       result = ""
       if output==1:
           result = "No Hate"
       else:
           result = "Hate"
       return render_template('textdetector.html',result=result,text=text)

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
         
           
    #    audio.export("output.mp3", format="mp3")
    #    print(audio)
           processedText  = preprocessing(text)

           model = fasttext.load_model('./models/fasttext.bin')
           hatedetecter=pickle.load(open('./models/hate_detector.pkl','rb'))
           output = hatedetecter.predict([model.get_sentence_vector
           (processedText)])[0]
    #    print(output)
    #    print(text)
           result = ""
           if output==1:
                result = "No Hate"
           else:
                result = "Hate"
           return render_template('textdetector.html',result=result,text=text)

    return render_template('text_detector.html')
if __name__ == '__main__':
    app.run()
