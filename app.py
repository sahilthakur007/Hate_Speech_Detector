
import numpy as np
import spacy
import mahaNLP
import  fasttext
import pickle
from mahaNLP.preprocess import Preprocess
from mahaNLP.tokenizer import Tokenize
from flask import Flask,request, url_for, redirect, render_template
app = Flask(__name__)

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
    return render_template('detector.html')

@app.route('/detect',methods=["POST"])
def get_text():
    if request.method == "POST":
       text = request.form.get("text")
       processedText  = preprocessing(text)

       model = fasttext.load_model('fasttext.bin')
       hatedetecter=pickle.load(open('hate_detector.pkl','rb'))
       output = hatedetecter.predict([model.get_sentence_vector
       (processedText)])[0]
       print(output)
       print(text)
       result = ""
       if output==1:
           result = "No Hate"
       else:
           result = "Hate"
       return render_template('detector.html',result=result)

    return render_template('detector.html')

if __name__ == '__main__':
    app.run()
