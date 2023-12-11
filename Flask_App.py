from flask import Flask, request, jsonify
from simpletransformers.classification import ClassificationModel
import re
import emoji
import pandas as pd


app = Flask(__name__)


# In[3]:


loaded_model = ClassificationModel('bert', './content/outputs/', num_labels=2, use_cuda=False)


# In[4]:


def strip_sent(x_labels):
  for i in range (len(x_labels)) :
    text = x_labels[i]
    if isinstance(text, str) and not pd.isna(text):
        words = text.strip().split()  # Tokenize the text
        stripped = [word for word in words if "#" not in word and "@" not in word and "pic" not in word and "com" not in word and "you" not in word and re.match(u'[^\u0900-\u097F]+[^\u0020-\u0039]+', word) is None and not emoji.is_emoji(word)]
        stripped = [word for word in stripped if len(word) > 1]
        # stripped = [word for word in stripped if word not in stopword_list]  # Remove stopwords
        x_labels[i] = " ".join(stripped)  # Join the tokens back into text
    else:
        x_labels[i] = ""  # Replace NaN or non-string values with an empty string
    return x_labels


# In[5]:


@app.route('/classify', methods=['POST'])
def classify():
    # Get the input text from the POST request
    data = request.get_json()
    input_text = data['text']

    # Preprocess the input text
    preprocessed_text = strip_sent([input_text])

    # Make predictions
    predictions, _ = loaded_model.predict([preprocessed_text])
    predicted_label = predictions[0]

    # Map prediction index to labels
    # label_mapping = {0: 'HOF', 1: 'NOT'}
    # predicted_label = label_mapping[predicted_label]

    # Return the result as JSON
    result = {'predicted_label': predicted_label}
    return jsonify(result)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)





