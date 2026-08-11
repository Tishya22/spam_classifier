import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

@st.cache_resource
def setup_nltk():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)

setup_nltk()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

tfidf = pickle.load(open(BASE_DIR / 'vectorizer.pkl', 'rb'))
model = pickle.load(open(BASE_DIR / 'model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")




if st.button('Predict'):

    #1. preprocess the test data/user input
    transformed_sms=transform_text(input_sms)

    #2 vectorize thetest data
    vector_input = tfidf.transform([transformed_sms])

    #3 predict 

    result = model.predict(vector_input)[0]


    #4 display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")


# model is the trained classifier loaded from model.pkl.
# vector_input is the new SMS converted into TF-IDF features using the saved vectorizer.
# .predict() is a built-in sklearn method that uses the patterns learned during .fit()
# to classify the new input and returns the predicted class: 0 = Ham, 1 = Spam.

# predict() returns an array because sklearn can predict multiple samples at once.
# Since we pass only one SMS, [0] extracts its single prediction (0 = Ham, 1 = Spam).
# but if we were to pass multiple sms at once, we would receive a predicted array as example: array[0,1,1,0] but just accessing the 0th element tells us theprediction of the first sms passed 
