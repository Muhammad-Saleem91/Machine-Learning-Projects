import streamlit as st
import numpy as np
import pandas as pd
import re
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# loading model and dependencies
model = pickle.load(open('fake-msg.pkl','rb'))
vectorizer = pickle.load(open('vect-form.pkl','rb'))
stem_proc = PorterStemmer()

# helper Functions
def stemming(content):
    stemmed_content = re.sub('[^a-z A-z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()

    # applying stemming to each word which is not stopwords after removing numbers and other characters

    stemmed_content = [stem_proc.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content



def news_detection(author, title,text):
    data = {
        'title' : [title],
        'author': [author],
        'text' : [text]
    }

    df = pd.DataFrame(data)
    df['main-content'] = df['title'] + ' ' + df['author']
    df['main-content'] = df['main-content'].apply(stemming)
    vectorizedContent = vectorizer.transform(df['main-content'].values)
    prediction = model.predict(vectorizedContent)
    return prediction





# creating web app
st.title('Fake News Classification App')
st.subheader("Input the News content below")

title = st.text_input('Enter Title of the News:')
author = st.text_input('Enter the Name of the Author:')
text = st.text_area('Enter Content of the News:')

predict_button = st.button('Check')

if predict_button:
    reliability = news_detection(title, author, text)
    if reliability == 0:
        st.success('News is Reliable.')
    elif reliability == 1:
        st.error('News is non reliable.')
    
