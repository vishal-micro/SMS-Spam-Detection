import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize Porter Stemmer
ps = PorterStemmer()

# Function to preprocess text
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

# Load the vectorizer and model
with open('vectorizer.pkl', 'rb') as vect_file:
    tfidf = pickle.load(vect_file)
    # Check if tfidf is fitted, if not fit it again
    if not hasattr(tfidf, 'vocabulary_'):
        raise ValueError("TfidfVectorizer instance is not fitted!")
        
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Preprocess input
        transformed_sms = transform_text(input_sms)
        
        # Vectorize input using the pre-fitted vectorizer
        vector_input = tfidf.transform([transformed_sms])
        
        # Predict using the pre-trained model
        result = model.predict(vector_input)
        
        # Display prediction
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
