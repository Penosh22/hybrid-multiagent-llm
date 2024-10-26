import nltk
from nltk.corpus import stopwords
import spacy
import os

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    doc = nlp(text)
    cleaned_text = ' '.join([token.lemma_ for token in doc if token.text.lower() not in stop_words and token.is_alpha])
    return cleaned_text
