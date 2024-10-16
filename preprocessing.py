import nltk
from nltk.corpus import stopwords
import spacy
import os

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

try:
    nlp = spacy.load(r"c:\Users\PENOSH YADAV\anaconda3\envs\mindyourstock310\lib\site-packages\en_core_web_sm")
except OSError:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def preprocess_text(text):
    doc = nlp(text)
    cleaned_text = ' '.join([token.lemma_ for token in doc if token.text.lower() not in stop_words and token.is_alpha])
    return cleaned_text
