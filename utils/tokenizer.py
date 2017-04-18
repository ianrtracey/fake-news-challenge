import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn import feature_extraction

def tokenize_text(text):
    words = word_tokenize(text)
    return words

def lemmatize_text(tokenizedText):
    wnl = WordNetLemmatizer()
    lemmatizedText = []
    for word in tokenizedText:
        try:
            lemma = wnl.lemmatize(word).lower()
            lemmatizedText.append(lemma)
        except:
            continue

    return lemmatizedText

def remove_stop_words(tokenizedText):
    stop_words = feature_extraction.text.ENGLISH_STOP_WORDS
    text = [word for word in tokenizedText if word not in stop_words]
    return text

def clean(text):
    tokenizedText = tokenize_text(text)
    lemmatizedText = lemmatize_text(tokenizedText)
    result = remove_stop_words(lemmatizedText)
    return result


