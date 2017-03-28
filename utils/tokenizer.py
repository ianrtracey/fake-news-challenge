import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn import feature_extraction

def tokenize_text(text):
    cleaned_text = " ".join(re.findall(r'\w+', text, flags=re.UNICODE)).lower()
    return word_tokenize(cleaned_text)

def lemmatize_text(tokenizedText):
    wnl = WordNetLemmatizer()
    lemmatizedText = [wnl.lemmatize(word).lower() for word in tokenizedText]
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


