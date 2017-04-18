import re
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn import feature_extraction

def word_net_tag(pos_tag):
    wordnet_tag = {'NN':'n','JJ':'a','VB':'v','RB':'r'}
    if pos_tag not in wordnet_tag.keys():
        return None
    return wordnet_tag[pos_tag]

def tokenize_text(text):
    words = word_tokenize(text)
    result = []
    for word in words:
        if word.isalnum():
            result.append(word)
    return result

def lemmatize_text(tokenizedText):
    wnl = WordNetLemmatizer()
    lemmatizedText = []
    for word in tokenizedText:
        try:
            _, pos = pos_tag([word.lower()])[0]
            tag = word_net_tag(pos)
            if tag is not None:
                lemma = wnl.lemmatize(word.lower(), pos=tag)
            else:
                lemma = wnl.lemmatize(word.lower())
            lemmatizedText.append(lemma)
        except:
            import pdb; pdb.set_trace()
            continue

    return lemmatizedText

def remove_stop_words(tokenizedText):
    all_stop_words = feature_extraction.text.ENGLISH_STOP_WORDS
    stop_words = set(all_stop_words) - set(['not'])
    text = [word for word in tokenizedText if word not in stop_words]
    return text

def clean(text):
    tokenizedText = tokenize_text(text)
    lemmatizedText = lemmatize_text(tokenizedText)
    result = remove_stop_words(lemmatizedText)
    return result


