import re
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn import feature_extraction

def word_net_tag(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''

def tokenize_text(text):
    words = word_tokenize(text)
    result = []
    for word in words:
        if word.isalnum():
            result.append(word)
    return result

def stem_text(text):
    stemmedText = []
    stemmer = PorterStemmer()
    for word in text:
        stemmedText.append(stemmer.stem(word))
    return stemmedText

def lemmatize_text(tokenizedText):
    wnl = WordNetLemmatizer()
    lemmatizedText = []
    for word in tokenizedText:
        try:
            _, pos = pos_tag([word.lower()])[0]
            tag = word_net_tag(pos)
            if tag != '':
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
    result = remove_top_words(lemmatizedText)
    return tokenizedText 


