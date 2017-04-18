from utils.terms import REFUTING_TERMS
from utils.ngram import get_n_gram
import utils.tokenizer as tokenizer
import collections


# NOTE: All features assume that the headline and body are tokenized, lemmatized, ext
def get_supported_features():
    return ['co_occurence', 'polarity', 'refuting', 'n_grams', 'word_overlap']


# assuming that the body and the headline are paired via BodyID
def get_refuting_feature(headline):
    result = [1 if term in headline else 0 for term in REFUTING_TERMS]
    return result

# returns (headline_polarity, body_polarity)
# NOTE: in the original baseline, each polarity result is %2 WHY?
def get_polarity_feature(headline, body):
    headline_polarity = sum([token in REFUTING_TERMS for token in headline])
    body_polarity = sum([token in REFUTING_TERMS for token in body])
    return (headline_polarity, body_polarity)

# count how many times a token from the headline appears in the body
# used for determining relevancy
def get_binary_co_occurence(headline, body):
    occurence_count = 0
    for word in body:
        if word in headline:
            occurence_count += 1

    return occurence_count

def n_gram_appears_early_in_text(ngram, body, threshold):
    # grabs the subset of the document that is considered
    # 'early' and then determines if the ngram appears before it
    return ngram in body[:threshold]

def get_n_grams_relevance(headline, body):
    headline_n_grams = get_n_gram(headline, 2)
    headline_n_grams_as_str = [' '.join(ng) for ng in headline_n_grams]
    body_text = ' '.join(body)
    n_gram_hits = 0
    n_gram_early_hits = 0
    n_gram_first_hits = 0
    for ngram in headline_n_grams_as_str:
        if ngram in body_text:
            n_gram_hits += 1
        if n_gram_appears_early_in_text(ngram, body_text, 255):
            n_gram_early_hits += 1
        if n_gram_appears_early_in_text(ngram, body_text, 100):
            n_gram_first_hits += 1
    return (n_gram_hits, n_gram_early_hits, n_gram_first_hits)


def get_word_overlap(headline, body):
    headline_set = set(headline)
    body_set = set(body)
    num_shared_words = len(headline_set.intersection(body_set))
    num_all_words = len(headline_set.union(body_set))
    return float(num_shared_words) / float(num_all_words)


def get_feature_set(headline, body):
    cleaned_headline = tokenizer.clean(headline)
    cleaned_body = tokenizer.clean(body)

    FeatureSet = collections.namedtuple('FeatureSet',
                    get_supported_features())
    co_occurence = get_binary_co_occurence(cleaned_headline, cleaned_body)
    polarity_score = get_polarity_feature(cleaned_headline, cleaned_body)
    refuting_score = get_refuting_feature(cleaned_headline)
    n_gram_hits = get_n_grams_relevance(cleaned_headline, cleaned_body)
    word_overlap = get_word_overlap(cleaned_headline, cleaned_body)

    feature_set = FeatureSet(co_occurence, polarity_score, refuting_score, n_gram_hits, word_overlap)
    return feature_set
