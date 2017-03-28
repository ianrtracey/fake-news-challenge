from utils.terms import REFUTING_TERMS
from utils.ngram import get_n_gram
import utils.tokenizer as tokenizer
import collections


# NOTE: All features assume that the headline and body are tokenized, lemmatized, ext.

# assuming that the body and the headline are paired via BodyID
def get_refuting_feature(headline, body):
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

def get_n_grams_relevance(headline, body):
    headline_str = " ".join(headline)
    tri_grams = get_n_gram(body, 3)
    n_gram_hit_count = 0
    for gram in tri_grams:
        gram_str = " ".join(gram)
        if gram_str in headline_str:
            n_gram_hit_count += 1

    return n_gram_hit_count




def get_feature_set(headline, body):
    features = {}
    cleaned_headline = tokenizer.clean(headline)
    cleaned_body = tokenizer.clean(body)

    FeatureSet = collections.namedtuple('FeatureSet',
                    ['co_occurence', 'polarity', 'refuting', 'n_grams'])
    co_occurence = get_binary_co_occurence(cleaned_headline, cleaned_body)
    polarity_score = get_polarity_feature(cleaned_headline, cleaned_body)
    refuting_score = get_refuting_feature(cleaned_headline, cleaned_body)
    n_gram_hits = get_n_grams_relevance(cleaned_headline, cleaned_body)

    feature_set = FeatureSet(co_occurence, polarity_score, refuting_score, n_gram_hits)
    return feature_set











