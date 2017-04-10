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
    for ngram in headline_n_grams_as_str:
        if ngram in body_text:
            n_gram_hits += 1
        if n_gram_appears_early_in_text(ngram, body_text, 255):
            n_gram_early_hits += 1
    return (n_gram_hits, n_gram_early_hits)


    # we might want to loop over n-grams from 0-10 or
    # something here in order to catch more exact terms
    tri_grams = get_n_gram(body, 2)
    n_gram_hit_count = 0
    for gram in tri_grams:
        gram_str = " ".join(gram)
        if gram_str in headline_str:
            n_gram_hit_count += 1

    return n_gram_hit_count


def get_feature_set(headline, body):
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
