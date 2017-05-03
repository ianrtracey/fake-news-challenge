import utils.tokenizer as tokenizer
from tqdm import tqdm
import features.features as FeatureFactory
from utils.dataset import DataSet
from utils.ngram import get_n_gram
from utils.utils import flatten
from sklearn import svm
import numpy as np
import logging

class Classifier(object):
    def __init__(self, train_headlines, train_bodies, classifications, **options):
        if 'size' in options:
            size = options['size']
            th = train_headlines[:size]
            tb = train_bodies[:size]
            articles = zip(th, tb)
            classifications = classifications[:size]
            self.training_size = size
        else:
            articles = zip(train_headlines, train_bodies)
            self.training_size = len(articles)

        features = []
        if 'features' in options:
            restricted_features = True
            self.supported_features = options['features']
        else:
            restricted_features = False

        print ( "[Classifier]: Collecting features..." )

        i = 0
        for article in tqdm(articles):
            logging.debug("stance: {}".format(classifications[i]))
            headline = article[0]
            body = article[1]
            feature = FeatureFactory.get_features_related(headline, body)
            i += 1
            features.append(feature)

        print ( "[Classifier]: Done." )
        classifier = svm.SVC()
        # sanitize classifications for bi-classification (related/unrelated)
        fc_pairs = zip(features, classifications)
        related_unrelated_pairs = []
        for pair in fc_pairs:
            feature, classification = pair
            if classification == 'unrelated':
                related_unrelated_pairs.append( (feature, 'unrelated') )
            else:
                related_unrelated_pairs.append( (feature, 'related') )

        related_unrelated_features, related_unrelated_classifications = zip(*related_unrelated_pairs)

        classifications_np = np.asarray(related_unrelated_classifications)
        features_np = np.asarray(related_unrelated_features)
        print ( "[Classifier]: Fitting training data...")
        classifier.fit(features_np, classifications_np)
        print ( "[Classifier]: Done." ) 
        self.classifier = classifier

    def _feature_switch(self, feature_name):
        return 'supported_features' not in dir(self) or feature_name in self.supported_features




    def _get_features(self, headline, body):
        feature_set = FeatureFactory.get_feature_set(headline, body)
        # TODO: compute n-grams from 2..6
        features = []
        ngram_hits = feature_set.n_grams[0]
        ngram_early_hits = feature_set.n_grams[1] 
        n_gram_first_hits = feature_set.n_grams[2]
        features.append(ngram_hits)
        features.append(ngram_early_hits)
        features.append(n_gram_first_hits)

        polarity_headline = feature_set.polarity[0]
        polarity_body = feature_set.polarity[1]
        features.append(polarity_headline)
        features.append(polarity_body)

        word_overlap = feature_set.word_overlap
        features.append(word_overlap)

        co_occurence = feature_set.co_occurence
        features.append(co_occurence)

        refuting_words_in_headline = feature_set.refuting
        features.append(refuting_words_in_headline)

        return features

    def get_supported_features(self):
        if 'supported_features' in dir(self):
            return self.supported_features
        return FeatureFactory.get_supported_features()

    def get_training_size(self):
        return self.training_size

    def predict(self, headline, body):
        features = FeatureFactory.get_features_related(headline, body)
        return self.classifier.predict([features])
