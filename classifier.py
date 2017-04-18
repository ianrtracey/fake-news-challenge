import utils.tokenizer as tokenizer
from tqdm import tqdm
import features.features as FeatureFactory
from utils.dataset import DataSet
from utils.ngram import get_n_gram
from utils.utils import flatten
from sklearn import svm
import numpy as np

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
            self.trainin_size = len(articles)

        features = []
        if 'features' in options:
            restricted_features = True
            self.supported_features = options['features']
        else:
            restricted_features = False

        print ( "[Classifier]: Collecting features..." )

        for article in tqdm(articles):
            headline = article[0]
            body = article[1]
            feature = self._get_features(headline, body)
            features.append(feature)

        print ( "[Classifier]: Done." )
        classifier = svm.SVC()
        features_np = np.asarray(features)
        classifications_np = np.asarray(classifications)
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
        if self._feature_switch('n_grams'): 
            ngram_hits = feature_set.n_grams[0]
            ngram_early_hits = feature_set.n_grams[1] 
            n_gram_first_hits = feature_set.n_grams[2]
            features.append(ngram_hits)
            features.append(ngram_early_hits)
            features.append(n_gram_first_hits)
        if self._feature_switch('polarity'): 
            polarity_headline = feature_set.polarity[0]
            polarity_body = feature_set.polarity[1]
            features.append(polarity_headline)
            features.append(polarity_body)
        if self._feature_switch('word_overlap'): 
            word_overlap = feature_set.word_overlap
            features.append(word_overlap)
        if self._feature_switch('co_occurence'): 
            co_occurence = feature_set.co_occurence
            features.append(co_occurence)
        if self._feature_switch('refuting'):
            refuting_words_in_headline = feature_set.refuting
            # features.append(refuting_words_in_headline)

        return features

    def get_supported_features(self):
        if 'supported_features' in dir(self):
            return self.supported_features
        return FeatureFactory.get_supported_features()

    def get_training_size(self):
        return self.training_size

    def predict(self, headline, body):
        features = self._get_features(headline, body)
        return self.classifier.predict(features)
