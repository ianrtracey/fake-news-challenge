import utils.tokenizer as tokenizer
from tqdm import tqdm
import features.features as FeatureFactory
from utils.dataset import DataSet
from utils.ngram import get_n_gram
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
        else:
            articles = zip(train_headlines, train_bodies)
        features = []
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

    def _get_features(self, headline, body):
        feature_set = FeatureFactory.get_feature_set(headline, body)
        ngram_hits = feature_set.n_grams[0]
        ngram_early_hits = feature_set.n_grams[1] 
        polarity_headline = feature_set.polarity[0]
        polarity_body = feature_set.polarity[1]
        feature = [feature_set.co_occurence, polarity_headline, polarity_body, ngram_hits, ngram_early_hits]
        return feature



    def predict(self, headline, body):
        features = self._get_features(headline, body)
        return self.classifier.predict(features)


