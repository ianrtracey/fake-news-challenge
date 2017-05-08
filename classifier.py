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
            classifications = classifications[:size]
            articles = zip(th, tb, classifications)
            self.training_size = size
        else:
            articles = zip(train_headlines, train_bodies, classifications)

        print ( "[Classifier]: Collecting features..." )

        relatedness_features = []
        stance_features = []
        for article in tqdm(articles):
            headline, body, classification = article

            related_features = FeatureFactory.get_features_relatedness(headline, body)
            relatedness_features.append( (related_features, classification) )
            if classification != 'unrelated':
                stance_feature = FeatureFactory.get_features_stance(headline, body)
                if 'debug' in options:
                    logging.debug("{0} <> {1}\n".format(stance_feature, classification))
                stance_features.append( (stance_feature, classification) )

        features, classifications = zip(*relatedness_features)
        self.relatedness_classifier = self._build_relatedness_classifier(features, classifications)
        
        features, classifications = zip(*stance_features)
        self.stance_classifier = self._build_stance_classifier(features, classifications)

    def _build_stance_classifier(self, features, classifications):
        stance_classifier = svm.SVC()
        classifications_np = np.asarray(classifications)
        features_np = np.asarray(features)
        print ( "[classifier]: fitting training data...")
        stance_classifier.fit(features_np, classifications_np)
        print ( "[classifier]: done." ) 
        return stance_classifier



    def _build_relatedness_classifier(self, features, classifications):
        relatedness_classifier = svm.SVC()
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
        print ( "[classifier]: fitting training data...")
        relatedness_classifier.fit(features_np, classifications_np)
        print ( "[classifier]: done." ) 
        return relatedness_classifier


    def predict(self, headline, body):
        # determines whether it is related or unrelated
        features = FeatureFactory.get_features_relatedness(headline, body)
        prediction = self.relatedness_classifier.predict([features])
        if prediction == 'unrelated':
            return prediction

        stance_features = FeatureFactory.get_features_stance(headline, body)
        stance_prediction = self.stance_classifier.predict([stance_features])
        return stance_prediction
