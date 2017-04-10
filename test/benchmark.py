from __future__ import division
import unittest
import utils.tokenizer as tokenizer
import features.features as FeatureFactory
import matplotlib.pyplot as plt
from utils.dataset import DataSet
from utils.ngram import get_n_gram
from sklearn import svm
import numpy as np

class TestBenchmark(unittest.TestCase):


    def _stance_val(self, stance):
        if stance == 'unrelated':
            return 0
        return 1

    def test_features_occurence_and_bi_grams(self):
        headline = "trump launches missiles at syrian air force base"
        body = "this week trump launch missiles at a base off the western coast of syria. We are all very confused by this"
        feature_set = FeatureFactory.get_feature_set(headline, body)
        self.assertTrue(feature_set.co_occurence == 4)
        hits, early_hits = feature_set.n_grams 
        self.assertTrue(hits == 2)
        self.assertTrue(early_hits == 2)


    def test_svm(self):
        '''
            Used to test the accuracy of related/unrelated headlines against bodies
        '''
        dataset = DataSet()
        features = []
        classifications = []
        for i in range(0, 1000):
            headline = dataset.stances[i]
            body = dataset.getBody(headline)
            feature_set = FeatureFactory.get_feature_set(headline['Headline'], body)
            ngram_hits = feature_set.n_grams[0]
            ngram_early_hits = feature_set.n_grams[1] 
            feature_sub_set = [feature_set.co_occurence, ngram_hits, ngram_early_hits]
            classification = headline['Stance']
            features.append(feature_sub_set)
            classifications.append(classification)

        classifier = svm.SVC()
        featuresArray = np.asarray(features)
        classificationArray = np.asarray(classifications)
        classifier.fit(featuresArray, classificationArray)

        print ( "SVM built and data fit")
        results = []
        correct_hits = 0
        for i in range(1500, 2500):
            headline = dataset.stances[i]
            body = dataset.getBody(headline)
            feature_set = FeatureFactory.get_feature_set(headline['Headline'], body)

            ngram_hits = feature_set.n_grams[0]
            ngram_early_hits = feature_set.n_grams[1] 
            feature_sub_set = [feature_set.co_occurence, ngram_hits, ngram_early_hits]
            classification = headline['Stance']
            prediction = classifier.predict(feature_sub_set)
            if classification == prediction[0]:
                correct_hits += 1

        print "Percent Correct for dual-classification", (correct_hits / 1000)


    def test_multi_classification(self):
        '''
            Used to test the accuracy of related/unrelated headlines against bodies
        '''
        dataset = DataSet()
        features = []
        classifications = []
        for i in range(0, 2500):
            headline = dataset.stances[i]
            body = dataset.getBody(headline)
            feature_set = FeatureFactory.get_feature_set(headline['Headline'], body)
            ngram_hits = feature_set.n_grams[0]
            ngram_early_hits = feature_set.n_grams[1] 
            polarity_headline = feature_set.polarity[0]
            polarity_body = feature_set.polarity[1]
            feature_sub_set = [feature_set.co_occurence, polarity_headline, polarity_body,  ngram_hits, ngram_early_hits]
            classification = headline['Stance']
            features.append(feature_sub_set)
            classifications.append(classification)

        classifier = svm.SVC()
        featuresArray = np.asarray(features)
        classificationArray = np.asarray(classifications)
        classifier.fit(featuresArray, classificationArray)

        print ( "SVM built and data fit")
        results = []
        correct_hits = 0
        for i in range(2500, 7500):
            headline = dataset.stances[i]
            body = dataset.getBody(headline)
            feature_set = FeatureFactory.get_feature_set(headline['Headline'], body)

            ngram_hits = feature_set.n_grams[0]
            ngram_early_hits = feature_set.n_grams[1] 
            polarity_headline = feature_set.polarity[0]
            polarity_body = feature_set.polarity[1]
            feature_sub_set = [feature_set.co_occurence, polarity_headline, polarity_body,  ngram_hits, ngram_early_hits]
            classification = headline['Stance']
            prediction = classifier.predict(feature_sub_set)
            if classification == prediction[0]:
                correct_hits += 1

        print "Percent correct for multi-classification", (correct_hits / 5000)





if __name__ == '__main__':
    unittest.main()
