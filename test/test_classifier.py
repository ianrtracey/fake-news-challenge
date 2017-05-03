from __future__ import division
import unittest
from tqdm import tqdm
from utils.dataset import DataSet, segmentize_dataset, zip_segments
import utils.scorer as Scorer
import features.features as FeatureFactory
from classifier import Classifier
import logging

class TestClassifier(unittest.TestCase):

    def test_partial_related_unrelated(self):
        TRAINING_SIZE = 1000
        TESTING_SIZE = 2000

        dataset = DataSet()
        segments = segmentize_dataset(dataset)
        train_headlines, train_bodies, train_classifications = segments
        classifier = Classifier(train_headlines,
                                train_bodies,
                                train_classifications,
                                size=TRAINING_SIZE) 


        test_data_set = DataSet(path="data",
                                bodies="train_bodies.csv",
                                stances="test_stances.csv")
        test_segments = segmentize_dataset(test_data_set)
        entries = zip_segments(test_segments)
        test_classifications = []
        predictions = []
        print ( 'Testing against test stances...')
        for entry in tqdm(entries[:TESTING_SIZE]):
            headline, body, classification = entry
            prediction = classifier.predict(headline, body)
            predictions.append(prediction)
            if classification == 'unrelated':
                test_classifications.append('unrelated')
            else:
                test_classifications.append('related')

        hits = 0
        results = zip(predictions, test_classifications)
        for result in results:
            prediction, actual = result
            if prediction == actual:
                hits += 1

        print ("Result: {0}%".format(float(hits) / float(TESTING_SIZE)))

if __name__ == '__main__':
    logging.basicConfig(filename="features.log", level=logging.DEBUG)
    unittest.main()
