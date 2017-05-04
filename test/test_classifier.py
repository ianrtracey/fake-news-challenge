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
        TRAINING_SIZE = 300
        TESTING_SIZE = 400

        dataset = DataSet()
        segments = segmentize_dataset(dataset)
        train_headlines, train_bodies, train_classifications = segments
        classifier = Classifier(train_headlines,
                                train_bodies,
                                train_classifications,
                                size=TRAINING_SIZE,
                                debug=True) 


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
            test_classifications.append(classification)
        
        score = Scorer.report_score(test_classifications, predictions)
        print ( score )


if __name__ == '__main__':
    logging.basicConfig(filename="features.log", level=logging.DEBUG)
    unittest.main()
