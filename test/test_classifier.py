from __future__ import division
import unittest
from tqdm import tqdm
from utils.dataset import DataSet, segmentize_dataset, zip_segments
import features.features as FeatureFactory
from classifier import Classifier

class TestClassifier(unittest.TestCase):

    def test_benchmark(self):
        dataset = DataSet()
        segments = segmentize_dataset(dataset)
        train_headlines, train_bodies, train_classifications = segments
        classifier = Classifier(train_headlines,
                                train_bodies,
                                train_classifications,
                                size=5000)

        test_data_set = DataSet(path="data",
                                bodies="train_bodies.csv",
                                stances="test_stances.csv")
        test_segments = segmentize_dataset(test_data_set)
        entries = zip_segments(test_segments)
        correct_hits = 0
        print ( 'Testing against test stances...')
        for entry in tqdm(entries):
            headline, body, classification = entry
            prediction = classifier.predict(headline, body)
            if classification == prediction:
                correct_hits += 1

        success_rate = (correct_hits / len(entries)) * 100
        print ("Test Results: {:10.4f}".format(success_rate))





if __name__ == '__main__':
    unittest.main()
