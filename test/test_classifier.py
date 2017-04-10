from __future__ import division
import unittest
from tqdm import tqdm
from utils.dataset import DataSet, segmentize_dataset, zip_segments
import utils.scorer as Scorer
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
        test_classifications = []
        predictions = []
        print ( 'Testing against test stances...')
        for entry in tqdm(entries):
            headline, body, classification = entry
            prediction = classifier.predict(headline, body)
            predictions.append(prediction)
            test_classifications.append(classification)

        Scorer.report_score(test_classifications, predictions)






if __name__ == '__main__':
    unittest.main()
