from __future__ import division
import unittest
from tqdm import tqdm
from utils.dataset import DataSet, segmentize_dataset, zip_segments
import utils.scorer as Scorer
import features.features as FeatureFactory
from classifier import Classifier

class TestClassifier(unittest.TestCase):

    def test_benchmark(self):
        TRAINING_SIZE = 5000
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
        for entry in tqdm(entries):
            headline, body, classification = entry
            prediction = classifier.predict(headline, body)
            predictions.append(prediction)
            test_classifications.append(classification)

        # prints omatrix and score and returns it
        logging_results = []
        score = Scorer.report_score(test_classifications, predictions)
        logging_results.append(str(score) + "%")
        logging_results.append("==== USING ====")
        features_used = ", ".join(classifier.get_supported_features())
        logging_results.append("Features: {0}".format(features_used))
        logging_results.append("Training Size: {0}".format(TRAINING_SIZE))
        print ( "\n".join(logging_results) )


        # write out to log file
        with open("classifier_performance.log", "a") as LogFile:
            LogFile.write( "\n".join(logging_results) + "\n\n" )






if __name__ == '__main__':
    unittest.main()
