from __future__ import division
import unittest
from tqdm import tqdm
from utils.dataset import DataSet, segmentize_dataset, zip_segments
import features.features as FeatureFactory
from classifier import Classifier

def log_training_data_features(training_size, features):
    TRAINING_SIZE = 50
    dataset = DataSet()
    segments = segmentize_dataset(dataset)
    train_headlines, train_bodies, train_classifications = segments
    classifier = Classifier(train_headlines,
                            train_bodies,
                            train_classifications,
                            size=TRAINING_SIZE,
                            features=['co_occurence', 'n_grams', 'word_overlap']
                            )
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
