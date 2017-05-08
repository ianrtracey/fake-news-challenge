from __future__ import division
import unittest
from tqdm import tqdm
from utils.dataset import DataSet, segmentize_dataset, zip_segments
import utils.scorer as Scorer
import features.features as FeatureFactory
from classifier import Classifier
import flamegraph
import logging


def get_train_features(article, pbar):
    headline, body, classification


TRAINING_SIZE = 500
TESTING_SIZE = 1000

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
for entry in tqdm(entries[:TESTING_SIZE]):
    headline, body, classification = entry
    prediction = classifier.predict(headline, body)
    predictions.append(prediction) 
    test_classifications.append(classification)

hits = 0 
results = zip(predictions, test_classifications)
for result in results:
    p, tc = result
    if p == tc:
        hits += 1

print ( "Percentage correct: {0}%".format( float(hits) / float(len(results))) )
score = Scorer.report_score(test_classifications, predictions)
print ( score )

