from __future__ import division
import unittest
from tqdm import tqdm
from utils.dataset import DataSet, segmentize_dataset, zip_segments
import utils.scorer as Scorer
import features.features as FeatureFactory
from classifier import Classifier
import time
import logging

logging.basicConfig(filename='mismatched.log', level=logging.DEBUG)
TRAINING_SIZE = 2000
TESTING_SIZE = 1000

dataset = DataSet()
segments = segmentize_dataset(dataset)
train_headlines, train_bodies, train_classifications = segments
start = time.time()
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
stance_features = []
predictions = []
for entry in tqdm(entries[:TESTING_SIZE]):
    headline, body, classification = entry
    prediction = classifier.predict(headline, body)
    predictions.append(prediction) 
    if prediction != classification:
        logging.debug("Headline: {0}\n".format(headline))
        logging.debug("Body: {0}\n".format(body))
        logging.debug("correct: {0}, predicted: {1}\n\n\n".format(classification, prediction))

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

