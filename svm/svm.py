import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

class SVM (object):

    def __init__(self, features, classifications):
        clf = svm.SVC(kernel='linear', C = 1.0)
        clf.fit(features, classifications)
        self.clf = clf

    def predict(self, features):
        return self.clf.predict(features)

