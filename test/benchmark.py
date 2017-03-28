import unittest
import utils.tokenizer as tokenizer
import features.features as FeatureFactory
from utils.dataset import DataSet
from utils.ngram import get_n_gram
from svm.svm import SVM

class TestBenchmark(unittest.TestCase):


    def _stance_val(self, stance):
        if stance == 'unrelated':
            return 0
        return 1

    def test_svm_relevance(self):
        dataset = DataSet()
        features = []
        classifications = []
        for i in range(0, 100):
            headline = dataset.stances[i]
            body = dataset.getBody(headline)
            print ( headline )
            feature_set = FeatureFactory.get_feature_set(headline['Headline'], body)
            feature_sub_set = [feature_set.co_occurence, feature_set.n_grams]
            classification = self._stance_val(headline['Stance'])
            print( feature_sub_set, classification, headline['Stance'])
            features.append(feature_sub_set)
            classifications.append(classification)

        svm = SVM(features, classifications)
        print ( "SVM built and data fit")

        results = []
        for i in range(150, 250):
            headline = dataset.stances[i]
            body = dataset.getBody(headline)
            print ( headline )
            feature_set = FeatureFactory.get_feature_set(headline['Headline'], body)
            feature_sub_set = [feature_set.co_occurence, feature_set.n_grams]
            classification = self._stance_val(headline['Stance'])
            print( feature_sub_set, classification, headline['Stance'])
            prediction = svm.predict(feature_sub_set)

            result = (classification, prediction)
            results.append(result)
            print(result)







if __name__ == '__main__':
    unittest.main()
