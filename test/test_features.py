import unittest
import utils.tokenizer as tokenizer
import features.features as FeatureFactory
from utils.dataset import DataSet
from utils.ngram import get_n_gram

class TestTokenizer(unittest.TestCase):

    def test_relevance(self):
        dataset = DataSet()
        for i in range(0, 10):
            headline = dataset.stances[i]
            body = dataset.getBody(headline)
            print ( headline )
            print ( body )
            feature_set = FeatureFactory.get_feature_set(headline['Headline'], body)
            print ( feature_set )

    def test_n_grams(self):
        dataset = DataSet()
        headline = dataset.stances[0]
        cleaned_headline = tokenizer.clean(headline['Headline'])
        print ( cleaned_headline )
        print ( get_n_gram(cleaned_headline, 3) )




if __name__ == '__main__':
    unittest.main()
