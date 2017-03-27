import unittest
import utils.tokenizer as tokenizer
from utils.dataset import DataSet

class TestTokenizer(unittest.TestCase):

    def test_tokenization(self):
        dataset = DataSet()
        text = dataset.articles[140]
        tokenizedArticle = tokenizer.tokenize_text(text)
        self.assertTrue(len(tokenizedArticle) > 0)

    def test_lemmatization(self):
        dataset = DataSet()
        text = dataset.articles[140]
        tokenizedArticle = tokenizer.tokenize_text(text)
        lemmatizedText = tokenizer.lemmatize_text(tokenizedArticle)
        self.assertTrue(len(lemmatizedText) > 0)

    def test_stop_word_removal(self):
        dataset = DataSet()
        text = dataset.articles[140]
        tokenizedArticle = tokenizer.tokenize_text(text)
        lemmatizedText = tokenizer.lemmatize_text(tokenizedArticle)
        removedStopWords = tokenizer.remove_stop_words(lemmatizedText)
        print(removedStopWords)




if __name__ == '__main__':
    unittest.main()
