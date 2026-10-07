import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentiment import NaiveBayesClassifier, tokenize

TEXTS = ["great movie loved it", "awesome and fun", "terrible movie hated it", "boring and awful"]
LABELS = ["positive", "positive", "negative", "negative"]


class TestTokenize(unittest.TestCase):
    def test_lowercase_and_stopwords(self):
        self.assertEqual(tokenize("The Movie WAS Great"), ["movie", "great"])

    def test_negation(self):
        self.assertEqual(tokenize("not good"), ["not", "NOT_good"])


class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = NaiveBayesClassifier().fit(TEXTS, LABELS)

    def test_predict(self):
        self.assertEqual(self.model.predict("loved this, so fun"), "positive")
        self.assertEqual(self.model.predict("awful and boring"), "negative")

    def test_probabilities_sum_to_one(self):
        self.assertAlmostEqual(sum(self.model.predict_proba("great").values()), 1.0)

    def test_unknown_words_do_not_crash(self):
        self.assertIn(self.model.predict("xyzzy qwerty"), {"positive", "negative"})

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.json")
            self.model.save(path)
            loaded = NaiveBayesClassifier.load(path)
        text = "fun movie"
        self.assertEqual(loaded.predict_proba(text), self.model.predict_proba(text))


if __name__ == "__main__":
    unittest.main()
