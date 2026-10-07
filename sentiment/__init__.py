"""A tiny Naive Bayes sentiment classifier, built from scratch."""
from .model import NaiveBayesClassifier
from .text import tokenize

__all__ = ["NaiveBayesClassifier", "tokenize"]
