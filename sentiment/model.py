import json
import math
from collections import Counter, defaultdict

from .text import tokenize


class NaiveBayesClassifier:
    """Multinomial Naive Bayes with Laplace smoothing."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.class_counts: Counter = Counter()
        self.word_counts: dict[str, Counter] = defaultdict(Counter)
        self.vocab: set[str] = set()

    def fit(self, texts: list[str], labels: list[str]) -> "NaiveBayesClassifier":
        for text, label in zip(texts, labels):
            self.class_counts[label] += 1
            for tok in tokenize(text):
                self.word_counts[label][tok] += 1
                self.vocab.add(tok)
        return self

    def _log_scores(self, text: str) -> dict[str, float]:
        total_docs = sum(self.class_counts.values())
        v = len(self.vocab)
        tokens = tokenize(text)
        scores = {}
        for label, n_docs in self.class_counts.items():
            total_words = sum(self.word_counts[label].values())
            score = math.log(n_docs / total_docs)
            for tok in tokens:
                count = self.word_counts[label][tok]
                score += math.log((count + self.alpha) / (total_words + self.alpha * v))
            scores[label] = score
        return scores

    def predict_proba(self, text: str) -> dict[str, float]:
        scores = self._log_scores(text)
        top = max(scores.values())
        exps = {k: math.exp(s - top) for k, s in scores.items()}
        z = sum(exps.values())
        return {k: e / z for k, e in exps.items()}

    def predict(self, text: str) -> str:
        probs = self.predict_proba(text)
        return max(probs, key=probs.get)

    def explain(self, text: str, label: str, top_n: int = 5) -> list[tuple[str, float]]:
        """Words in the text that pushed hardest toward `label`."""
        others = [c for c in self.class_counts if c != label]
        v = len(self.vocab)

        def logp(tok, c):
            total = sum(self.word_counts[c].values())
            return math.log((self.word_counts[c][tok] + self.alpha) / (total + self.alpha * v))

        weights = {}
        for tok in set(tokenize(text)):
            weights[tok] = logp(tok, label) - max(logp(tok, o) for o in others)
        return sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:top_n]

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"alpha": self.alpha,
                       "class_counts": self.class_counts,
                       "word_counts": self.word_counts}, f)

    @classmethod
    def load(cls, path: str) -> "NaiveBayesClassifier":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        m = cls(alpha=data["alpha"])
        m.class_counts = Counter(data["class_counts"])
        for label, counts in data["word_counts"].items():
            m.word_counts[label] = Counter(counts)
            m.vocab.update(counts)
        return m
