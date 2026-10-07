"""Train the model, report held-out accuracy, and save it."""
import csv
import random
from pathlib import Path

from sentiment import NaiveBayesClassifier

ROOT = Path(__file__).parent


def load(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r["text"] for r in rows], [r["label"] for r in rows]


def evaluate(texts, labels, rounds=20, test_size=0.2, seed=42):
    """Average accuracy over several random 80/20 train/test splits."""
    rng = random.Random(seed)
    idx = list(range(len(texts)))
    scores = []
    for _ in range(rounds):
        rng.shuffle(idx)
        cut = int(len(idx) * test_size)
        test, train = idx[:cut], idx[cut:]
        model = NaiveBayesClassifier().fit([texts[i] for i in train], [labels[i] for i in train])
        scores.append(sum(model.predict(texts[i]) == labels[i] for i in test) / len(test))
    return sum(scores) / len(scores)


def main():
    texts, labels = load(ROOT / "data" / "reviews.csv")
    acc = evaluate(texts, labels)
    print(f"Held-out accuracy: {acc:.0%} (avg of 20 random 80/20 splits, {len(texts)} examples)")

    model = NaiveBayesClassifier().fit(texts, labels)
    model.save(ROOT / "model.json")
    print(f"Saved model with {len(model.vocab)} words to model.json")


if __name__ == "__main__":
    main()
