"""Classify text from the command line, or chat interactively.

    python predict.py "the service was amazing"
    python predict.py            # interactive mode
"""
import sys
from pathlib import Path

from sentiment import NaiveBayesClassifier

MODEL = Path(__file__).parent / "model.json"
EMOJI = {"positive": "😊", "negative": "😞"}


def show(model, text):
    probs = model.predict_proba(text)
    label = max(probs, key=probs.get)
    print(f"{EMOJI.get(label, '')} {label.upper()}  ({probs[label]:.0%} confident)")
    words = [w for w, s in model.explain(text, label) if s > 0]
    if words:
        print("   key words:", ", ".join(w.replace("NOT_", "not ") for w in words))


def main():
    if not MODEL.exists():
        sys.exit("No model.json found. Run `python train.py` first.")
    model = NaiveBayesClassifier.load(MODEL)
    if len(sys.argv) > 1:
        show(model, " ".join(sys.argv[1:]))
        return
    print("Type a sentence (empty line to quit).")
    while (text := input("> ").strip()):
        show(model, text)


if __name__ == "__main__":
    main()
