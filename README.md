# Sentiment AI 🧠

A sentiment analyzer built **from scratch in pure Python**: no scikit-learn, no NumPy, no API keys.
It uses a **Multinomial Naive Bayes** classifier to label text as positive or negative, and shows which words drove the decision.

```text
$ python predict.py "the support team was amazing and fast"
😊 POSITIVE  (93% confident)
   key words: amazing, fast

$ python predict.py "not good, very slow"
😞 NEGATIVE  (96% confident)
   key words: slow, not, not good, very
```

## How it works

1. **Tokenize**: lowercase the text, drop common stopwords, and mark negations, so that `not good` becomes `NOT_good` and isn't counted as `good`.
2. **Train**: count how often each word appears in positive and negative examples.
3. **Predict**: use Bayes' theorem with Laplace smoothing to compute
   `P(class | words) ∝ P(class) × Π P(word | class)`, working in log-space so the numbers don't underflow.
4. **Explain**: rank the words in the input by how strongly they favour the predicted class.

## Quick start

Requires Python 3.9+.

```bash
python train.py                         # train + evaluate + save model.json
python predict.py "I love this product" # classify one sentence
python predict.py                       # interactive mode
python -m unittest discover -s tests    # run the tests
```

## Results

| Metric | Value |
|---|---|
| Dataset | 100 labelled reviews (`data/reviews.csv`) |
| Held-out accuracy | **~72%** (average of 20 random 80/20 splits) |

That's reasonable for a dataset this small. Adding rows to `data/reviews.csv` is the easiest way to improve it.

## Project structure

```
sentiment/
  text.py      # tokenizer with negation handling
  model.py     # Naive Bayes: fit, predict, explain, save/load
data/
  reviews.csv  # training data
train.py       # training + evaluation
predict.py     # CLI
tests/         # unit tests
```

## Ideas to extend

- Train on a larger public dataset (e.g. IMDb reviews)
- Add a neutral class
- Use bigrams as features
- Wrap it in a small Flask/FastAPI endpoint

## License

MIT
