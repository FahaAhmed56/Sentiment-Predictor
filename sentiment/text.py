import re

NEGATIONS = {"not", "no", "never", "dont", "don't", "isnt", "isn't", "wasnt",
             "wasn't", "didnt", "didn't", "cant", "can't", "wont", "won't", "nothing"}
STOPWORDS = {"a", "an", "the", "and", "or", "is", "it", "this", "that", "i", "to",
             "of", "was", "for", "in", "on", "my", "with", "me", "be", "at", "so", "are"}


def tokenize(text: str) -> list[str]:
    """Lowercase, split into words, drop stopwords and mark negated words.

    "not good" becomes ["not", "NOT_good"] so the model can tell it apart from "good".
    """
    words = re.findall(r"[a-z']+", text.lower())
    tokens, negate = [], False
    for w in words:
        if w in NEGATIONS:
            negate = True
            tokens.append(w)
            continue
        if w in STOPWORDS:
            continue
        tokens.append(f"NOT_{w}" if negate else w)
        negate = False
    return tokens
