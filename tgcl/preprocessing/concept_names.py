"""Singular form of concept names ("dogs" -> "dog", but "gas" stays "gas").

inflect.singular_noun strips a trailing "s" from words that are already singular
(gas -> ga, bus -> bu, grass -> gras, lens -> len), which breaks the match
between a class name and the model's answer. Here WordNet's noun lemmatizer
proposes the candidates and the most frequent reading (SemCor counts) wins: gas
>> ga (gallium), bowl >> bowls (the game). A word whose verb/adjective/adverb
reading dominates is kept ("was" is "be", not "wa"), as is any word that would
shrink below 3 letters. Words unknown to WordNet fall back to inflect.
"""

import functools
import os

import inflect
import nltk

_INFLECT = inflect.engine()


@functools.lru_cache(maxsize=1)
def _wordnet():
    """The WordNet reader; fetched once into $NLTK_DATA when missing."""
    try:
        nltk.data.find(
            "corpora/wordnet.zip"
        )  # nltk.download stores it zipped; the reader opens the zip
    except LookupError:
        nltk.download(
            "wordnet",
            download_dir=os.environ.get("NLTK_DATA") or None,
            quiet=True,
        )
    from nltk.corpus import wordnet

    return wordnet


def _count(wn, lemma: str, pos: str) -> int:
    return sum(
        l.count()
        for s in wn.synsets(lemma, pos)
        for l in s.lemmas()
        if l.name().lower() == lemma
    )


@functools.lru_cache(maxsize=None)
def singularize(word: str) -> str:
    """Singular of a lower-case word; non-nouns and singular nouns unchanged."""
    wn = _wordnet()
    # _morphy lists every base form; the public morphy() returns only one.
    candidates = wn._morphy(word, wn.NOUN)  # pylint: disable=protected-access
    if not candidates:
        return (
            word if wn.synsets(word) else (_INFLECT.singular_noun(word) or word)
        )
    best = max(candidates, key=lambda c: (_count(wn, c, wn.NOUN), c == word))
    if best == word:
        return word
    other = max(
        [
            _count(wn, base, pos)
            for pos in (wn.VERB, wn.ADJ, wn.ADV)
            for base in wn._morphy(word, pos)  # pylint: disable=protected-access
        ]
        + [0]
    )
    if len(best) < 3 or other > _count(wn, best, wn.NOUN):
        return word
    return best


def singularize_phrase(phrase: str) -> str:
    """Singularize every word of a whitespace-separated phrase."""
    return " ".join(singularize(w) for w in phrase.split())
