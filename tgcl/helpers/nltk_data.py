"""NLTK corpora, downloaded on first use into $NLTK_DATA (else NLTK's default).

Every module reads the English word list through english_words(), so a fresh
machine needs no manual `nltk.download` before the pipeline runs.
"""

import functools
import os
from typing import List

import nltk

# NLTK resource name -> path that nltk.data.find() resolves when it is present.
_RESOURCE_PATHS = {"words": "corpora/words", "wordnet": "corpora/wordnet.zip"}


@functools.lru_cache(maxsize=None)
def ensure(name: str) -> None:
    """Download an NLTK resource unless it can already be loaded.

    Args:
        name: Resource name, a key of _RESOURCE_PATHS (e.g. "words").
    """
    try:
        nltk.data.find(_RESOURCE_PATHS[name])
    except LookupError:
        nltk.download(
            name, download_dir=os.environ.get("NLTK_DATA") or None, quiet=True
        )


@functools.lru_cache(maxsize=1)
def english_words() -> List[str]:
    """The NLTK English word list (corpora/words), fetched once if missing."""
    ensure("words")
    from nltk.corpus import words  # pylint: disable=import-outside-toplevel

    return words.words()
