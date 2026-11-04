"""Word filter of the concept text grounding: English words, not stop words."""

from pathlib import Path
from typing import List

__all__ = ["GIST_FILE_PATH", "get_stopwords", "is_number", "valid_word"]

# Stop words of the grounding filter; packaged with tgcl (independent of the
# working directory).
GIST_FILE_PATH = str(
    Path(__file__).resolve().parent.parent / "assets" / "gist_stopwords.txt"
)


def is_number(string: str) -> bool:
    """Whether the string parses as a float."""
    try:
        float(string)
        return True
    except ValueError:
        return False


def get_stopwords(gist_file_path: str = GIST_FILE_PATH) -> List[str]:
    """The comma-separated stop words of a file (default: the packaged list)."""
    with open(gist_file_path, "r", encoding="utf-8") as f:
        return f.read().split(",")


def valid_word(word, eng_corpus: List[str], stopwords: List[str]) -> bool:
    """A number, or a non-empty English word (in eng_corpus) that is not a stop
    word; compared lower-cased and stripped."""
    word = word.lower().strip()
    return (
        word in eng_corpus and len(word) > 0 and word not in stopwords
    ) or is_number(word)
