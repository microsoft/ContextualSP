from typing import List, Set

from text.stemmer import PorterStemmer


# Extract term sets from a phrase containing " AND " and " OR " tokens. A phrase like "foo OR bar AND fnord OR gnarf"
# is turned into a list of term sets like [{"foo", "bar"}, {"fnord", "gnarf"}] to match to another phrase's term sets.
def extract_termsets(phrase: str) -> List[Set[str]]:
    outer = [p.strip() for p in phrase.split(" AND ")]
    inner = [set(item.split(" OR ")) for item in outer]
    return inner


# Extract term sets from a phrase containing " AND " and " OR " tokens. A phrase like "foo OR bar AND fnord OR gnarf"
# is turned into a list of term sets like [{"foo", "bar"}, {"fnord", "gnarf"}] to match to another phrase's term sets.
#
# This function normalizes each word.
def extract_termsets_with_normalization(phrase: str) -> List[Set[str]]:
    outer = [p.strip() for p in phrase.split(" AND ")]
    inner = [set(_normalize_words(item.split(" OR "))) for item in outer]
    return inner


def terms_overlap(phrase1_terms: List[Set[str]], phrase2_terms: List[Set[str]]):
    num = 0
    for t1 in phrase1_terms:
        for t2 in phrase2_terms:
            if t1.intersection(t2):
                num += 1
    return num


def _normalize_words(words: List[str]) -> List[str]:
    stemmed = []  # type: List[str]

    for w in words:
        if not w or len(w.strip()) == 0:
            return [""]
        w_lower = w.lower()
        # Remove leading articles from the phrase (e.g., the rays => rays).
        articles = ["a", "an", "the", "your", "his", "their", "my", "another", "other", "this", "that"]

        starting_article = next((article for article in articles if w_lower.startswith(_leading_word(article))), None)
        if starting_article is not None:
            w_lower = w_lower.replace(_leading_word(starting_article), "", 1)

        # Porter stemmer: rays => ray
        stemmed.append(PorterStemmer().stem(w_lower).strip())

    return stemmed


def _leading_word(word):
    return word + " "
