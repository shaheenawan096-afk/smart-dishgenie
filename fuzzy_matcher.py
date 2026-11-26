"""
core/fuzzy_matcher.py
---------------------
Fuzzy string matching using RapidFuzz.
Used as FALLBACK when hashmap lookup returns nothing.

Handles:
  - Spelling mistakes     (suger → sugar, tomatto → tomato)
  - Near-miss Roman Urdu  (murgh → chicken via fuzzy on known English list)

Algorithm: WRatio (Weighted Ratio) — handles partial matches + transpositions
Threshold: 78  (lower = more matches but more false positives)
"""

from rapidfuzz import process, fuzz
from hashmap import _HASHMAP

# Build sorted unique list of all known English ingredients (the values)
_ENGLISH_DB: list[str] = sorted(set(_HASHMAP.values()))
_VARIATIONS: list[str] = sorted(k for k in _HASHMAP if len(k) >= 4)


# ── Public API ─────────────────────────────────────────────────────────────────
def fuzzy_match(word: str, threshold: int = 78) -> str | None:
    """
    Find closest matching English ingredient using fuzzy string similarity.
    Returns matched ingredient if score >= threshold, else None.

    Examples:
        fuzzy_match("tomatto")  →  "tomato"   (score ~93)
        fuzzy_match("suger")    →  "sugar"    (score ~91)
        fuzzy_match("chiken")   →  "chicken"  (score ~89)
        fuzzy_match("xyz123")   →  None       (score too low)
    """
    word = word.strip().lower()
    result = process.extractOne(
        word,
        _ENGLISH_DB,
        scorer=fuzz.ratio,
        score_cutoff=threshold,
    )
    # Also look at every known VARIATION (Roman Urdu, other spellings) so a misspelled Roman Urdu word
    # like "tamaatarr" or "piyyaz" still finds its ingredient. Stricter cut-off here (85) to avoid junk.
    alt = process.extractOne(word, _VARIATIONS, scorer=fuzz.ratio, score_cutoff=max(threshold, 85))
    if alt and (not result or alt[1] > result[1]):
        return _HASHMAP[alt[0]]
    return result[0] if result else None


def fuzzy_score(word: str, candidate: str) -> int:
    """Return similarity score (0-100) between two strings."""
    return int(fuzz.WRatio(word.lower(), candidate.lower()))


def known_ingredients_count() -> int:
    """Return number of unique English ingredients in matching database."""
    return len(_ENGLISH_DB)
