"""
core/hashmap.py
---------------
HashMap (dictionary) based lookup.
Loads ingredient_dictionary.csv into a Python dict — O(1) average lookup.

Handles:
  - Roman Urdu  →  English   (tamatar → tomato)
  - Common alternate spellings that are exact matches
"""

import csv
import os

_BASE = os.path.dirname(os.path.abspath(__file__))
_DICT_PATH = os.path.join(_BASE, "ingredient_dictionary.csv")

# ── Build the hashmap once at import time ──────────────────────────────────────
_HASHMAP: dict[str, str] = {}

def _load() -> dict:
    mapping = {}
    with open(_DICT_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            variation  = row.get("Variation", "").strip().lower()
            ingredient = row.get("Ingredient", "").strip().lower()
            if variation and ingredient:
                mapping[variation] = ingredient
    return mapping

_HASHMAP = _load()


# ── Public API ─────────────────────────────────────────────────────────────────
def hashmap_lookup(word: str) -> str | None:
    """
    Exact O(1) dictionary lookup.
    Returns matched English ingredient, or None if not found.

    Examples:
        hashmap_lookup("tamatar")  →  "tomato"
        hashmap_lookup("chawal")   →  "rice"
        hashmap_lookup("xyz123")   →  None
    """
    word = word.strip().lower()

    # 1. Exact key match
    if word in _HASHMAP:
        return _HASHMAP[word]

    # 2. Whole-word phrase match (e.g. "hari mirch powder" contains the phrase "hari mirch").
    #    Whole words only, so "price" can never match "rice". The longest phrase wins.
    padded = f" {word} "
    best_key = None
    for key in _HASHMAP:
        if len(key) >= 3 and f" {key} " in padded:
            if best_key is None or len(key) > len(best_key):
                best_key = key
    if best_key:
        return _HASHMAP[best_key]

    return None


def dictionary_size() -> int:
    """Return total number of entries in the hashmap."""
    return len(_HASHMAP)
