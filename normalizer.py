"""
core/normalizer.py
------------------
Ingredient normalization pipeline.
Combines HashMap + Fuzzy Matcher in order.

Pipeline per ingredient:
  Step 1 → HashMap lookup   (O(1), handles Roman Urdu + exact)
  Step 2 → Fuzzy match      (handles spelling mistakes, fallback)
  Step 3 → Passthrough      (return as-is if nothing matched)

Usage:
    from normalizer import normalize_one, normalize_all
"""

import re
from hashmap import hashmap_lookup
from fuzzy_matcher import fuzzy_match


# ── Single ingredient ──────────────────────────────────────────────────────────
def normalize_one(ingredient: str) -> dict:
    """
    Normalize a single ingredient string.

    Returns dict:
        {
            "original": "tamatar",
            "matched":  "tomato",
            "method":   "hashmap"   # hashmap | fuzzy | passthrough
        }
    """
    word = ingredient.strip().lower()

    # Step 1 — HashMap
    result = hashmap_lookup(word)
    if result:
        return {"original": word, "matched": result, "method": "hashmap"}

    # Step 2 — Fuzzy
    result = fuzzy_match(word)
    if result:
        return {"original": word, "matched": result, "method": "fuzzy"}

    # Step 3 — Passthrough
    return {"original": word, "matched": word, "method": "passthrough"}


# ── Text typed without commas ──────────────────────────────────────────────────
# small words that are not ingredients ("thori si mirchi", "aloo aur piyaz", "and")
_FILLERS = {"aur", "or", "and", "bhi", "sirf", "thori", "thora", "thoray", "thoda", "thodi", "si", "sa",
            "se", "ki", "ka", "ke", "kuch", "mere", "pas", "paas", "hai", "hain", "ha", "with", "some"}
# describing words: never an ingredient on their own ("hari piyaz" must stay together, it is spring onion)
_MODIFIERS = {"hari", "hara", "haray", "lal", "laal", "kali", "kala", "kaali", "safed", "sufaid", "sabz",
              "kacha", "kachi", "kachay", "pakka", "pakki", "taza", "tazi", "sukhi", "sookhi", "sukha",
              "bari", "bara", "choti", "chota", "desi", "moti", "mota", "pili", "peela", "meetha", "meethi",
              "green", "red", "black", "white", "yellow", "fresh", "dry", "dried", "big", "small"}


def _split_spaced(part: str) -> list[str]:
    """
    "aloo piyaz mirchi" -> ["aloo", "piyaz", "mirchi"]  (longest dictionary phrases first, so
    "hari mirch" or "chicken breast" stay together). A part that is already ONE known ingredient,
    or that does not split into at least 2 known ingredients, is returned unchanged.
    """
    from hashmap import _HASHMAP
    words = part.split()
    if len(words) < 2 or part in _HASHMAP:
        return [part]
    found, i, unknown = [], 0, 0
    while i < len(words):
        if words[i] in _FILLERS:                     # "aur", "bhi", "thori si" ... are not ingredients
            i += 1
            continue
        for n in (4, 3, 2, 1):
            phrase = " ".join(words[i:i + n])
            if len(words[i:i + n]) == n and phrase in _HASHMAP:
                if n == 1 and phrase in _MODIFIERS:
                    return [part]                     # e.g. "hari piyaz": keep the phrase whole
                found.append(phrase)
                i += n
                break
        else:
            w = words[i]
            if w in _MODIFIERS:
                return [part]
            if w not in _FILLERS:
                if len(w) >= 3 and fuzzy_match(w):     # a spelling mistake of a known ingredient
                    found.append(w)
                else:
                    unknown += 1
            i += 1
    if len(found) >= 2 and unknown == 0:
        return found
    return [part]


# ── Multiple ingredients ───────────────────────────────────────────────────────
def normalize_all(raw_input: str) -> tuple[list[str], list[dict]]:
    """
    Normalize a comma/newline separated ingredient string.

    Returns:
        clean   — deduplicated list of matched English ingredients
        details — per-ingredient breakdown dicts (shown as chips in the app)

    Example:
        normalize_all("tamatar, suger, chawal")
        → (["tomato", "sugar", "rice"], [...details...])
    """
    # Split on commas, Urdu comma (،), newlines
    raw_input = re.sub(r"\s+", " ", raw_input.lower().strip())
    parts = [p.strip() for p in re.split(r"[,،\n]+", raw_input) if p.strip()]

    # People often type without commas: "aloo piyaz mirchi". Split such a part into ingredients.
    expanded = []
    for p in parts:
        expanded.extend(_split_spaced(p))
    parts = expanded

    details = [normalize_one(p) for p in parts]
    clean   = list(dict.fromkeys(d["matched"] for d in details))  # deduplicate

    return clean, details
