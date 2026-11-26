"""
services/recipe_engine.py
--------------------------
Recipe retrieval engine  (v5 - ingredient-first).

WHY THIS WAS REWRITTEN
  Before: only the sentence-embedding similarity decided the result, so typing "rice"
  could return dishes that contain NO rice at all (embeddings only measure "feels similar").

  Now the ranking is HYBRID and the user's ingredients are the boss:
    1. Every recipe is checked word-by-word: does it really contain the ingredient?
       ("rice" matches "basmati rice" but NOT "rice vinegar" / "rice flour" / "licorice")
    2. A recipe that contains NONE of the user's ingredients is never shown.
    3. Recipes that contain ALL your ingredients come first.
    4. Then: how much of the recipe you already have (coverage), then the embedding
       similarity only as a small tie-breaker.

Public functions (same names as before, so the rest of the app keeps working):
    load_recipes(), find_best_recipes(), get_missing_ingredients(),
    ingredient_coverage_score()  + new: matched_ingredients(), match_summary()
"""

import csv
import os
import re

import streamlit as st

_BASE         = os.path.dirname(os.path.abspath(__file__))
_RECIPES_PATH = os.path.join(_BASE, "recipes.csv")


# ── Load recipes ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_builtin() -> list[dict]:
    recipes = []
    with open(_RECIPES_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ings = [i.strip().lower() for i in row.get("Ingredient Names", "").split(",") if i.strip()]
            recipes.append({
                "name":         row.get("Dish Name", "").strip(),
                "ingredients":  ings,
                "cuisine":      row.get("Cuisine", "").strip(),
                "category":     row.get("Category", "").strip(),
                "instructions": row.get("Instructions", "").strip(),
            })
    return recipes


def load_recipes() -> list[dict]:
    """Built-in dataset first, then the dishes the user added (re-read every time, so new dishes show instantly)."""
    from user_recipes import load_user_recipes
    return _load_builtin() + load_user_recipes()


# ══════════════════════════════════════════════════════════════════════════════
# Word-level ingredient matching
# ══════════════════════════════════════════════════════════════════════════════
_WORD = re.compile(r"[a-z]+")

# things everybody has at home - they never count as "missing" and never inflate the score
STAPLES = {
    "salt", "water", "pepper", "black pepper", "oil", "cooking oil", "vegetable oil", "sugar",
    "salt and pepper", "ice", "cooking spray", "kosher salt", "sea salt", "ground black pepper",
    "freshly ground black pepper", "warm water", "cold water", "hot water", "boiling water",
}

# if the user's word is only a *modifier* in front of one of these, it is a different product
# (rice -> rice vinegar / rice flour / rice noodles;  milk -> milk powder ...)
_DIFFERENT_THING = {
    "vinegar", "flour", "wine", "noodle", "noodle", "paper", "sauce", "stock", "broth", "powder",
    "oil", "paste", "seasoning", "cake", "bran", "syrup", "vermicelli", "pudding", "flake", "cereal",
    "krispie", "bread", "cracker", "cookie", "chip", "juice", "extract", "essence", "liqueur",
    "bun", "wrapper", "starch", "milk", "cream", "yogurt", "butter", "cheese", "salt", "sugar",
    "dressing", "mix", "soup", "bouillon", "cube", "spread", "jam", "jelly", "sprinkle", "colour",
    "color", "oi",
}
# ...but these heads mean "the same thing, just a cut / form"  (chicken breast, onion slice ...)
_SAME_THING = {
    "breast", "thigh", "leg", "wing", "drumstick", "fillet", "filet", "piece", "meat", "cube",
    "mince", "minced", "ground", "liver", "steak", "chop", "slice", "clove", "head", "bulb", "leaf",
    "sprig", "bunch", "stalk", "stem", "root", "seed", "bean", "pea", "grain", "floret", "tail",
    "shoulder", "rib", "neck", "loin", "tenderloin", "strip", "chunk", "wedge", "ring", "half",
}
_NON_FOOD_WORDS = {
    "fresh", "dried", "dry", "chopped", "sliced", "diced", "minced", "ground", "large", "small",
    "medium", "ripe", "raw", "cooked", "boiled", "whole", "finely", "roughly", "crushed", "grated",
    "frozen", "canned", "boneless", "skinless", "extra", "virgin", "organic", "good", "quality",
    "of", "and", "or", "a", "the", "to", "taste", "for", "plain", "unsalted", "salted", "lean",
}


_SPECIAL_SINGULAR = {"chilies": "chili", "chilis": "chili", "chillies": "chili", "chilli": "chili",
                     "chillis": "chili", "chile": "chili", "chiles": "chili", "leaves": "leaf",
                     "loaves": "loaf", "potatoes": "potato", "tomatoes": "tomato", "mangoes": "mango"}

# a word in front that turns it into a different product: "coconut milk" is not milk, "sweet potato" is not potato
_FALSE_PREFIX = {"coconut", "almond", "soy", "oat", "sweet", "peanut", "cocoa", "coffee", "cashew", "water"}


def _singular(w: str) -> str:
    if w in _SPECIAL_SINGULAR:
        return _SPECIAL_SINGULAR[w]
    if len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"                      # berries -> berry, chilies -> chily (both sides same)
    if w.endswith(("oes", "ches", "shes", "sses", "xes")):
        return w[:-2]                            # tomatoes -> tomato, peaches -> peach
    if w.endswith("ss") or w.endswith("us") or w.endswith("is"):
        return w
    if w.endswith("s"):
        return w[:-1]                            # eggs -> egg, onions -> onion
    return w


def _tokens(text: str) -> list[str]:
    words = [_singular(w) for w in _WORD.findall(text.lower())]
    return [w for w in words if w not in _NON_FOOD_WORDS]


def _norm_phrase(text: str) -> str:
    return " ".join(_tokens(text))


def _is_staple(ing: str) -> bool:
    n = _norm_phrase(ing)
    return ing.strip().lower() in STAPLES or n in {_norm_phrase(s) for s in STAPLES}


def _contains(hay: list[str], needle: list[str]) -> int:
    """Index where `needle` appears inside `hay` as consecutive words, else -1."""
    n = len(needle)
    for i in range(len(hay) - n + 1):
        if hay[i:i + n] == needle:
            return i
    return -1


# ── Same ingredient, different names ─────────────────────────────────────────
# Recipes come from many sources, so one ingredient is written in different ways
# ("prawns" / "shrimp", "beef mince" / "ground beef"). Every name in a group counts as the same thing.
SYNONYM_GROUPS = [
    {"prawns", "shrimp", "king prawns", "tiger prawns"},
    {"beef mince", "minced beef", "ground beef", "beef keema"},
    {"mutton mince", "lamb mince", "minced lamb", "ground lamb", "minced mutton"},
    {"chicken mince", "minced chicken", "ground chicken"},
    {"mutton", "lamb", "goat meat"},
    {"sweet corn", "corn", "corn kernels", "corn on the cob", "baby corn", "sweetcorn"},
    {"fresh coriander", "coriander", "coriander leaves", "cilantro", "cilantro leaves"},
    {"eggplant", "aubergine", "brinjal"},
    {"capsicum", "bell pepper", "green bell pepper", "red bell pepper"},
    {"spring onion", "green onion", "scallion"},
    {"white chickpeas", "chickpeas", "garbanzo beans", "kabuli chana"},
    {"gram flour", "besan", "besan flour", "chickpea flour"},
    {"wheat flour", "atta", "atta flour", "whole wheat flour", "chapati flour"},
    {"yogurt", "yoghurt", "curd", "curds", "dahi"},
    {"basmati rice", "rice", "long grain rice", "jasmine rice", "cooked rice", "white rice"},
    {"peanuts", "groundnut", "groundnuts"},
    {"lemon", "lime"},                       # "nimbu" is used for both in Pakistan
    {"zucchini", "courgette"},
    {"okra", "bhindi", "lady finger"},
    {"dal", "lentils"},
    {"paneer", "cottage cheese"},
]
# a general word also covers its kinds ("pasta" -> spaghetti, penne ...), but not the other way round
GENERAL_TERMS = {
    "pasta": ["spaghetti", "penne", "macaroni", "fusilli", "rigatoni", "linguine", "fettuccine", "ziti",
              "rotini", "orzo", "farfalle", "tagliatelle", "pappardelle", "angel hair", "lasagna noodles",
              "lasagne sheets", "noodles"],
    "cheese": ["mozzarella", "parmesan", "cheddar", "ricotta", "feta", "pecorino", "gruyere", "provolone",
               "monterey jack", "mascarpone"],
    "fish": ["tilapia", "cod", "salmon", "tuna", "snapper", "haddock", "pomfret", "mackerel", "sole", "basa",
             "white fish", "fish fillets"],
}
_FALSE_PREFIX.add("bean")                    # "bean curd" is tofu, not yogurt


def _key(text: str) -> str:
    return " ".join(_tokens(text))


_ALTERNATIVES: dict[str, list[str]] = {}
for _g in SYNONYM_GROUPS:
    for _name in _g:
        _ALTERNATIVES.setdefault(_key(_name), []).extend(x for x in _g if _key(x) != _key(_name))
for _general, _kinds in GENERAL_TERMS.items():
    _ALTERNATIVES.setdefault(_key(_general), []).extend(_kinds)


def match_strength(user_ing: str, recipe_ing: str) -> float:
    """
    Like _direct_strength, but other names of the same ingredient also count
    (the user's "prawns" matches a recipe's "shrimp").
    """
    best = _direct_strength(user_ing, recipe_ing)
    if best >= 1.0:
        return best
    for alt in _ALTERNATIVES.get(_key(user_ing), ()):
        best = max(best, _direct_strength(alt, recipe_ing))
        if best >= 1.0:
            break
    return best


def _direct_strength(user_ing: str, recipe_ing: str) -> float:
    """
    How well does ONE recipe ingredient satisfy ONE user ingredient?
        1.0  = the real thing        ("rice" ~ "basmati rice", "chicken" ~ "chicken breast")
        0.3  = related product       ("rice" ~ "rice vinegar")  -> counts only a little
        0.0  = no match
    """
    u = _tokens(user_ing)
    r = _tokens(recipe_ing)
    if not u or not r:
        return 0.0
    if u == r:
        return 1.0
    # Pakistani / Indian recipes write "moong dal", "masoor dal", "chana dal"; the user's "dal" is normalised to "lentils"
    if u == ["lentil"] and any(w in ("dal", "daal", "dhal", "lentil") for w in r):
        return 1.0
    pos = _contains(r, u)
    if pos < 0:
        # user typed several words ("green chili") and the recipe words are scattered/reordered
        if len(u) > 1 and set(u) <= set(r):
            return 0.9
        return 0.0
    end = pos + len(u)
    if pos > 0 and r[pos - 1] in _FALSE_PREFIX and r[pos - 1] not in u:
        return 0.3                         # coconut milk / sweet potato / peanut butter
    if end == len(r):                      # user's words are the END of the phrase: basmati rice
        return 1.0
    head = r[-1]                           # user's words are a modifier: "rice vinegar", "chicken breast"
    if head in _SAME_THING:
        return 1.0
    if head in _DIFFERENT_THING:
        return 0.3
    return 0.85                            # e.g. "garlic ginger paste" style leftovers: probably related


def _best_for_user_ing(user_ing: str, recipe_ings: list[str]) -> tuple[float, str]:
    best, which = 0.0, ""
    for ri in recipe_ings:
        s = match_strength(user_ing, ri)
        if s > best:
            best, which = s, ri
            if s >= 1.0:
                break
    return best, which


def matched_ingredients(user_ings: list[str], recipe_ings: list[str]) -> list[str]:
    """Recipe ingredients that the user already has (strong matches only)."""
    out = []
    for ri in recipe_ings:
        if any(match_strength(ui, ri) >= 0.85 for ui in user_ings):
            out.append(ri)
    return out


def get_missing_ingredients(user_ings: list[str], recipe_ings: list[str]) -> list[str]:
    """Ingredients the recipe needs but the user doesn't have (salt / water / oil etc. are ignored)."""
    missing = []
    for ri in recipe_ings:
        if _is_staple(ri):
            continue
        if not any(match_strength(ui, ri) >= 0.85 for ui in user_ings):
            if ri not in missing:
                missing.append(ri)
    return missing[:6]


def ingredient_coverage_score(user_ings: list[str], recipe_ings: list[str]) -> float:
    """What share of the recipe's real (non-staple) ingredients does the user have?  0.0 - 1.0"""
    real = [ri for ri in recipe_ings if not _is_staple(ri)]
    if not real:
        return 0.0
    have = sum(1 for ri in real if any(match_strength(ui, ri) >= 0.85 for ui in user_ings))
    return have / len(real)


def match_summary(user_ings: list[str], recipe: dict) -> dict:
    """Numbers the UI shows on a recipe card."""
    real = [ri for ri in recipe["ingredients"] if not _is_staple(ri)]
    have = matched_ingredients(user_ings, real)
    return {
        "have": len(have),
        "total": len(real),
        "have_list": have,
        "missing": get_missing_ingredients(user_ings, recipe["ingredients"]),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main retrieval
# ══════════════════════════════════════════════════════════════════════════════
def _embedding_scores(user_ings: list[str], idx: list[int]):
    """
    Optional semantic tie-breaker, only for the BUILT-IN recipes (their embeddings are cached on disk).
    Dishes added by the user simply get 0 here, so adding a dish never triggers a slow re-embedding.
    Returns {global_index: similarity} or None if the model is not available.
    """
    try:
        from embedder import embed_query, get_recipe_embeddings, cosine_similarity_scores
        builtin = _load_builtin()
        use = [i for i in idx if i < len(builtin)]
        if not use:
            return {}
        db = get_recipe_embeddings([", ".join(r["ingredients"]) for r in builtin])[use]
        sims = cosine_similarity_scores(embed_query(", ".join(user_ings)), db)
        return {g: float(sims[k]) for k, g in enumerate(use)}
    except Exception:
        return None


def find_best_recipes(user_ings: list[str], cuisine: str, top_n: int = 5, context: dict | None = None) -> list[tuple]:
    """
    Returns [(score 0-1, recipe_dict), ...]  best first.
    Only recipes that really contain at least one of the user's ingredients are returned.
    `context` (optional, from services.context_engine.build_context): when given, dishes that also suit the
    current weather and time of day are ranked higher. Ingredients always stay more important.
    """
    user_ings = [u for u in user_ings if u and u.strip()]
    if not user_ings:
        return []

    all_recipes = load_recipes()
    idx = [i for i, r in enumerate(all_recipes) if r["cuisine"].lower() == cuisine.lower()]
    pool = [all_recipes[i] for i in idx]
    if not pool:
        return []

    n_user = len(user_ings)
    real_user = [u for u in user_ings if not _is_staple(u)]   # what the user typed minus salt, pepper, oil ...
    scored = []          # (all_matched, global_index, match_score, coverage, name_bonus, recipe)

    for pos, (gidx, rec) in enumerate(zip(idx, pool)):
        ris = rec["ingredients"]
        strengths, name_bonus = [], 0.0
        name_tokens = set(_tokens(rec["name"]))
        for ui in user_ings:
            s, _ = _best_for_user_ing(ui, ris)
            strengths.append(s)
            if s >= 0.85 and set(_tokens(ui)) & name_tokens:
                name_bonus += 1.0 / n_user          # the dish is "named after" your ingredient
        strong = sum(1 for s in strengths if s >= 0.85)
        if strong == 0:
            continue                                # NOTHING the user typed is in this recipe -> never show
        if real_user and not any(s >= 0.85 for u, s in zip(user_ings, strengths) if u in real_user):
            continue                                # only salt / pepper / oil matched -> not a real match
        match_score = sum(strengths) / n_user       # 0..1 : how many of YOUR ingredients are used
        coverage    = ingredient_coverage_score(user_ings, ris)
        all_matched = 1 if strong == n_user else 0
        scored.append((all_matched, gidx, match_score, coverage, name_bonus, rec))

    if not scored:
        return []

    emb = _embedding_scores(user_ings, idx)

    ranked = []
    for all_matched, gidx, match_score, coverage, name_bonus, rec in scored:
        sem = max(0.0, emb.get(gidx, 0.0)) if emb else 0.0
        # ingredient use matters most, then "do I have most of what this dish needs", then small extras
        final = 0.55 * match_score + 0.30 * coverage + 0.10 * min(name_bonus, 1.0) + 0.05 * sem
        if rec.get("user"):
            final += 0.12                       # the user's own dishes get a small boost so they are easy to find
        if context:
            from context_engine import context_score
            final = 0.8 * final + 0.2 * context_score(rec, context)
        ranked.append((all_matched, final, rec))

    # dishes using ALL typed ingredients first, then by score
    ranked.sort(key=lambda x: (-x[0], -x[1]))

    # avoid showing 5 near-identical dishes with the same name
    out, seen = [], set()
    for _, final, rec in ranked:
        key = rec["name"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append((min(final, 1.0), rec))
        if len(out) >= top_n:
            break
    return out
