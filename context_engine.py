"""
services/context_engine.py
--------------------------
Module 2: suggest dishes that suit the CURRENT WEATHER and TIME OF DAY, automatically.

Pieces
  1. get_live_weather(city)   - current temperature + sky for a city from Open-Meteo
                                (free, no API key). Needs internet; if it fails the app
                                falls back to manual weather + the computer's clock.
  2. build_context(...)       - turns temperature / sky / hour into labels:
                                  weather : hot | cold | mild   (+ rainy)
                                  meal    : breakfast | main | snack
  3. dish_labels(recipe)      - the same labels for a dish. Uses your labelled dataset
                                (data/context_labels.csv) and, for dishes that are not in it
                                (for example dishes the user added), a rule-based fallback.
  4. context_score(...)       - 0..1 how well a dish suits the context.
  5. suggest_for_context(...) - top dishes for "right now" without any ingredients.
"""

import csv
import os
import random
import re
from datetime import datetime

import streamlit as st

_BASE   = os.path.dirname(os.path.abspath(__file__))
_LABELS = os.path.join(_BASE, "context_labels.csv")

HOT_FROM  = 30     # °C  and above  -> hot
COLD_UPTO = 17     # °C  and below  -> cold

_MEAL_TEXT = {"breakfast": "breakfast time", "lunch": "lunch time", "snack": "evening snack time",
              "dinner": "dinner time", "late": "late-night snack time"}
_WEATHER_TEXT = {"hot": "hot weather", "cold": "cold weather", "mild": "pleasant weather"}


# ══════════════════════════════════════════════════════════════════════════════
# 1. Live weather (Open-Meteo)
# ══════════════════════════════════════════════════════════════════════════════
def _sky_from_code(code: int) -> str:
    if code in (51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 71, 73, 75, 77, 80, 81, 82, 85, 86, 95, 96, 99):
        return "rain"                         # drizzle / rain / snow / thunderstorm -> "rainy"
    if code in (0, 1):
        return "clear"
    return "cloudy"


@st.cache_data(ttl=600, show_spinner=False)
def get_live_weather(city: str) -> dict | None:
    """{'temp': 31.2, 'sky': 'clear', 'hour': 14, 'minute': 5, 'city': 'Karachi'} or None if offline."""
    city = (city or "").strip()
    if not city:
        return None
    try:
        import requests
        g = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                         params={"name": city, "count": 1, "language": "en"}, timeout=6).json()
        if not g.get("results"):
            return None
        place = g["results"][0]
        w = requests.get("https://api.open-meteo.com/v1/forecast", timeout=6, params={
            "latitude": place["latitude"], "longitude": place["longitude"],
            "current": "temperature_2m,weather_code", "timezone": "auto"}).json()["current"]
        t = datetime.fromisoformat(w["time"])           # local time AT that city
        return {"temp": float(w["temperature_2m"]), "sky": _sky_from_code(int(w["weather_code"])),
                "hour": t.hour, "minute": t.minute, "city": place.get("name", city)}
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 2. Context -> labels
# ══════════════════════════════════════════════════════════════════════════════
def meal_for_hour(hour: int) -> tuple[str, str]:
    """(dataset label, display key)"""
    if 5 <= hour < 11:
        return "breakfast", "breakfast"
    if 11 <= hour < 15:
        return "main", "lunch"
    if 15 <= hour < 19:
        return "snack", "snack"
    if 19 <= hour < 23:
        return "main", "dinner"
    return "snack", "late"


def build_context(temp: float, sky: str, hour: int, minute: int = 0, city: str = "", source: str = "live") -> dict:
    temp_class = "hot" if temp >= HOT_FROM else "cold" if temp <= COLD_UPTO else "mild"
    meal, meal_key = meal_for_hour(hour)
    rainy = sky == "rain"
    h12 = hour % 12 or 12
    clock = f"{h12}:{minute:02d} {'am' if hour < 12 else 'pm'}"
    sky_txt = {"clear": "clear sky", "cloudy": "cloudy", "rain": "rain"}.get(sky, sky)
    return {
        "temp": temp, "sky": sky, "hour": hour, "city": city, "source": source,
        "temp_class": temp_class, "rainy": rainy, "meal": meal, "meal_key": meal_key,
        "weather_text": _WEATHER_TEXT[temp_class] + (" with rain" if rainy else ""),
        "meal_text": _MEAL_TEXT[meal_key],
        "summary": f"{temp:.0f}°C, {sky_txt}" + (f" in {city}" if city else "") + f"  ·  {clock}",
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. Dish labels
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def _load_labels() -> dict:
    out = {}
    if os.path.exists(_LABELS):
        with open(_LABELS, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                lab = {k: int(r[k]) for k in ("hot", "cold", "rainy", "mild", "breakfast", "main", "snack")}
                name = r["Dish Name"].strip().lower()
                out[(name, (r.get("Cuisine") or "").strip().lower())] = lab   # same name can exist in 2 cuisines
                out.setdefault((name, ""), lab)                                 # name-only fallback
    return out


def _has(text: str, words) -> bool:
    return any(re.search(rf"\b{w}", text) for w in words)


_COLD_W  = ("soup", "stew", "curry", "hot chocolate", "chai", "tea", "haleem", "nihari", "paya", "roast", "casserole",
            "braise", "chili", "chilli", "ramen", "hot pot", "hotpot", "halwa", "coffee", "porridge", "gravy", "dal",
            "daal", "karahi", "korma", "bake", "baked", "lasagna", "shorba", "chowder", "pot pie", "mulled")
_HOT_W   = ("salad", "lassi", "raita", "smoothie", "juice", "cold", "iced", "ice cream", "kulfi", "chaat", "sorbet",
            "gazpacho", "cooler", "shake", "falooda", "lemonade", "sharbat", "cucumber", "watermelon", "fruit",
            "mojito", "popsicle", "granita", "slushie", "sandwich", "wrap")
_RAINY_W = ("pakora", "pakoda", "samosa", "fried", "fry", "fritter", "tempura", "chai", "tea", "hot", "soup", "roll",
            "paratha", "kebab", "kabab", "bhajia", "puri", "jalebi", "fries", "cutlet", "noodle", "ramen", "corn")
_BREAKFAST_W = ("breakfast", "egg", "omelet", "omelette", "pancake", "waffle", "toast", "paratha", "porridge", "oatmeal",
                "granola", "muffin", "cereal", "bagel", "scone", "crepe", "brunch", "halwa puri", "hash brown", "quiche")
_SNACK_W = ("cookie", "cake", "brownie", "dessert", "pakora", "samosa", "chaat", "dip", "bar", "bite", "candy", "fudge",
            "pudding", "smoothie", "lassi", "shake", "drink", "beverage", "appetizer", "snack", "fritter", "roll",
            "chips", "popcorn", "tea", "coffee", "juice", "ice cream", "kulfi", "tart", "pie", "dessert", "sweet")


def _heuristic_labels(rec: dict) -> dict:
    text = f"{rec.get('name', '')} {rec.get('category', '')}".lower()
    cold, hot = _has(text, _COLD_W), _has(text, _HOT_W)
    lab = {"hot": int(hot and not cold), "cold": int(cold and not hot), "rainy": int(_has(text, _RAINY_W)),
           "breakfast": int(_has(text, _BREAKFAST_W)), "snack": int(_has(text, _SNACK_W))}
    lab["mild"] = int(not lab["hot"] and not lab["cold"] or (hot and cold))
    lab["main"] = int(not lab["breakfast"] and not lab["snack"])
    return lab


def dish_labels(rec: dict) -> dict:
    """
    1) your fine-tuned Module 2 model (services/context_model.py) decides hot / cold / mild / breakfast / main / snack
    2) 'rainy' and any dish the model did not see come from the old label file / simple word rules
    """
    labels = _load_labels()
    name = rec["name"].strip().lower()
    base = (labels.get((name, rec.get("cuisine", "").strip().lower()))
            or (None if rec.get("cuisine") else labels.get((name, "")))
            or _heuristic_labels(rec))
    try:
        from context_model import predicted_labels
        m = predicted_labels(rec)
    except Exception:
        m = None
    if m:
        out = dict(base)
        out.update(m)
        return out
    return base


# ══════════════════════════════════════════════════════════════════════════════
# 4. Score + explanation
# ══════════════════════════════════════════════════════════════════════════════
def dish_probs(rec: dict):
    """The model's confidence (0..1) for hot / cold / mild / breakfast / main / snack, or None."""
    try:
        from context_model import predicted_probs
        return predicted_probs(rec)
    except Exception:
        return None


def _parts(rec: dict, ctx: dict) -> tuple[float, float]:
    """
    w = how well the dish fits the WEATHER (0..1),  m = how well it fits the MEAL TIME (0.15..1).
    With the model's probabilities the fit is graded, so a dish the model is SURE about ranks above one it is unsure about
    (the old 0/1 labels made far too many dishes tie).
    """
    lab = dish_labels(rec)
    pr = dish_probs(rec)
    if pr:
        tc = ctx["temp_class"]
        total = pr["hot"] + pr["cold"] + pr["mild"]
        share = pr[tc] / total if total > 0 else 0.0          # how much this dish prefers THIS weather over the others
        w = 0.5 * pr[tc] + 0.5 * share
        if ctx["rainy"] and lab["rainy"]:
            w = max(w, 0.8)                                    # rainy-day dishes (pakora, soup ...) get a boost
        m = 0.15 + 0.85 * pr[ctx["meal"]]
        return w, m
    # no probabilities available -> the simple 0/1 version
    if lab[ctx["temp_class"]] or (ctx["rainy"] and lab["rainy"]):
        w = 1.0
    elif lab["mild"]:
        w = 0.35
    else:
        w = 0.0
    m = 1.0 if lab[ctx["meal"]] else 0.15
    return w, m


def context_score(rec: dict, ctx: dict) -> float:
    w, m = _parts(rec, ctx)
    return 0.55 * w + 0.45 * m


def reason(rec: dict, ctx: dict) -> str:
    """Short text for the recipe card, e.g. 'Good for hot weather and lunch time'."""
    w, m = _parts(rec, ctx)
    graded = dish_probs(rec) is not None
    w_ok = w >= (0.55 if graded else 1.0)
    m_ok = m >= (0.70 if graded else 1.0)
    bits = []
    if w_ok:
        lab = dish_labels(rec)
        bits.append("rainy weather" if ctx["rainy"] and lab["rainy"] and not lab[ctx["temp_class"]] else _WEATHER_TEXT[ctx["temp_class"]])
    if m_ok:
        bits.append(ctx["meal_text"])
    return ("Good for " + " and ".join(bits)) if bits else ""


# ══════════════════════════════════════════════════════════════════════════════
# 5. "What should I cook now?"  (no ingredients needed)
# ══════════════════════════════════════════════════════════════════════════════
def suggest_for_context(cuisine: str, ctx: dict, top_n: int = 6) -> list[tuple]:
    from recipe_engine import load_recipes
    scored = []
    for rec in load_recipes():
        if rec["cuisine"].lower() != cuisine.lower():
            continue
        if len(rec["instructions"]) < 150 or len(rec["ingredients"]) < 3:
            continue                                         # skip very thin recipes
        scored.append((context_score(rec, ctx) + (0.1 if rec.get("user") else 0.0), rec))
    if not scored:
        return []
    # MEAL TIME FIRST: in the morning people eat breakfast whatever the weather.
    # Keep only dishes labelled for this meal (breakfast / main / snack); the weather then decides
    # which of them come first (e.g. lassi in summer, paratha in winter).
    fit = [x for x in scored if dish_labels(x[1]).get(ctx["meal"]) == 1]
    if len(fit) >= top_n * 2:
        scored = fit
    scored.sort(key=lambda x: -x[0])
    best = scored[0][0]
    pool = [x for x in scored if x[0] >= best - 0.08][:80]     # the dishes that fit best (almost tied with the top one)
    if len(pool) < top_n:
        pool = scored[:max(top_n * 4, 20)]
    # a different (but stable during that hour) selection each hour, so the list does not look the same all day
    rng = random.Random(f"{datetime.now():%Y-%m-%d-%H}-{cuisine}-{ctx['temp_class']}-{ctx['meal']}")
    pick = rng.sample(pool, min(top_n, len(pool)))
    pick.sort(key=lambda x: -x[0])
    return [(min(s, 1.0), r) for s, r in pick]
