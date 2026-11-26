"""
services/ui_helpers.py
----------------------
Display-only helpers for the redesigned interface.
Nothing here changes the dataset, the search, the models or any AI logic:
it only formats what the existing functions return.
"""
import base64
import glob
import html
import os
import re

_BASE = os.path.dirname(os.path.abspath(__file__))
_IMG_DIR = os.path.join(_BASE, "dish_images")

CUISINE_COLORS = {          # soft background, strong colour
    "Pakistani": ("#E7F2EA", "#176B4D"),
    "Indian":    ("#FCEFD9", "#B5651D"),
    "Chinese":   ("#FBE8EA", "#A8283E"),
    "Italian":   ("#E6EEF8", "#2B5B9A"),
    "Mexican":   ("#EEF3DC", "#5D7A12"),
    "Thai":      ("#F1E9F7", "#6E3A96"),
}

_ICONS = [
    (("beverage", "drink", "lassi", "shake", "sharbat", "juice", "tea", "chai", "coffee", "smoothie", "lemonade"), "🥤"),
    (("dessert", "sweet", "halwa", "kheer", "cake", "kulfi", "ice cream", "pudding", "gelato", "cookie"), "🍮"),
    (("soup", "stew", "shorba", "yakhni", "broth"), "🍲"),
    (("salad", "raita"), "🥗"),
    (("bread", "paratha", "naan", "roti", "puri", "toast", "sandwich"), "🫓"),
    (("rice", "biryani", "pulao", "fried rice", "khichdi", "congee", "risotto"), "🍚"),
    (("pasta", "noodle", "spaghetti", "lasagna", "chow mein"), "🍝"),
    (("seafood", "fish", "prawn", "shrimp", "jhinga"), "🐟"),
    (("egg", "omelette", "anda", "frittata"), "🍳"),
    (("snack", "appetizer", "pakora", "samosa", "roll", "kebab", "chaat", "fritter", "taco"), "🥟"),
]


def dish_icon(rec: dict) -> str:
    text = f"{rec.get('category', '')} {rec.get('name', '')}".lower()
    for words, icon in _ICONS:
        if any(w in text for w in words):
            return icon
    return "🍛"


def img_tag(data: bytes, alt: str, cls: str = "dg-img") -> str:
    b64 = base64.b64encode(data).decode("ascii")
    mime = "image/png" if data[:4] == b"\x89PNG" else "image/jpeg"
    return f'<img class="{cls}" src="data:{mime};base64,{b64}" alt="{html.escape(alt)}">'


def picture_html(rec: dict, state: str, data=None, note: str = "") -> str:
    """One picture box, always the same size: a real picture, a loading box, or a clean fallback."""
    if state == "done" and data:
        return f'<div class="dg-pic">{img_tag(data, rec["name"])}</div>'
    soft, strong = CUISINE_COLORS.get(rec.get("cuisine", ""), ("#E7F2EA", "#176B4D"))
    if state == "loading":
        return ('<div class="dg-pic dg-pic-loading"><div class="dg-pic-inner">'
                '<div class="dg-dots"><span></span><span></span><span></span></div>'
                '<div class="dg-pic-note">Preparing the picture</div></div></div>')
    small = f'<div class="dg-pic-note">{html.escape(note)}</div>' if note else ""
    return (f'<div class="dg-pic dg-pic-fallback" style="background:{soft};color:{strong}">'
            f'<div class="dg-pic-inner"><div class="dg-pic-icon">{dish_icon(rec)}</div>'
            f'<div class="dg-pic-cuisine">{html.escape(rec.get("cuisine", ""))}</div>{small}</div></div>')


_MIN = re.compile(r"(\d+)\s*(?:-|to|–)?\s*(\d+)?\s*(minutes|minute|mins|min)\b", re.I)
_HR = re.compile(r"(\d+(?:\.\d+)?)\s*(?:-|to|–)?\s*(\d+(?:\.\d+)?)?\s*(hours|hour|hrs|hr)\b", re.I)


def cook_minutes(instructions: str):
    """Total time written in the steps (e.g. 'fry for 6-7 minutes'), or None when the steps give no times."""
    text = instructions or ""
    total, found = 0.0, False
    for a, b, _ in _MIN.findall(text):
        total += (float(a) + float(b)) / 2 if b else float(a)
        found = True
    for a, b, _ in _HR.findall(text):
        total += 60 * ((float(a) + float(b)) / 2 if b else float(a))
        found = True
    if not found or total <= 0:
        return None
    return int(round(total / 5.0) * 5) or 5


def time_label(minutes) -> str:
    if not minutes:
        return ""
    if minutes < 60:
        return f"{minutes} min"
    h, m = divmod(minutes, 60)
    return f"{h} h {m} min" if m else f"{h} h"


def split_steps(text: str):
    """'note\\n1. step\\n2. step' (the translator's output) -> (note, [steps])."""
    note, steps = "", []
    for line in (text or "").split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(\d+)[.)]\s*(.*)$", line)
        if m:
            steps.append(m.group(2))
        elif not steps:
            note = (note + " " + line).strip()
        else:
            steps[-1] = steps[-1] + " " + line
    if not steps and note:
        steps, note = [note], ""
    return note, steps


def hero_pictures(n: int = 4) -> list[bytes]:
    """The most recent dish pictures already made on this computer (for the hero collage)."""
    files = sorted(glob.glob(os.path.join(_IMG_DIR, "*.jpg")), key=os.path.getmtime, reverse=True)[:n]
    out = []
    for f in files:
        try:
            with open(f, "rb") as fh:
                out.append(fh.read())
        except OSError:
            pass
    return out
