"""
services/translator.py
-----------------------
OFFLINE translation using Meta NLLB-200 (facebook/nllb-200-distilled-600M).

  translate_recipe(text, lang)         -> recipe text in English / Roman Urdu / Urdu / Sindhi / Pashto
  ingredients_to_english(text, lang)   -> spoken/typed ingredients in Urdu/Sindhi/Pashto -> English

Roman Urdu = NLLB Urdu translation, then rule-based Urdu-script -> Roman letters.
No internet / no API is used after the model is downloaded once.
"""

import re
import streamlit as st
from config import NLLB_MODEL, NLLB_CODES, get_device, ONLINE, GOOGLE_CODES

_tok = None
_model = None


def _load():
    global _tok, _model
    if _model is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        _tok = AutoTokenizer.from_pretrained(NLLB_MODEL)
        _model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL).to(get_device()).eval()
    return _tok, _model


def translate_batch(texts: list[str], src: str, tgt: str, batch_size: int = 8, num_beams: int = 3) -> list[str]:
    """Translate a list of short texts. src/tgt are keys of NLLB_CODES (English/Urdu/Sindhi/Pashto)."""
    if ONLINE:
        return _google_batch(texts, src, tgt)
    import torch
    tok, model = _load()
    dev = get_device()
    tok.src_lang = NLLB_CODES[src]
    bos = tok.convert_tokens_to_ids(NLLB_CODES[tgt])

    results = [""] * len(texts)
    order = sorted((i for i, t in enumerate(texts) if t.strip()), key=lambda i: len(texts[i]))
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True,
                  truncation=True, max_length=256).to(dev)
        with torch.inference_mode():
            out = model.generate(**enc, forced_bos_token_id=bos,
                                 num_beams=num_beams, max_new_tokens=256)
        for i, t in zip(idx, tok.batch_decode(out, skip_special_tokens=True)):
            results[i] = t.strip()
    return results


def _google_batch(texts: list[str], src: str, tgt: str) -> list[str]:
    """Online version: Google Translate through deep-translator (no model on the server)."""
    from deep_translator import GoogleTranslator
    tr = GoogleTranslator(source=GOOGLE_CODES[src], target=GOOGLE_CODES[tgt])
    out = []
    for t in texts:
        out.append((tr.translate(t) or "").strip() if t.strip() else "")
    return out


# ── Recipe text handling ───────────────────────────────────────────────────────
def _pieces(text: str, limit: int = 250) -> list[str]:
    """Split a step into sentence groups <= limit chars (NLLB works best on short inputs)."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    out, cur = [], ""
    for s in sents:
        if cur and len(cur) + len(s) + 1 > limit:
            out.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        out.append(cur)
    # a single very long sentence: cut on commas / words
    final = []
    for p in out:
        while len(p) > limit * 1.6:
            cut = p.rfind(",", 0, limit)
            cut = cut if cut > 40 else p.rfind(" ", 0, limit)
            final.append(p[:cut + 1].strip())
            p = p[cut + 1:].strip()
        final.append(p)
    return final


def parse_recipe(text: str):
    """Return (note, steps, numbered). Handles 'Step 1: ... Step 2: ...' format and plain paragraphs."""
    parts = re.split(r"(?:^|\s)Step\s*\d+\s*:\s*", text.strip())
    if len(parts) > 1:
        note = parts[0].strip()
        steps = [p.strip() for p in parts[1:] if p.strip()]
        return note, steps, True
    return "", [text.strip()], False


def _render(note: str, steps: list[str], numbered: bool) -> str:
    lines = [note] if note else []
    for i, s in enumerate(steps, 1):
        lines.append(f"{i}. {s}" if numbered else s)
    return "\n".join(lines)


@st.cache_data(show_spinner=False, max_entries=128)
def translate_recipe(text: str, lang: str) -> str:
    """
    Recipe instructions -> lang (English | Roman Urdu | Urdu | Sindhi | Pashto).
    RAISES on failure (so Streamlit does not cache errors); the caller shows the message.
    """
    if not text.strip():
        return ""
    note, steps, numbered = parse_recipe(text)

    if lang == "English":
        return _render(note, steps, numbered)

    if lang == "Roman Urdu":
        urdu = translate_recipe(text, "Urdu")
        return "\n".join(_urdu_to_roman(line) for line in urdu.split("\n"))

    units = ([note] if note else []) + steps
    flat, owner = [], []
    for u_i, u in enumerate(units):
        for p in _pieces(u):
            flat.append(p)
            owner.append(u_i)
    outs = translate_batch(flat, "English", lang)
    merged = [""] * len(units)
    for o, t in zip(owner, outs):
        merged[o] = f"{merged[o]} {t}".strip()

    if note:
        return _render(merged[0], merged[1:], numbered)
    return _render("", merged, numbered)


# ── Spoken / typed ingredients in another language -> English ──────────────────
_SPLIT = re.compile(r"[،,؛;\n۔.]+|\s+(?:اور|او|۽)\s+")   # commas + 'and' in Urdu / Pashto / Sindhi


def _split_en(text: str) -> list[str]:
    return [x.strip().strip(".").lower() for x in re.split(r"[,;\n]+|\s+\band\b\s+", text) if x.strip().strip(".")]


def ingredients_to_english_checked(text: str, src_lang: str):
    """
    Spoken/typed ingredients -> (known, unknown) English lists.
      1) exact offline Urdu-script lexicon  (fast + accurate for common ingredients)
      2) anything left goes to NLLB INSIDE a sentence ("Cooking ingredients: ...") because
         translation models are very unreliable on isolated single words
      3) every English result must be recognised by the project's dictionary / fuzzy matcher,
         otherwise it is returned in 'unknown' instead of polluting the ingredient list
    """
    if src_lang == "English":
        return _split_en(text), []

    from ingredient_lexicon import lookup
    from normalizer import normalize_one
    from rapidfuzz import fuzz

    items = [p.strip() for p in _SPLIT.split(text) if p and p.strip()]
    candidates, leftovers = [], []
    for it in items:
        hit = lookup(it)
        if hit:
            candidates.extend(hit)
        else:
            leftovers.append(it)

    unknown = []
    if leftovers:
        carrier = translate_batch(["Cooking ingredients:"], "English", src_lang)[0]
        out = translate_batch([f"{carrier} " + "، ".join(leftovers)], src_lang, "English", num_beams=5)[0]
        if ":" in out[:60]:
            out = out.split(":", 1)[1]
        for e in _split_en(out):
            candidates.append(e)

    known, seen = [], set()
    for c in candidates:
        if c in known or c in unknown:
            continue
        r = normalize_one(c)
        # Fuzzy matches must be STRICT here (plain spelling slips like tomatto->tomato). The normal
        # fuzzy matcher is lenient and would accept junk such as "the world" -> "corn on the cob".
        ok = r["method"] == "hashmap" or (r["method"] == "fuzzy" and fuzz.ratio(c, r["matched"]) >= 80)
        if ok:
            shown = r["matched"] if r["method"] == "fuzzy" else c      # fix spelling slips, else keep the clean word
            key = r["matched"].lower()                                  # internal key just to drop duplicates
            if key not in seen:
                seen.add(key)
                known.append(shown)
        else:
            unknown.append(c)
    return known, unknown


def ingredients_to_english(text: str, src_lang: str) -> str:
    """Compatibility helper: known ingredients as a comma-separated English string."""
    known, _ = ingredients_to_english_checked(text, src_lang)
    return ", ".join(known)


# ── Urdu script → Roman Urdu character map ────────────────────────────────────
def _urdu_to_roman(urdu_text: str) -> str:
    URDU_ROMAN = {
        'ا': 'a', 'آ': 'aa', 'ب': 'b', 'پ': 'p', 'ت': 't', 'ٹ': 'T',
        'ث': 's', 'ج': 'j', 'چ': 'ch', 'ح': 'h', 'خ': 'kh', 'د': 'd',
        'ڈ': 'D', 'ذ': 'z', 'ر': 'r', 'ڑ': 'R', 'ز': 'z', 'ژ': 'zh',
        'س': 's', 'ش': 'sh', 'ص': 's', 'ض': 'z', 'ط': 't', 'ظ': 'z',
        'ع': '', 'غ': 'gh', 'ف': 'f', 'ق': 'q', 'ک': 'k', 'گ': 'g',
        'ل': 'l', 'م': 'm', 'ن': 'n', 'ں': 'n', 'و': 'o', 'ہ': 'h',
        'ھ': 'h', 'ء': '', 'ی': 'i', 'ے': 'e', 'ئ': 'y',
        'اے': 'ae', 'او': 'oo', '\u0627\u06CC': 'ai',
        '\u064e': 'a', '\u064f': 'u', '\u0650': 'i',
        '\u0651': '', '\u0652': '', '\u0654': '',
        '،': ',', '؟': '?', '۔': '.', '؛': ';',
    }
    result, i = [], 0
    while i < len(urdu_text):
        if i + 1 < len(urdu_text) and urdu_text[i:i+2] in URDU_ROMAN:
            result.append(URDU_ROMAN[urdu_text[i:i+2]])
            i += 2
        elif urdu_text[i] in URDU_ROMAN:
            result.append(URDU_ROMAN[urdu_text[i]])
            i += 1
        elif urdu_text[i] == ' ':
            result.append(' ')
            i += 1
        elif urdu_text[i] in '0123456789.,!?():;\n':
            result.append(urdu_text[i])
            i += 1
        else:
            if ord(urdu_text[i]) < 128:
                result.append(urdu_text[i])
            i += 1
    return re.sub(r' +', ' ', ''.join(result)).strip()
