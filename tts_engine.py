"""
services/tts_engine.py
-----------------------
OFFLINE text-to-speech.
  English -> Kokoro-82M (hexgrad/Kokoro-82M): natural, human-like voice.
             If Kokoro is not installed, it falls back to facebook/mms-tts-eng.
  Urdu    -> facebook/mms-tts-urd-script_arabic   (Roman Urdu is spoken using the Urdu text)
No free offline voice model exists for Pashto / Sindhi, so those stay text-only.
"""
import io
import re
import wave
import numpy as np
import streamlit as st
from config import TTS_MODELS, KOKORO_MODEL, KOKORO_VOICE, get_device, ONLINE, GTTS_CODES

_models = {}
_kokoro = None          # None = not tried yet, False = not available

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
         "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def _n2w(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + (" " + _ONES[n % 10] if n % 10 else "")
    if n < 1000:
        return _ONES[n // 100] + " hundred" + (" " + _n2w(n % 100) if n % 100 else "")
    if n < 1_000_000:
        return _n2w(n // 1000) + " thousand" + (" " + _n2w(n % 1000) if n % 1000 else "")
    return " ".join(_ONES[int(d)] for d in str(n))


def _frac(m):
    a, b = int(m.group(1)), int(m.group(2))
    known = {(1, 2): "one half", (1, 4): "one quarter", (3, 4): "three quarters", (1, 3): "one third"}
    return known.get((a, b), f"{_n2w(a)} over {_n2w(b)}")


def _prep_english(text: str) -> str:
    """The English voice model skips digits, so write numbers as words."""
    t = re.sub(r"(?m)^\s*\d+\.\s+", "", text)                                   # list numbering "1. "
    t = re.sub(r"(\d)\s*[-–]\s*(\d)", r"\1 to \2", t)                           # 5-10 -> 5 to 10
    t = re.sub(r"(\d+)\s*[°º]\s*([CF])",
               lambda m: f"{_n2w(int(m.group(1)))} degrees {'celsius' if m.group(2) == 'C' else 'fahrenheit'}", t)
    t = re.sub(r"(\d+)\s*/\s*(\d+)", _frac, t)
    t = re.sub(r"(\d+)\.(\d+)",
               lambda m: f"{_n2w(int(m.group(1)))} point {' '.join(_ONES[int(d)] for d in m.group(2))}", t)
    t = re.sub(r"\d+", lambda m: _n2w(int(m.group(0))), t)
    return t


def _prep_urdu(text: str) -> str:
    return re.sub(r"(?m)^\s*\d+\.\s+", "", text)


def _chunks(text: str, limit: int = 200) -> list[str]:
    sents = [s.strip() for s in re.split(r"(?<=[.!?۔؟])\s+|\n+", text) if s.strip()]
    out, cur = [], ""
    for s in sents:
        while len(s) > limit:                       # very long sentence: cut at a space
            cut = s.rfind(" ", 0, limit)
            cut = cut if cut > 30 else limit
            out.append(s[:cut].strip())
            s = s[cut:].strip()
        if cur and len(cur) + len(s) + 1 > limit:
            out.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        out.append(cur)
    return [c for c in out if re.search(r"\w", c)]


def _load(lang: str):
    if lang not in _models:
        from transformers import VitsModel, AutoTokenizer
        name = TTS_MODELS[lang]
        tok = AutoTokenizer.from_pretrained(name)
        model = VitsModel.from_pretrained(name).to(get_device()).eval()
        _models[lang] = (tok, model)
    return _models[lang]


def _to_wav(parts: list, sr: int) -> bytes:
    gap = np.zeros(int(sr * 0.3), dtype=np.float32)
    audio = np.concatenate([x for p in parts for x in (p, gap)])
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _load_kokoro():
    """Kokoro English pipeline, or False when the package / model is not available."""
    global _kokoro
    if _kokoro is None:
        try:
            from kokoro import KPipeline
            _kokoro = KPipeline(lang_code="a", repo_id=KOKORO_MODEL, device=get_device())
        except Exception:
            try:                                   # older kokoro versions have no device argument
                from kokoro import KPipeline
                _kokoro = KPipeline(lang_code="a", repo_id=KOKORO_MODEL)
            except Exception:
                _kokoro = False
    return _kokoro


def _kokoro_wav(text: str):
    """WAV bytes from Kokoro, or None if Kokoro cannot be used."""
    pipe = _load_kokoro()
    if not pipe:
        return None
    parts = []
    for chunk in _chunks(_prep_english(text), limit=350):
        for r in pipe(chunk, voice=KOKORO_VOICE, speed=0.95):
            audio = getattr(r, "audio", None)
            if audio is None and isinstance(r, (tuple, list)):
                audio = r[2]
            if audio is not None:
                parts.append(np.asarray(audio.cpu() if hasattr(audio, "cpu") else audio, dtype=np.float32).reshape(-1))
    return _to_wav(parts, 24000) if parts else None


@st.cache_data(show_spinner=False, max_entries=32)
def text_to_speech(text: str, lang: str) -> bytes:
    """
    lang = 'English' or 'Urdu'.  Returns WAV bytes (offline models) or MP3 bytes (online version, gTTS).
    RAISES on failure (so errors are not cached).
    """
    if ONLINE:
        from gtts import gTTS
        text = _prep_english(text) if lang == "English" else _prep_urdu(text)
        buf = io.BytesIO()
        gTTS(text=text, lang=GTTS_CODES[lang]).write_to_fp(buf)
        data = buf.getvalue()
        if not data:
            raise ValueError("Nothing to speak.")
        return data
    if lang == "English":
        try:
            wav = _kokoro_wav(text)
            if wav:
                return wav
        except Exception:
            pass                                   # fall back to MMS below
    import torch
    tok, model = _load(lang)
    dev = get_device()
    text = _prep_english(text) if lang == "English" else _prep_urdu(text)
    parts = []
    for chunk in _chunks(text):
        inputs = tok(chunk, return_tensors="pt").to(dev)
        if inputs["input_ids"].shape[-1] == 0:
            continue
        torch.manual_seed(555)                       # same voice every time
        with torch.inference_mode():
            wav = model(**inputs).waveform[0].cpu().numpy().astype(np.float32)
        parts.append(wav)
    if not parts:
        raise ValueError("Nothing to speak.")
    return _to_wav(parts, model.config.sampling_rate)
