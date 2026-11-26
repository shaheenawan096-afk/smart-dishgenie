"""
services/speech_to_text.py
---------------------------
OFFLINE voice input.
  English, Urdu     -> Whisper (faster-whisper)
  Pashto, Sindhi    -> Meta MMS-1B-all (wav2vec2 + language adapters). Whisper is very weak on these two.
"""
import os
import tempfile
from config import WHISPER_SIZE, WHISPER_CODES, MMS_ASR_MODEL, MMS_ASR_CODES, get_device, ONLINE, GSR_CODES

_model = None
_mms = None

# A short hint helps Whisper expect food words instead of random sentences.
_HINTS = {
    "English": "Ingredients: tomato, onion, garlic, ginger, chicken, rice, potato, green chili.",
    "Urdu":    "آلو، پیاز، ٹماٹر، لہسن، ادرک، ہری مرچ، چکن، چاول، دال، گاجر، مٹر، پالک، دہی، نمک",
}


def _load():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")
    return _model


def _load_mms():
    global _mms
    if _mms is None:
        from transformers import Wav2Vec2ForCTC, AutoProcessor
        proc = AutoProcessor.from_pretrained(MMS_ASR_MODEL)
        model = Wav2Vec2ForCTC.from_pretrained(MMS_ASR_MODEL).to(get_device()).eval()
        _mms = (proc, model)
    return _mms


def _transcribe_mms(path: str, language: str) -> str:
    """Pashto / Sindhi speech -> Arabic-script words. MMS has no punctuation, so words are joined with
    the Arabic comma: every spoken word is treated as one ingredient (say one ingredient per word)."""
    import torch
    from faster_whisper.audio import decode_audio          # decodes any audio file to 16 kHz mono float32
    code = MMS_ASR_CODES[language]
    proc, model = _load_mms()
    wav = decode_audio(path, sampling_rate=16000)[: 16000 * 60]       # max 60 seconds
    proc.tokenizer.set_target_lang(code)
    model.load_adapter(code)
    inputs = proc(wav, sampling_rate=16000, return_tensors="pt").to(get_device())
    with torch.inference_mode():
        logits = model(**inputs).logits
    text = proc.decode(torch.argmax(logits, dim=-1)[0])
    return "، ".join(w for w in text.split() if w)


def transcribe(audio_bytes: bytes, language: str = "English") -> str:
    """Audio bytes (wav/mp3/m4a...) -> text in the spoken language."""
    if ONLINE:
        return _transcribe_google(audio_bytes, language)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        path = f.name
    try:
        if language in MMS_ASR_CODES:
            return _transcribe_mms(path, language)
        segments, _ = _load().transcribe(
            path, language=WHISPER_CODES[language], beam_size=5, vad_filter=True,
            initial_prompt=_HINTS.get(language), condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segments).strip()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def _transcribe_google(audio_bytes: bytes, language: str) -> str:
    """Online version: Google Speech Recognition (free web API) through the SpeechRecognition package.
    st.audio_input records WAV, which SpeechRecognition can read directly."""
    import io
    import speech_recognition as sr
    r = sr.Recognizer()
    with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
        audio = r.record(source)
    try:
        return r.recognize_google(audio, language=GSR_CODES.get(language, "en-US"))
    except sr.UnknownValueError:
        return ""
