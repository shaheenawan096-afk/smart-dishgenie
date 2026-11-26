"""
services/config.py
------------------
ONE place for all model names + language settings.
Every model below runs 100% OFFLINE after you run:  python download_models.py
"""

# ── Models (Hugging Face ids) ──────────────────────────────────────────────────
NLLB_MODEL   = "facebook/nllb-200-distilled-600M"          # translation: EN/Urdu/Sindhi/Pashto (~2.5 GB)
CLIP_MODEL   = "openai/clip-vit-large-patch14"             # photo -> ingredients (~1.7 GB)
# Dish pictures. IMAGE_ENGINE picks the picture model:
#   "turbo"     -> SD-Turbo: 2 steps, about 5-10 s per picture on a laptop CPU (~2.6 GB download)   <- default
#   "realistic" -> Realistic Vision + LCM: 6 steps, more photo-like but about 15-30 s per picture (~3.9 GB)
IMAGE_ENGINE = "turbo"
TURBO_MODEL  = "stabilityai/sd-turbo"                      # fast picture model (distilled Stable Diffusion 2.1)
TURBO_STEPS  = 2        # 1 = fastest, 2 = clearly better detail, 3-4 = little extra gain
TURBO_SIZE   = 512      # SD-Turbo was trained at 512 x 512
IMAGE_MODEL  = "SG161222/Realistic_Vision_V5.1_noVAE"      # PHOTO-REALISTIC dish images (~3.5 GB)
IMAGE_VAE    = "stabilityai/sd-vae-ft-mse"                 # image decoder for the model above (~0.3 GB)
IMAGE_LORA   = "latent-consistency/lcm-lora-sdv1-5"        # makes it fast: 8 steps instead of 30 (~0.14 GB)
IMAGE_STEPS  = 6        # LCM needs only 4-8 steps. 6 keeps a laptop CPU at about 15-30 s per picture (use 8-10 with a GPU)
IMAGE_SIZE   = 448      # picture size in pixels (512 = the model's natural size, 448 is ~25% faster on a CPU)
KOKORO_MODEL = "hexgrad/Kokoro-82M"                        # natural English voice, offline (~330 MB)
KOKORO_VOICE = "af_heart"                                  # warm, clear female English voice
TTS_MODELS   = {                                           # voice output (~150 MB each); English is the fallback if Kokoro is missing
    "English": "facebook/mms-tts-eng",
    "Urdu":    "facebook/mms-tts-urd-script_arabic",
}
MMS_ASR_MODEL = "facebook/mms-1b-all"                      # voice input for Pashto + Sindhi (~4 GB; Whisper is weak on these)
EMBED_MODEL  = "sentence-transformers/all-mpnet-base-v2"   # recipe search (already used)
WHISPER_SIZE = "medium"  # voice input. "medium" understands Urdu much better than "small" (~1.5 GB, a few seconds slower)

# ── Languages ──────────────────────────────────────────────────────────────────
OUTPUT_LANGUAGES = ["English", "Roman Urdu", "Urdu", "Sindhi", "Pashto"]   # recipe text output
INPUT_LANGUAGES  = ["English", "Urdu", "Pashto", "Sindhi"]                 # voice input
VOICE_LANGUAGES  = ["English", "Roman Urdu", "Urdu"]                       # voice output (no offline TTS exists for Sindhi/Pashto)

NLLB_CODES = {
    "English": "eng_Latn",
    "Urdu":    "urd_Arab",
    "Sindhi":  "snd_Arab",
    "Pashto":  "pbt_Arab",
}
WHISPER_CODES = {"English": "en", "Urdu": "ur"}                 # these are heard by Whisper
MMS_ASR_CODES = {"Pashto": "pus", "Sindhi": "snd"}              # these are heard by Meta MMS-1B-all (ISO 639-3 codes)


def get_device() -> str:
    """'cuda' if an NVIDIA GPU is available, else 'cpu'."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
