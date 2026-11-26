"""
services/image_generator.py
----------------------------
OFFLINE PHOTO-REALISTIC dish images.

  Realistic Vision V5.1  (Stable Diffusion 1.5 model trained for real photographs)
  + sd-vae-ft-mse        (clean colours)
  + LCM-LoRA             (8 steps instead of ~30, so it is usable on a laptop CPU)

The prompt is built from the REAL recipe: dish name + cuisine + its main visible
ingredients (salt, oil, water, spices etc. are skipped because you cannot see them).
"""
import io
import os
import re
import zlib
import threading
import traceback
from collections import deque
import streamlit as st
from config import (IMAGE_MODEL, IMAGE_VAE, IMAGE_LORA, IMAGE_STEPS, IMAGE_SIZE, get_device,
                             IMAGE_ENGINE, TURBO_MODEL, TURBO_STEPS, TURBO_SIZE)

TURBO = IMAGE_ENGINE == "turbo"
_SIZE = TURBO_SIZE if TURBO else IMAGE_SIZE

# every picture is saved here, so the same dish never has to be painted twice (even after a restart)
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dish_images")

_pipe = None
_pipe_lock = threading.Lock()      # only one picture is painted at a time

# pantry items that do not show up in a food photo
_INVISIBLE = {
    "salt", "pepper", "black pepper", "water", "oil", "cooking oil", "vegetable oil", "sugar", "flour",
    "baking powder", "baking soda", "vinegar", "stock", "broth", "cornstarch", "corn starch", "yeast",
    "turmeric", "cumin", "coriander powder", "chili powder", "red chili powder", "garam masala",
    "paprika", "cinnamon", "nutmeg", "cloves", "cardamom", "bay leaf", "bay leaves", "vanilla",
    "soy sauce", "fish sauce", "oyster sauce", "worcestershire sauce", "ice", "cooking spray",
}

NEGATIVE = ("cartoon, illustration, painting, drawing, anime, 3d render, sketch, text, watermark, logo, "
            "blurry, low quality, deformed, ugly, oversaturated, plastic, people, hands, face, "
            "empty plate, empty bowl, bad plating, extra food, multiple dishes, messy")

# how a dish of this kind is normally served (keeps the picture believable)
_SERVING = [
    (("biryani", "pulao", "pilau", "rice", "fried rice", "risotto", "paella"), "a mound of rice dish served on a plate"),
    (("curry", "karahi", "korma", "masala", "nihari", "haleem", "salan", "gravy", "stew", "dal", "daal"), "served in a bowl, rich thick gravy"),
    (("soup", "broth", "shorba", "ramen", "pho"), "served hot in a deep bowl, steam rising"),
    (("salad",), "fresh salad in a bowl"),
    (("cake", "pudding", "halwa", "kheer", "ice cream", "dessert", "cookie", "brownie", "tart", "pie", "gulab", "barfi", "ladoo"), "dessert served on a small plate"),
    (("pizza",), "a whole pizza on a wooden board"),
    (("pasta", "spaghetti", "penne", "lasagna", "lasagne", "noodle", "macaroni", "fettuccine"), "pasta served on a plate"),
    (("burger", "sandwich", "wrap", "taco", "burrito", "quesadilla", "roll", "paratha", "naan", "roti", "bread"), "served on a plate"),
    (("kebab", "tikka", "bbq", "grill", "roast", "steak", "chop", "seekh", "boti"), "grilled meat on a plate"),
    (("juice", "smoothie", "lassi", "shake", "drink", "tea", "coffee", "beverage"), "served in a glass"),
]
_CUISINE_LOOK = {
    "Pakistani": "traditional Pakistani food, garnished with fresh coriander",
    "Indian": "traditional Indian food, garnished with fresh coriander",
    "Chinese": "Chinese restaurant style food",
    "Italian": "Italian trattoria style food",
    "Mexican": "Mexican street food style",
    "Thai": "Thai restaurant style food, fresh herbs",
}
# stop-words in dish names that confuse the picture model
_NAME_NOISE = re.compile(r"\b(easy|quick|simple|best|perfect|famous|my|grandma'?s|mom'?s|homemade|style|"
                         r"recipe|crock ?pot|slow cooker|instant pot|dutch oven|leftovers?|healthy|low fat|"
                         r"low calorie|restaurant|copycat|authentic|traditional|ultimate|the)\b", re.I)


def _load():
    global _pipe
    if _pipe is None:
        import torch
        dev = get_device()
        dtype = torch.float16 if dev == "cuda" else torch.float32
        if TURBO:
            # SD-Turbo: the small fp16 files are downloaded and turned into float32 for the CPU
            from diffusers import AutoPipelineForText2Image
            try:
                pipe = AutoPipelineForText2Image.from_pretrained(TURBO_MODEL, torch_dtype=dtype, variant="fp16",
                                                                 safety_checker=None, requires_safety_checker=False)
            except (OSError, ValueError):       # only the normal (non-fp16) files are on this computer
                pipe = AutoPipelineForText2Image.from_pretrained(TURBO_MODEL, torch_dtype=dtype,
                                                                 safety_checker=None, requires_safety_checker=False)
        else:
            from diffusers import StableDiffusionPipeline, LCMScheduler, AutoencoderKL
            vae = AutoencoderKL.from_pretrained(IMAGE_VAE, torch_dtype=dtype)
            pipe = StableDiffusionPipeline.from_pretrained(
                IMAGE_MODEL, vae=vae, torch_dtype=dtype,
                safety_checker=None, requires_safety_checker=False,
            )
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            pipe.load_lora_weights(IMAGE_LORA)
            pipe.fuse_lora()
        pipe = pipe.to(dev)
        if dev == "cpu":
            try:
                torch.set_num_threads(max(1, os.cpu_count() or 1))     # use every CPU core
            except Exception:
                pass
        pipe.set_progress_bar_config(disable=True)
        _pipe = pipe
    return _pipe


def _serving_for(name: str, category: str) -> str:
    cat = category.lower()
    for words, serving in _SERVING:            # the dataset's Category (dessert, soup, beverage...) is the most reliable hint
        if cat and any(w == cat or w == cat.rstrip("s") for w in words):
            return serving
    text = name.lower()
    for words, serving in _SERVING:
        if any(w in text for w in words):
            return serving
    return "served on a plate"


def build_prompt(name: str, cuisine: str, ingredients, category: str = "") -> str:
    """
    SHORT, focused prompt (Stable Diffusion only reads the first ~75 words).
    Order = most important first: the dish, how it is served, 3 visible ingredients, photo quality.
    """
    clean = _NAME_NOISE.sub(" ", name)
    clean = re.sub(r"[^A-Za-z\u00C0-\u024F' -]", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip() or name

    visible = []
    for ing in ingredients:
        i = re.sub(r"\b(fresh|chopped|sliced|diced|minced|ground|large|small|boneless|skinless)\b", "", ing.strip().lower())
        i = re.sub(r"\s+", " ", i).strip()
        if (i and i not in _INVISIBLE and i not in visible
                and not any(w in i for w in ("powder", "sauce", "paste", "stock", "broth", "masala", "seasoning", "spice"))):
            visible.append(i)
    main = ", ".join(visible[:4])
    with_part = f", with {main}" if main else ""
    look = _CUISINE_LOOK.get(cuisine, f"{cuisine} food")
    return (f"professional food photography of {clean}, {look}{with_part}, "
            f"{_serving_for(name, category)}, freshly cooked, appetizing, close-up, "
            f"soft natural light, shallow depth of field, sharp focus, highly detailed")


def _cache_path(name: str, cuisine: str, ingredients: tuple, category: str) -> str:
    key = f"{name}|{cuisine}|{category}|{'|'.join(ingredients)}|{IMAGE_ENGINE}|{_SIZE}|{TURBO_STEPS if TURBO else IMAGE_STEPS}"
    return os.path.join(_CACHE_DIR, f"{zlib.crc32(key.encode('utf-8')):08x}.jpg")


def cached_dish_image(name: str, cuisine: str, ingredients: tuple, category: str = ""):
    """The saved picture of this dish, or None if it has not been painted yet (never runs the model)."""
    path = _cache_path(name, cuisine, ingredients, category)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    return None


def _paint(name: str, cuisine: str, ingredients: tuple, category: str = "") -> bytes:
    """Runs the picture model once and saves the JPEG to disk. RAISES on failure."""
    saved = cached_dish_image(name, cuisine, ingredients, category)
    if saved:
        return saved
    import torch
    with _pipe_lock:
        pipe = _load()
        gen = torch.Generator(device="cpu").manual_seed(zlib.crc32(name.encode("utf-8")))   # different dish -> different seed
        if TURBO:     # SD-Turbo is trained to work WITHOUT guidance (guidance_scale must be 0)
            img = pipe(prompt=build_prompt(name, cuisine, ingredients, category),
                       num_inference_steps=TURBO_STEPS, guidance_scale=0.0,
                       height=_SIZE, width=_SIZE, generator=gen).images[0]
        else:
            img = pipe(
                prompt=build_prompt(name, cuisine, ingredients, category),
                negative_prompt=NEGATIVE,
                num_inference_steps=IMAGE_STEPS,
                guidance_scale=2.0,    # LCM-LoRA works best between 1 and 2
                height=_SIZE, width=_SIZE,
                generator=gen,
            ).images[0]
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    data = buf.getvalue()
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(_cache_path(name, cuisine, ingredients, category), "wb") as f:
            f.write(data)
    except OSError:
        pass
    return data


def generate_dish_image(name: str, cuisine: str, ingredients: tuple, category: str = "") -> bytes:
    """Paints the picture right now (waits). Used by scripts; the app uses request_image() instead."""
    return _paint(name, cuisine, ingredients, category)


# ---------------------------------------------------------------------------
# BACKGROUND PAINTER
# The website must never wait for the picture model (on a laptop CPU it can take
# minutes to load). Pictures are queued here and painted by ONE background thread;
# the page only checks every few seconds whether the saved picture is ready.
# ---------------------------------------------------------------------------
_queue = deque()
_status = {}                 # cache path -> "waiting" | "painting" | "done" | "error"
_errors = {}                 # cache path -> short reason
_model_error = None          # set when the model itself cannot be loaded (then nothing is painted)
_qlock = threading.Lock()
_wake = threading.Event()
_worker = None


def _friendly(e: Exception) -> str:
    msg = f"{type(e).__name__}: {e}"
    low = msg.lower()
    if any(k in low for k in ("offline", "localentrynotfound", "cannot find the requested files", "does not appear to have",
                              "no such file", "not found in the cache", "couldn't connect", "outgoing traffic has been disabled")):
        return "The picture model is not downloaded yet. Close the app, run  python download_models.py  and start the app again."
    if isinstance(e, MemoryError) or "out of memory" in low or "not enough memory" in low or "defaultcpuallocator" in low:
        return "Not enough memory (RAM) to make pictures. Close other programs and try again."
    if isinstance(e, ImportError):
        return ("A package is missing (" + str(getattr(e, "name", "") or e) + "). Run:  pip install -r requirements.txt  "
                "then  python download_models.py  and start the app again.")
    return msg[:220]


def _run_worker():
    global _model_error
    while True:
        _wake.wait()
        with _qlock:
            if not _queue:
                _wake.clear()
                continue
            args = _queue.popleft()
        key = _cache_path(*args)
        if _model_error:
            _status[key], _errors[key] = "error", _model_error
            continue
        _status[key] = "painting"
        try:
            _paint(*args)
            _status[key] = "done"
        except Exception as e:
            reason = _friendly(e)
            print("[DishGenie] picture failed for", args[0], "->", repr(e))
            traceback.print_exc()
            if _pipe is None:                 # the model could not even be loaded: stop trying for every dish
                _model_error = reason
            _status[key], _errors[key] = "error", reason


def request_image(name: str, cuisine: str, ingredients: tuple, category: str = "", first: bool = False) -> None:
    """Ask for a picture without waiting. first=True puts it at the front (the dish the user opened)."""
    global _worker
    args = (name, cuisine, tuple(ingredients), category)
    key = _cache_path(*args)
    if os.path.exists(key) or _status.get(key) in ("waiting", "painting"):
        return
    if _status.get(key) == "error" and not _model_error:
        return                               # this dish already failed once; do not loop on it
    with _qlock:
        _status[key] = "waiting"
        (_queue.appendleft if first else _queue.append)(args)
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_run_worker, name="dish-picture-painter", daemon=True)
            _worker.start()
        _wake.set()


def pictures_available() -> bool:
    """True only when the picture model is on this computer and has not failed to load.
    When it is missing, the website simply shows no picture box at all."""
    if _model_error:
        return False
    try:
        import importlib.util
        if importlib.util.find_spec("diffusers") is None:
            return False
        from huggingface_hub import try_to_load_from_cache
        if TURBO:
            need = ((TURBO_MODEL, "model_index.json"), (TURBO_MODEL, "unet/config.json"))
        else:
            need = ((IMAGE_MODEL, "model_index.json"), (IMAGE_VAE, "config.json"), (IMAGE_LORA, "pytorch_lora_weights.safetensors"))
        for repo, f in need:
            hit = try_to_load_from_cache(repo, f)
            if not isinstance(hit, str):
                return False
        return True
    except Exception:
        return False


def drop_waiting() -> None:
    """A new search was made: forget pictures still waiting for the old results (they would only delay the new ones)."""
    with _qlock:
        for args in _queue:
            _status.pop(_cache_path(*args), None)
        _queue.clear()


def image_state(name: str, cuisine: str, ingredients: tuple, category: str = ""):
    """('done', jpeg bytes) | ('error', reason) | ('waiting', None)"""
    key = _cache_path(name, cuisine, tuple(ingredients), category)
    data = cached_dish_image(name, cuisine, tuple(ingredients), category)
    if data:
        return "done", data
    if _status.get(key) == "error":
        return "error", _errors.get(key) or _model_error or "Picture could not be made."
    return "waiting", None


def retry_pictures() -> None:
    """Forget earlier failures (e.g. after the user downloaded the model)."""
    global _model_error, _pipe
    _model_error = None
    for k, v in list(_status.items()):
        if v == "error":
            _status.pop(k, None)
            _errors.pop(k, None)
