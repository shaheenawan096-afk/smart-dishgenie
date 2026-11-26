"""
services/image_recognizer.py
-----------------------------
OFFLINE photo -> ingredients using CLIP (openai/clip-vit-large-patch14, zero-shot).

A photo with many ingredients is checked as: the whole photo + 4 overlapping
corner crops, so several different ingredients can be found in one picture.
"""
from config import CLIP_MODEL, get_device

INGREDIENT_LABELS = [
    # vegetables
    "tomato", "onion", "garlic", "ginger", "potato", "carrot", "cucumber", "spinach", "cauliflower",
    "cabbage", "broccoli", "green peas", "green beans", "bell pepper", "green chili", "red chili",
    "eggplant", "okra", "pumpkin", "bottle gourd", "bitter gourd", "radish", "turnip", "beetroot",
    "lettuce", "celery", "zucchini", "corn", "sweet potato", "mushroom", "spring onion",
    "coriander leaves", "mint leaves", "curry leaves", "fenugreek leaves", "asparagus", "leek",
    "avocado", "jalapeno", "capsicum", "mustard greens", "bok choy", "bean sprouts",
    # fruits
    "lemon", "lime", "apple", "banana", "mango", "orange", "pineapple", "grapes", "strawberry",
    "watermelon", "pomegranate", "papaya", "coconut", "peach", "pear", "dates", "fig", "tamarind",
    # meat, fish, eggs
    "chicken", "beef", "mutton", "lamb", "fish", "shrimp", "egg", "minced meat", "sausage", "bacon",
    "tofu",
    # dals, beans, nuts
    "lentils", "chickpeas", "kidney beans", "black beans", "mung beans", "peanuts", "almonds",
    "cashews", "walnuts", "pistachios", "raisins", "sesame seeds",
    # dairy
    "milk", "yogurt", "butter", "cheese", "cream", "paneer", "ghee",
    # grains, bread, pasta
    "rice", "wheat flour", "bread", "pasta", "noodles", "oats", "semolina", "tortilla", "cornmeal",
    "spaghetti", "rice noodles",
    # spices, herbs, sauces
    "cumin seeds", "coriander seeds", "turmeric", "black pepper", "cinnamon", "cardamom", "cloves",
    "bay leaf", "star anise", "saffron", "red chili powder", "salt", "sugar", "mustard seeds",
    "fennel seeds", "basil", "oregano", "parsley", "rosemary", "thyme", "lemongrass", "galangal",
    "soy sauce", "olive oil", "cooking oil", "vinegar", "honey", "coconut milk", "tomato paste",
    "tomato sauce", "fish sauce", "chocolate", "baking powder", "garam masala",
]

_PROMPTS = [f"a photo of {x}, a cooking ingredient" for x in INGREDIENT_LABELS]
_model = None
_proc = None


def _load():
    global _model, _proc
    if _model is None:
        from transformers import CLIPModel, CLIPProcessor
        _proc = CLIPProcessor.from_pretrained(CLIP_MODEL)
        _model = CLIPModel.from_pretrained(CLIP_MODEL).to(get_device()).eval()
    return _model, _proc


def detect_ingredients(image, max_items: int = 8):
    """
    image: PIL.Image.  Returns [(ingredient, probability), ...] best first.
    Thresholds (0.15 whole photo / 0.35 crops) can be tuned if you get too many / too few.
    """
    import torch
    model, proc = _load()
    dev = get_device()
    w, h = image.size
    crops = [image]
    for x0, y0, x1, y1 in [(0, 0, .6, .6), (.4, 0, 1, .6), (0, .4, .6, 1), (.4, .4, 1, 1)]:
        crops.append(image.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))))

    inputs = proc(text=_PROMPTS, images=crops, return_tensors="pt", padding=True).to(dev)
    with torch.inference_mode():
        probs = model(**inputs).logits_per_image.softmax(dim=-1).cpu()   # (5 crops, n_labels)

    best = {}
    for ci in range(len(crops)):
        k, thr = (3, 0.15) if ci == 0 else (1, 0.35)
        top = probs[ci].topk(k)
        for score, li in zip(top.values.tolist(), top.indices.tolist()):
            if score >= thr:
                name = INGREDIENT_LABELS[li]
                best[name] = max(best.get(name, 0.0), score)
    return sorted(best.items(), key=lambda x: -x[1])[:max_items]
