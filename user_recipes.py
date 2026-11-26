"""
services/user_recipes.py
------------------------
Lets the USER add, edit and delete their own dishes.

Storage (the original recipes.csv is never touched):
    data/user_recipes.csv      one row per user dish
    data/user_dishes/<id>.png  optional photo of the dish

Every dish added here is searched exactly like the built-in ones.
"""

import csv
import os
import uuid

_BASE   = os.path.dirname(os.path.abspath(__file__))
_CSV    = os.path.join(_BASE, "user_recipes.csv")
_IMGDIR = os.path.join(_BASE, "user_dishes")
_FIELDS = ["ID", "Dish Name", "Cuisine", "Category", "Ingredient Names", "Instructions", "Image"]

CUISINES   = ["Pakistani", "Indian", "Chinese", "Italian", "Mexican", "Thai"]
CATEGORIES = ["main dish", "chicken", "beef", "mutton", "seafood", "vegetarian", "rice", "lentils", "eggs",
              "pasta", "bread", "snacks", "appetizer", "soup", "salad", "dessert", "beverage", "side dish",
              "condiment", "miscellaneous"]


def _read_rows() -> list[dict]:
    if not os.path.exists(_CSV):
        return []
    with open(_CSV, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_rows(rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(_CSV), exist_ok=True)
    with open(_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in _FIELDS})


def _row_to_recipe(row: dict) -> dict:
    img = row.get("Image", "").strip()
    img_path = os.path.join(_IMGDIR, img) if img else ""
    return {
        "id":           row.get("ID", ""),
        "name":         row.get("Dish Name", "").strip(),
        "ingredients":  [i.strip().lower() for i in row.get("Ingredient Names", "").split(",") if i.strip()],
        "cuisine":      row.get("Cuisine", "").strip(),
        "category":     row.get("Category", "").strip(),
        "instructions": row.get("Instructions", "").strip(),
        "image":        img_path if img_path and os.path.exists(img_path) else "",
        "user":         True,
    }


def load_user_recipes() -> list[dict]:
    return [_row_to_recipe(r) for r in _read_rows() if r.get("Dish Name", "").strip()]


def _save_image(dish_id: str, data: bytes | None, filename: str = "") -> str:
    if not data:
        return ""
    os.makedirs(_IMGDIR, exist_ok=True)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in (".png", ".jpg", ".jpeg", ".webp"):
        ext = ".png"
    name = f"{dish_id}{ext}"
    with open(os.path.join(_IMGDIR, name), "wb") as f:
        f.write(data)
    return name


def validate(name: str, ingredients: list[str], instructions: str) -> str | None:
    """Returns an error message, or None if everything is fine."""
    if len(name.strip()) < 2:
        return "Please write the dish name."
    if not ingredients:
        return "Please add at least one ingredient."
    if len(instructions.strip()) < 20:
        return "Please write the cooking steps (at least a couple of sentences)."
    return None


def add_recipe(name, cuisine, category, ingredients, instructions, image_bytes=None, image_name="") -> str:
    rows = _read_rows()
    dish_id = uuid.uuid4().hex[:8]
    rows.append({
        "ID": dish_id, "Dish Name": name.strip(), "Cuisine": cuisine, "Category": category,
        "Ingredient Names": ", ".join(ingredients), "Instructions": instructions.strip(),
        "Image": _save_image(dish_id, image_bytes, image_name),
    })
    _write_rows(rows)
    return dish_id


def update_recipe(dish_id, name, cuisine, category, ingredients, instructions,
                  image_bytes=None, image_name="", remove_image=False) -> None:
    rows = _read_rows()
    for r in rows:
        if r["ID"] == dish_id:
            r.update({"Dish Name": name.strip(), "Cuisine": cuisine, "Category": category,
                      "Ingredient Names": ", ".join(ingredients), "Instructions": instructions.strip()})
            if remove_image and r.get("Image"):
                _delete_image(r["Image"])
                r["Image"] = ""
            if image_bytes:
                if r.get("Image"):
                    _delete_image(r["Image"])
                r["Image"] = _save_image(dish_id, image_bytes, image_name)
    _write_rows(rows)


def _delete_image(filename: str) -> None:
    p = os.path.join(_IMGDIR, filename)
    if filename and os.path.exists(p):
        os.remove(p)


def delete_recipe(dish_id: str) -> None:
    rows = _read_rows()
    for r in rows:
        if r["ID"] == dish_id:
            _delete_image(r.get("Image", ""))
    _write_rows([r for r in rows if r["ID"] != dish_id])
