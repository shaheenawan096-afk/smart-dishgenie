"""
services/context_model.py
-------------------------
MODULE 2 inside the app: dish  ->  labels (hot / cold / mild / rainy / breakfast / main / snack)
predicted by YOUR fine-tuned all-mpnet-base-v2 model (trained on Food.com tags + monsoon recipe lists).
Older 6-label files (no rainy) still work: the labels are read from the csv header / info.json.

Two ways the app gets the labels (both optional, the app still works without them):

  1) data/module2_predictions.csv      (small file, made once in Colab for every dish in recipes.csv)
        -> instant lookup, nothing heavy runs inside the app.

  2) data/module2_model/               (the folder from Colab: model.pt, info.json, tokenizer files)
        -> used only for dishes that are NOT in the csv, e.g. a dish the user just added.
        Needs  torch + transformers  installed.  If they are missing this step is skipped.
"""

import csv
import json
import os

import streamlit as st

_BASE      = os.path.dirname(os.path.abspath(__file__))
_PRED_CSV  = os.path.join(_BASE, "module2_predictions.csv")
_MODEL_DIR = os.path.join(_BASE, "module2_model")

ALL_LABELS = ["hot", "cold", "mild", "rainy", "breakfast", "main", "snack"]
LABELS     = ["hot", "cold", "mild", "breakfast", "main", "snack"]      # default (old model, no rainy)
TEMPS      = ["hot", "cold", "mild"]
MEALS      = ["breakfast", "main", "snack"]

_runtime: dict = {}          # labels for dishes predicted while the app is running


def _key(name: str, cuisine: str = "") -> tuple:
    return (name.strip().lower(), cuisine.strip().lower())


@st.cache_resource(show_spinner=False)
def _load_table() -> dict:
    table = {}
    if os.path.exists(_PRED_CSV):
        with open(_PRED_CSV, encoding="utf-8", newline="") as f:
            rd = csv.DictReader(f)
            labels = [k for k in ALL_LABELS if k in (rd.fieldnames or [])]
            for r in rd:
                lab = {k: int(float(r[k])) for k in labels}
                try:
                    prob = {k: float(r["p_" + k]) for k in labels}      # the model's confidence 0..1
                except (KeyError, ValueError):
                    prob = None
                table[_key(r["Dish Name"], r.get("Cuisine", ""))] = (lab, prob)
                table.setdefault(_key(r["Dish Name"]), (lab, prob))        # name-only fallback (no cuisine given)
    return table


def labels_from_probs(p, labels=None, info=None) -> dict:
    """probabilities -> 0/1 labels (same rule as the Colab notebook):
    ONE weather (hot/cold/mild, the most likely) + rainy if above its threshold;
    meal: ONE (the most likely) when info.json says meal_rule = "one", else all above the threshold."""
    info = info or {}
    th = info.get("thresholds", {})
    pr = dict(zip(labels or LABELS, p))
    out = {k: 0 for k in pr}
    out[max(TEMPS, key=lambda k: pr.get(k, 0))] = 1
    if "rainy" in pr:
        out["rainy"] = int(pr["rainy"] >= th.get("rainy", 0.5))
    best = max(MEALS, key=lambda k: pr.get(k, 0))
    if info.get("meal_rule") == "one":
        meals = [best]
    else:
        meals = [k for k in MEALS if pr.get(k, 0) >= th.get(k, 0.5)] or [best]
    for k in meals:
        out[k] = 1
    return out


# ── optional: run the model inside the app (for brand-new dishes) ──────────────
@st.cache_resource(show_spinner=False)
def _load_model():
    if not os.path.exists(os.path.join(_MODEL_DIR, "model.pt")):
        return None
    try:
        import torch
        import torch.nn as nn
        from transformers import AutoModel, AutoTokenizer

        info = json.load(open(os.path.join(_MODEL_DIR, "info.json"), encoding="utf-8"))
        labels = info.get("labels", LABELS)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc  = AutoModel.from_pretrained(info["base_model"])
                self.drop = nn.Dropout(0.1)
                self.head = nn.Linear(self.enc.config.hidden_size, len(labels))

            def forward(self, input_ids, attention_mask):
                h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                m = attention_mask.unsqueeze(-1).float()
                return self.head(self.drop((h * m).sum(1) / m.sum(1).clamp(min=1)))

        net = Net()
        net.load_state_dict(torch.load(os.path.join(_MODEL_DIR, "model.pt"), map_location="cpu"))
        net.eval()
        tok = AutoTokenizer.from_pretrained(_MODEL_DIR)
        return net, tok, info.get("max_len", 64), torch, labels, info
    except Exception:
        return None


def _predict_one(rec: dict):
    loaded = _load_model()
    if loaded is None:
        return None
    net, tok, max_len, torch, labels, info = loaded
    text = rec["name"] + " | " + ", ".join(rec["ingredients"])
    enc = tok([text], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    with torch.no_grad():
        p = torch.sigmoid(net(enc["input_ids"], enc["attention_mask"]))[0].tolist()
    return labels_from_probs(p, labels, info), dict(zip(labels, p))


def predicted(rec: dict):
    """(labels, probabilities) from the fine-tuned model, or None if the model files are not in the project."""
    k = _key(rec["name"], rec.get("cuisine", ""))
    table = _load_table()
    if k in table:
        return table[k]
    if k in _runtime:
        return _runtime[k]
    if rec.get("user"):                      # only dishes added by the user are predicted live
        res = _predict_one(rec)
        if res:
            _runtime[k] = res
            return res
    if rec.get("cuisine"):                   # exact (name, cuisine) only: an Indian "Chicken Biryani" prediction
        return None                          # must NOT override the curated desi labels of the Pakistani one
    return table.get(_key(rec["name"]))      # name-only fallback (dish without a cuisine)


def predicted_labels(rec: dict):
    r = predicted(rec)
    return r[0] if r else None


def predicted_probs(rec: dict):
    r = predicted(rec)
    return r[1] if r else None


def model_status() -> str:
    n = sum(1 for k in _load_table() if k[1])          # dishes counted once (name + cuisine)
    has_model = os.path.exists(os.path.join(_MODEL_DIR, "model.pt"))
    if n or has_model:
        return f"Module 2 model labels: {n:,} dishes" + (" + live model" if has_model else "")
    return ""
