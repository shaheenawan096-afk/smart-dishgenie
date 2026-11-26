"""
services/ingredient_lexicon.py
-------------------------------
Offline Urdu-script -> English ingredient list. Used for VOICE input so common
ingredients are translated exactly (translation models are unreliable on single words,
e.g. NLLB turned the word for potato into "come on now").
"""
import re

_RAW = {
    # vegetables
    "آلو": "potato", "پیاز": "onion", "ٹماٹر": "tomato", "لہسن": "garlic", "ادرک": "ginger",
    "ہری مرچ": "green chili", "لال مرچ": "red chili", "مرچ": "chili", "شملہ مرچ": "bell pepper",
    "گاجر": "carrot", "مٹر": "green peas", "پالک": "spinach", "گوبھی": "cauliflower",
    "پھول گوبھی": "cauliflower", "بند گوبھی": "cabbage", "بھنڈی": "okra", "بینگن": "eggplant",
    "کدو": "pumpkin", "لوکی": "bottle gourd", "کریلا": "bitter gourd", "شلجم": "turnip",
    "مولی": "radish", "چقندر": "beetroot", "کھیرا": "cucumber", "ہرا دھنیا": "coriander leaves",
    "دھنیا": "coriander", "پودینہ": "mint leaves", "کڑی پتہ": "curry leaves", "کری پتہ": "curry leaves",
    "ہری پیاز": "spring onion", "مشروم": "mushroom", "مکئی": "corn", "شکرقندی": "sweet potato",
    "سلاد": "lettuce", "بروکلی": "broccoli", "ہری پھلی": "green beans", "فرنچ بینز": "green beans",
    # fruits
    "لیموں": "lemon", "سیب": "apple", "کیلا": "banana", "آم": "mango", "سنترہ": "orange",
    "انار": "pomegranate", "انگور": "grapes", "ناریل": "coconut", "کھجور": "dates",
    "اناناس": "pineapple", "تربوز": "watermelon", "پپیتا": "papaya", "املی": "tamarind",
    # meat, fish, eggs
    "مرغی": "chicken", "چکن": "chicken", "گوشت": "meat", "بیف": "beef", "گائے کا گوشت": "beef",
    "بکرے کا گوشت": "mutton", "مٹن": "mutton", "قیمہ": "minced meat", "مچھلی": "fish",
    "جھینگے": "shrimp", "جھینگا": "shrimp", "انڈا": "egg", "انڈے": "egg", "ٹوفو": "tofu",
    # grains / staples
    "چاول": "rice", "آٹا": "wheat flour", "میدہ": "all-purpose flour", "سوجی": "semolina",
    "ڈبل روٹی": "bread", "پاستا": "pasta", "نوڈلز": "noodles", "جو": "oats", "دلیہ": "oats",
    # dals, beans, nuts
    "دال": "lentils", "مسور کی دال": "red lentils", "مونگ کی دال": "mung beans", "چنے": "chickpeas",
    "کالے چنے": "black chickpeas", "لوبیا": "kidney beans", "راجما": "kidney beans",
    "مونگ پھلی": "peanuts", "بادام": "almonds", "کاجو": "cashews", "اخروٹ": "walnuts",
    "پستہ": "pistachios", "کشمش": "raisins", "تل": "sesame seeds",
    # dairy
    "دودھ": "milk", "دہی": "yogurt", "مکھن": "butter", "گھی": "ghee", "پنیر": "cheese",
    "کریم": "cream", "ملائی": "cream",
    # oils, sauces, sweeteners
    "تیل": "oil", "نمک": "salt", "چینی": "sugar", "شہد": "honey", "سرکہ": "vinegar",
    "سویا ساس": "soy sauce", "ٹماٹر کا پیسٹ": "tomato paste", "ناریل کا دودھ": "coconut milk",
    # spices
    "ہلدی": "turmeric", "زیرہ": "cumin", "سفید زیرہ": "cumin seeds", "کالی مرچ": "black pepper",
    "دار چینی": "cinnamon", "الائچی": "cardamom", "لونگ": "cloves", "تیز پات": "bay leaf",
    "گرم مصالحہ": "garam masala", "ہینگ": "asafoetida", "سونف": "fennel seeds", "میتھی": "fenugreek leaves",
    "رائی": "mustard seeds", "کلونجی": "nigella seeds", "زعفران": "saffron", "جائفل": "nutmeg",
    "دھنیا پاؤڈر": "coriander powder", "لال مرچ پاؤڈر": "red chili powder", "پسی ہوئی لال مرچ": "red chili powder",
}


def norm_urdu(t: str) -> str:
    """Remove diacritics/joiners and unify look-alike letters so spelling variants still match."""
    t = re.sub(r"[\u064B-\u065F\u0670\u200c\u200d\u200e\u200f]", "", t)
    t = (t.replace("ي", "ی").replace("ى", "ی").replace("ك", "ک").replace("ۀ", "ہ")
          .replace("ة", "ہ").replace("ھ", "ہ").replace("ۓ", "ے"))
    return re.sub(r"\s+", " ", t).strip()


LEXICON = {norm_urdu(k): v for k, v in _RAW.items()}
_MAX_N = max(len(k.split()) for k in LEXICON)


def lookup(item: str) -> list[str]:
    """'دو ٹماٹر' / 'ہری مرچ' -> ['tomato'] / ['green chili']. [] if no known ingredient inside."""
    tokens = norm_urdu(item).split()
    out, i = [], 0
    while i < len(tokens):
        for n in range(min(_MAX_N, len(tokens) - i), 0, -1):    # longest phrase first
            phrase = " ".join(tokens[i:i + n])
            if phrase in LEXICON:
                if LEXICON[phrase] not in out:
                    out.append(LEXICON[phrase])
                i += n
                break
        else:
            i += 1
    return out
