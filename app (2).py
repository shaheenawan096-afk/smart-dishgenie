"""
app.py — Smart DishGenie v4
Run: streamlit run app.py

Structure:
  core/       — ingredient normalization (hashmap + fuzzy)
  services/   — recipe engine, weather and time model, translator, voice, pictures
(Model evaluation is done in the training notebook, not in the app.)
"""
import os
# After `python download_models.py` finished, force every library to use ONLY local files (100% offline).
if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".models_ready")):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

import html as _html
import streamlit as st

st.set_page_config(
    page_title="Smart DishGenie System",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@500;700;800&family=Figtree:wght@400;500;600;700&family=Noto+Nastaliq+Urdu:wght@400;700&display=swap');
:root{
  --ground:#FFFFFF; --surface:#FFFFFF; --ink:#123526; --muted:#4F6B5C; --line:#CFE6D7;
  --green:#1B6B47; --green-deep:#0E3B27; --green-soft:#E4F4EA;
  --saffron:#EDA81B; --saffron-soft:#FFF3D6; --berry:#A8284A; --berry-soft:#FBE7EC;
}
html, body, [class*="css"], .stApp { font-family:'Figtree',system-ui,sans-serif; color:var(--ink); }
/* no blur / fade while the app is loading: the page stays clear, only the spinner shows */
[data-stale="true"], .stale-element, [data-testid="stElementContainer"][data-stale="true"],
[data-testid="element-container"][data-stale="true"] { opacity:1 !important; filter:none !important; transition:none !important; }
.stApp { background:var(--ground); }
.block-container { max-width:1120px; padding-top:4rem; padding-bottom:3rem; }
h1,h2,h3,h4,h5 { font-family:'Bricolage Grotesque','Figtree',sans-serif; letter-spacing:-.01em; color:var(--ink); }

/* ---- hero ---- */
.hero { background:var(--green-deep); border-radius:22px; padding:2.2rem 2.4rem; margin-bottom:1.6rem;
        display:grid; grid-template-columns:1.25fr 1fr; gap:2rem; align-items:center; }
.hero h1 { font-family:'Bricolage Grotesque',sans-serif; font-size:2.7rem; font-weight:800; color:#fff; margin:0; line-height:1.08; }
.hero p  { color:#C6DACD; font-size:1.02rem; margin:.7rem 0 0; max-width:34rem; line-height:1.55; }
.steps { display:flex; flex-direction:column; gap:.6rem; }
.step  { display:flex; align-items:center; gap:.9rem; background:rgba(255,255,255,.07); border-radius:14px; padding:.7rem 1rem; color:#fff; font-weight:600; }
.step b { width:1.9rem; height:1.9rem; border-radius:50%; background:var(--saffron); color:var(--green-deep);
          display:flex; align-items:center; justify-content:center; font-size:.95rem; flex:none; }
.step span { font-weight:400; color:#C6DACD; font-size:.88rem; display:block; }
@media (max-width:820px){ .hero{grid-template-columns:1fr; padding:1.6rem;} .hero h1{font-size:2.1rem;} }

/* ---- section panels ---- */
.panel-title { font-family:'Bricolage Grotesque',sans-serif; font-size:1.25rem; font-weight:700; margin:0 0 .15rem; }
.panel-sub   { color:var(--muted); font-size:.92rem; margin:0 0 .8rem; }
.fdiv { display:flex; align-items:center; gap:12px; color:var(--muted); font-size:.9rem; font-weight:600; margin:1.8rem 0 1rem; }
.fdiv::before,.fdiv::after { content:""; flex:1; height:1px; background:var(--line); }

/* ---- chips ---- */
.chips { display:flex; flex-wrap:wrap; gap:8px; margin:.5rem 0 .8rem; }
.chip  { display:inline-flex; align-items:center; gap:5px; padding:5px 13px; border-radius:999px; font-size:.84rem; font-weight:600; }
.chip-green { background:var(--green-soft); color:var(--green); border:1px solid #BCD9C6; }
.chip-blue  { background:var(--saffron-soft); color:#8A5A00; border:1px solid #F1D48C; }
.chip-gray  { background:#F1F1EE; color:#555; border:1px solid #DDD; }
.miss-chip  { background:var(--berry-soft); color:var(--berry); border:1px solid #F0BFCB; display:inline-flex; align-items:center;
              padding:5px 13px; border-radius:999px; font-size:.84rem; font-weight:600; }

/* ---- recipe cards ---- */
.rcard { background:var(--surface); border-radius:18px; padding:1.1rem 1.2rem .9rem; border:1px solid var(--line); margin-bottom:.5rem; }
.rcard-title { font-family:'Bricolage Grotesque',sans-serif; font-size:1.08rem; font-weight:700; line-height:1.25; margin-bottom:.25rem; }
.rcard-cat   { font-size:.82rem; color:var(--muted); margin-bottom:.7rem; }
.sbar-wrap { background:#E8EEE4; border-radius:6px; height:8px; overflow:hidden; margin:.35rem 0 .35rem; }
.sbar      { height:8px; border-radius:6px; background:var(--green); }
.sbar-note { font-size:.84rem; color:var(--ink); font-weight:600; }
.sbar-miss { font-size:.8rem; color:var(--berry); margin-top:.3rem; }
.badge { display:inline-block; padding:3px 10px; border-radius:999px; font-size:.76rem; font-weight:700; white-space:nowrap; }
.badge-pk{background:#E3F0E7;color:#1D6B4B;} .badge-in{background:#FFF0DC;color:#A5500A;}
.badge-cn{background:#FBE7EC;color:#A8284A;} .badge-it{background:#E5EEFA;color:#1F4E8C;}
.badge-mx{background:#EEF4D8;color:#52690F;} .badge-th{background:#F1E8F8;color:#6A2E94;}

/* ---- weather and time bar ---- */
.ctxbar { display:flex; align-items:center; gap:14px; background:var(--saffron-soft); border:1px solid #F1D48C; border-radius:16px; padding:.8rem 1.1rem; margin:.4rem 0 1.2rem; }
.ctxicon { font-size:2rem; line-height:1; }
.ctxmain { font-family:'Bricolage Grotesque',sans-serif; font-weight:700; font-size:1.08rem; }
.ctxsub  { color:#6B5200; font-size:.92rem; margin-top:2px; }
.ctxtag  { display:inline-block; margin-top:.5rem; background:var(--saffron-soft); color:#8A5A00; border:1px solid #F1D48C; border-radius:999px; padding:3px 11px; font-size:.78rem; font-weight:700; }

/* ---- recipe text (one block per language) ---- */
.rtab-en,.rtab-ru { font-size:1rem; line-height:1.85; padding:1.3rem 1.4rem; background:var(--surface);
                    border-radius:16px; border:1px solid var(--line); border-left:5px solid var(--green); white-space:pre-wrap; }
.rtab-ru { border-left-color:var(--saffron); }
.rtab-ur { font-family:'Noto Nastaliq Urdu',serif; font-size:1.25rem; line-height:2.6; direction:rtl; text-align:right;
           padding:1.4rem 1.5rem; background:var(--surface); border-radius:16px; border:1px solid var(--line);
           border-right:5px solid var(--green); white-space:pre-wrap; }
.dish-title { font-family:'Bricolage Grotesque',sans-serif; font-size:1.9rem; font-weight:800; margin:0; line-height:1.15; }

/* ---- widgets ---- */
.stButton>button { border-radius:12px!important; font-weight:700!important; border:1px solid var(--line)!important; }
.stButton>button[kind="primary"] { background:var(--green)!important; border-color:var(--green)!important; color:#fff!important; }
.stButton>button[kind="primary"]:hover { background:var(--green-deep)!important; border-color:var(--green-deep)!important; }
.stTabs [data-baseweb="tab"] { font-weight:600!important; font-size:.98rem!important; }
.stTabs [aria-selected="true"] { color:var(--green)!important; }
div[data-testid="stTextArea"] textarea { border-radius:14px!important; border:1px solid var(--line)!important; background:var(--surface)!important; font-size:1rem!important; }
div[data-testid="stTextArea"] textarea:focus { border-color:var(--green)!important; box-shadow:0 0 0 2px rgba(29,107,75,.2)!important; }
div[data-testid="stExpander"] { background:var(--surface); border-radius:16px; border:1px solid var(--line); }
.stButton>button:focus-visible, textarea:focus-visible { outline:3px solid var(--saffron)!important; outline-offset:2px; }
/* ================= RESPONSIVE: desktop > 1024px | tablet 641-1024px | phone <= 640px ================= */
* { box-sizing:border-box; }
img, video, canvas, svg { max-width:100%; height:auto; }
.stApp, .block-container { overflow-x:hidden; }
.btn-spacer { height:1.9rem; }

/* touch-friendly controls on every screen */
.stButton>button { min-height:46px; }
div[data-testid="stTextArea"] textarea, div[data-testid="stTextInput"] input { font-size:16px!important; }   /* 16px stops iPhone auto-zoom */
audio { width:100%; }

/* pill style radios (cuisine and language): they wrap to the next line instead of overflowing.
   Works on new and older Streamlit versions. */
div[data-testid="stRadio"] [role="radiogroup"] { flex-wrap:wrap!important; gap:8px!important; }
div[data-testid="stRadio"] label[data-testid="stRadioOption"],
div[data-testid="stRadio"] label[data-baseweb="radio"] { background:var(--surface); border:1px solid var(--line); border-radius:999px;
                                                           padding:7px 14px; margin:0!important; min-height:40px; align-items:center; cursor:pointer; }
div[data-testid="stRadio"] label[data-testid="stRadioOption"][data-selected="true"],
div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) { background:var(--green-soft); border-color:var(--green); font-weight:700; }
div[data-testid="stRadio"] label[data-testid="stRadioOption"] > div > div:first-child,     /* the round dot (new Streamlit) */
div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child { display:none!important; }   /* the round dot (older Streamlit) */

/* tabs scroll sideways on small screens instead of squeezing */
.stTabs [data-baseweb="tab-list"] { overflow-x:auto; scrollbar-width:none; gap:4px; }
.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display:none; }
.stTabs [data-baseweb="tab"] { white-space:nowrap; }

/* ---------- tablet ---------- */
@media (max-width:1024px){
  .block-container { padding:4rem 1.2rem 2.5rem; }
  .hero { padding:1.8rem; gap:1.4rem; }
  .hero h1 { font-size:2.3rem; }
  /* columns wrap to 2 per row so recipe cards never get squeezed */
  div[data-testid="stHorizontalBlock"] { flex-wrap:wrap!important; gap:1rem!important; }
  div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"],
  div[data-testid="stHorizontalBlock"] > div[data-testid="column"] { flex:1 1 calc(50% - 1rem)!important; min-width:calc(50% - 1rem)!important; width:auto!important; }
}

/* ---------- phone ---------- */
@media (max-width:640px){
  .block-container { padding:3.6rem .75rem 2rem; }
  .hero { grid-template-columns:1fr; padding:1.3rem 1.1rem; border-radius:18px; margin-bottom:1.1rem; }
  .hero h1 { font-size:1.85rem; }
  .hero p { font-size:.95rem; }
  .step { padding:.6rem .8rem; }
  .btn-spacer { display:none; }
  div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"],
  div[data-testid="stHorizontalBlock"] > div[data-testid="column"] { flex:1 1 100%!important; min-width:100%!important; }
  .panel-title { font-size:1.12rem; }
  .dish-title { font-size:1.5rem; }
  .rcard { padding:.95rem 1rem .8rem; }
  .rtab-en, .rtab-ru { font-size:.96rem; padding:1rem 1rem; line-height:1.75; }
  .rtab-ur { font-size:1.12rem; line-height:2.4; padding:1rem; }
  .stTabs [data-baseweb="tab"] { font-size:.9rem!important; padding:.5rem .7rem!important; }
  div[data-testid="stRadio"] label[data-testid="stRadioOption"], div[data-testid="stRadio"] label[data-baseweb="radio"] { padding:6px 11px; font-size:.9rem; }
  .fdiv { margin:1.3rem 0 .8rem; }
}

/* ===================== UI: professional theme ===================== */
header[data-testid="stHeader"] { display:none !important; }
.block-container { padding-top:1.1rem !important; }
.stApp { background:var(--ground); }

/* top bar with logo */
.topbar { display:flex; align-items:center; justify-content:space-between; gap:1rem; flex-wrap:wrap;
          background:var(--surface); border:1px solid var(--line); border-radius:18px; padding:.65rem 1.1rem;
          margin-bottom:1rem; box-shadow:0 4px 18px rgba(18,55,42,.06); }
.brand { display:flex; align-items:center; gap:.7rem; text-decoration:none; }
.brand svg { width:46px; height:46px; flex:none; }
.brand-name { font-family:'Bricolage Grotesque',sans-serif; font-weight:800; font-size:1.35rem; line-height:1; color:var(--green-deep); }

.brand-tag { font-size:.78rem; color:var(--muted); margin-top:3px; }
.topstats { display:flex; gap:.5rem; flex-wrap:wrap; }
.tstat { background:#F4F7F2; color:var(--muted); border:1px solid var(--line); border-radius:999px; padding:5px 12px; font-size:.8rem; font-weight:700; white-space:nowrap; }

/* hero */
.hero { position:relative; overflow:hidden; background:linear-gradient(135deg,#12372A 0%,#1D6B4B 100%);
        box-shadow:0 14px 36px rgba(18,55,42,.25); }

.hero > * { position:relative; z-index:1; }

/* tabs as pills */
.stTabs [data-baseweb="tab-list"], [data-testid="stTabs"] [data-baseweb="tab-list"], div[role="tablist"] {
  background:var(--surface)!important; border:1px solid var(--line)!important; border-radius:14px!important; padding:5px!important; gap:4px!important; }
.stTabs [data-baseweb="tab"], [data-testid="stTabs"] button[role="tab"], div[role="tablist"] button[role="tab"] {
  border-radius:10px!important; padding:.55rem 1.1rem!important; height:auto!important; }
.stTabs [aria-selected="true"], div[role="tablist"] button[role="tab"][aria-selected="true"] { background:var(--green-soft)!important; color:var(--green)!important; }
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display:none!important; }

/* recipe cards */
.rcard { position:relative; overflow:hidden; box-shadow:0 3px 14px rgba(18,55,42,.06); transition:transform .15s ease, box-shadow .15s ease; padding-top:1.35rem; }
.rcard:hover { transform:translateY(-3px); box-shadow:0 10px 26px rgba(18,55,42,.12); }
.rcard::before { content:""; position:absolute; top:0; left:0; right:0; height:5px; background:var(--green); }
.rcard.cu-in::before{background:#E07B1F;} .rcard.cu-cn::before{background:#A8284A;} .rcard.cu-it::before{background:#2F6DB5;}
.rcard.cu-mx::before{background:#7A9A1A;} .rcard.cu-th::before{background:#7B3FA8;}

/* footer */
.sitefoot { margin-top:2.5rem; background:var(--green-deep); color:#C6DACD; border-radius:20px; padding:1.4rem 1.6rem;
            display:flex; align-items:center; justify-content:space-between; gap:1rem; flex-wrap:wrap; font-size:.85rem; }
.sitefoot b { color:#fff; }
.sitefoot svg { width:34px; height:34px; }

@media (max-width:640px){
  .topbar { padding:.55rem .8rem; border-radius:14px; }
  .brand svg { width:38px; height:38px; }
  .brand-name { font-size:1.12rem; }
  .brand-tag { display:none; }
  .topstats { display:none; }
    .hero .steps { display:none; }   /* phone: shorter hero, the steps are shown on bigger screens */
  .sitefoot { flex-direction:column; text-align:center; }
}

/* professional hero (single column, no decorations) */
.hero.hero-pro { grid-template-columns:1fr; padding:2.4rem 2.6rem; }
.hero.hero-pro h1 { font-size:2.6rem; max-width:40rem; }
.hero.hero-pro p { max-width:44rem; }
/* recipe card picture */
.dish-thumb img, [data-testid="stImage"] img { border-radius:16px; aspect-ratio:4/3; object-fit:cover; width:100%; }
.thumb-none { animation:none!important; background:#EEF2EA!important; text-align:center; padding:1rem; }
.thumb-why { font-weight:500; font-size:.78rem; margin-top:.4rem; line-height:1.4; }
.thumb-wait { aspect-ratio:4/3; border-radius:16px; background:linear-gradient(110deg,#EEF2EA 8%,#F7F9F5 18%,#EEF2EA 33%);
              background-size:200% 100%; animation:shine 1.4s linear infinite; display:flex; align-items:center; justify-content:center;
              color:var(--muted); font-size:.85rem; font-weight:600; margin-bottom:.5rem; }
@keyframes shine { to { background-position-x:-200%; } }
/* full-screen loading page */
.dg-loading { position:fixed; inset:0; z-index:999999; background:rgba(243,246,239,.97); display:flex; flex-direction:column;
              align-items:center; justify-content:center; gap:1.1rem; text-align:center; padding:1.5rem; }
.dg-loading svg.lg { width:72px; height:72px; }
.dg-spin { width:46px; height:46px; border-radius:50%; border:4px solid #DCE4D6; border-top-color:var(--green); animation:dgspin .9s linear infinite; }
@keyframes dgspin { to { transform:rotate(360deg); } }
.dg-loading h3 { margin:0; font-size:1.3rem; }
.dg-loading p { margin:0; color:var(--muted); font-size:.95rem; max-width:26rem; }
@media (max-width:640px){ .hero.hero-pro { padding:1.5rem 1.2rem; } .hero.hero-pro h1 { font-size:1.75rem; } }
/* ---- approved logo ---- */
.brand svg.dg-logo-header { width:auto!important; height:56px!important; display:block; }
.dg-loading svg.dg-logo-stack { width:170px; height:auto; }
.sitefoot svg.dg-logo-dark { width:240px!important; height:auto!important; display:block; }
.foot-brand { display:flex; flex-direction:column; gap:.2rem; }
/* light hero, text only (as in the approved design) */
.hero.hero-light { background:transparent!important; box-shadow:none!important; padding:1.6rem .3rem 1.4rem!important; }
.hero.hero-light h1 { color:var(--green-deep)!important; font-size:3rem; letter-spacing:-.02em; }
.hero.hero-light p { color:#4A5A50!important; font-size:1.08rem; max-width:40rem; }
.hero-kicker { font-size:.85rem; font-weight:800; color:var(--green); letter-spacing:.08em; text-transform:uppercase; margin-bottom:.7rem; }
.hero-stats { display:flex; gap:2rem; flex-wrap:wrap; margin-top:1.2rem; }
.hero-stats b { display:block; font-family:'Bricolage Grotesque',sans-serif; font-size:1.6rem; font-weight:800; color:var(--green-deep); line-height:1.1; }
.hero-stats span { font-size:.85rem; color:var(--muted); }
@media (max-width:640px){
  .brand svg.dg-logo-header { height:44px!important; }
  .hero.hero-light { padding:1rem .2rem .8rem!important; }
  .hero.hero-light h1 { font-size:1.9rem!important; }
  .hero-stats { gap:1.4rem; } .hero-stats b { font-size:1.3rem; }
  .foot-brand { align-items:center; }
}
/* ---- final polish ---- */
[data-testid="stTabs"] [role="tablist"] { gap:6px!important; }
[data-testid="stTabs"] button[role="tab"], [data-testid="stTab"] { padding:.6rem 1.3rem!important; border-radius:10px!important; }
[data-testid="stTabs"] button[role="tab"] p { font-size:.98rem!important; font-weight:600!important; }
.rcard { min-height:205px; }
.stButton>button:not([kind="primary"]):hover { border-color:var(--green)!important; color:var(--green)!important; background:var(--green-soft)!important; }
@media (max-width:640px){ .rcard { min-height:0; } }
/* ---- full-width website layout (laptop, tablet and mobile) ---- */
.block-container, [data-testid="stMainBlockContainer"] { max-width:100%!important; width:100%!important;
  padding-left:clamp(1rem, 5vw, 5.5rem)!important; padding-right:clamp(1rem, 5vw, 5.5rem)!important; }
.hero.hero-light h1 { max-width:none!important; font-size:clamp(1.9rem, 4.2vw, 3.6rem)!important; }
.hero.hero-light p { max-width:62rem!important; font-size:clamp(.98rem, 1.2vw, 1.15rem)!important; }
@media (max-width:1024px){ .block-container, [data-testid="stMainBlockContainer"] { padding-left:1.5rem!important; padding-right:1.5rem!important; } }
@media (max-width:640px){ .block-container, [data-testid="stMainBlockContainer"] { padding-left:.8rem!important; padding-right:.8rem!important; } }
/* ---- style: Times New Roman, centred hero, dark green / light green / white ---- */
.stApp, .stApp p, .stApp span, .stApp div, .stApp label, .stApp li, .stApp a, .stApp button, .stApp input, .stApp textarea,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, .stApp small, .stApp b, .stApp strong, .stApp em,
.panel-title, .rcard-title, .ctxmain, .dish-title, .hero-stats b, .brand-name {
  font-family:'Times New Roman', Times, serif!important; }
.stApp svg text, .stApp svg tspan { font-family:'Times New Roman', Times, serif!important; }
.stApp span[data-testid="stIconMaterial"], .stApp [data-testid="stIconMaterial"], .stApp span[class*="material-symbols"] {
  font-family:'Material Symbols Rounded'!important; font-weight:400!important; font-style:normal!important; letter-spacing:normal!important; }
.rtab-ur, .rtab-ur *, .dg-rtl, .dg-rtl * { font-family:'Noto Nastaliq Urdu', serif!important; }
/* logo bar without the white box */
.topbar { background:transparent!important; border:none!important; box-shadow:none!important; padding:.3rem 0!important; }
.brand svg text { font-weight:700!important; letter-spacing:.2px!important; }
.brand svg text:last-of-type { letter-spacing:4px!important; }
/* centred hero */
.hero.hero-light, .hero.hero-light > div { text-align:center!important; }
.hero.hero-light h1 { color:var(--green-deep)!important; font-weight:700!important; letter-spacing:0!important; line-height:1.12!important;
  margin-left:auto!important; margin-right:auto!important; }
.hero.hero-light h1 em { font-style:italic; color:var(--green)!important; font-weight:700; }
.hero.hero-light p { margin-left:auto!important; margin-right:auto!important; color:var(--muted)!important; }
.hero-stats { justify-content:center!important; }
.hero-stats b { color:var(--green-deep)!important; }
/* weather bar and tags in light green instead of yellow */
.ctxbar { background:var(--green-soft)!important; border:1px solid #BFE0CC!important; }
.ctxsub { color:var(--green-deep)!important; }
.ctxtag, .chip-blue { background:var(--green-soft)!important; color:var(--green-deep)!important; border-color:#BFE0CC!important; }
.rtab-ru { border-left-color:var(--green)!important; }
.sitefoot { background:var(--green-deep)!important; }
</style>
""", unsafe_allow_html=True)


# ── Imports from modules ───────────────────────────────────────────────────────
from core.normalizer import normalize_all
from services.context_engine import get_live_weather, build_context, suggest_for_context, reason as ctx_reason
from services.recipe_engine import find_best_recipes, get_missing_ingredients, ingredient_coverage_score, matched_ingredients, match_summary
from services.config import OUTPUT_LANGUAGES, INPUT_LANGUAGES, VOICE_LANGUAGES
from services.translator import translate_recipe, ingredients_to_english_checked
from services.tts_engine import text_to_speech
from services.speech_to_text import transcribe
from services.image_recognizer import detect_ingredients
from services.image_generator import request_image, image_state, drop_waiting, pictures_available
from rapidfuzz import fuzz

# ── Full-width widgets: works on old and new Streamlit versions ────────────────
def _stretch_kwargs():
    try:
        major, minor = (int(x) for x in st.__version__.split(".")[:2])
        if (major, minor) >= (1, 50):
            return {"width": "stretch"}
    except Exception:
        pass
    return {"use_container_width": True}
STRETCH = _stretch_kwargs()

# ── Constants ──────────────────────────────────────────────────────────────────
FLAGS  = {"Pakistani":"🇵🇰","Indian":"🇮🇳","Chinese":"🇨🇳","Italian":"🇮🇹","Mexican":"🇲🇽","Thai":"🇹🇭"}
BADGES = {"Pakistani":"badge-pk","Indian":"badge-in","Chinese":"badge-cn","Italian":"badge-it","Mexican":"badge-mx","Thai":"badge-th"}

# ── Session state ──────────────────────────────────────────────────────────────
DEFAULTS = {
    "step": "input", "clean_ings": [], "norm_details": [],
    "matched": [], "chosen": None,
    "recipe_en": "", "upd_recipe_en": "",
    "missing": [], "sel_missing": [], "cuisine": "Pakistani",
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Top bar + Hero ─────────────────────────────────────────────────────────────
from services.recipe_engine import load_recipes as _lr_top
from services.logo import LOGO_HEADER, LOGO_STACK, LOGO_DARK
def show_loading(title: str, note: str = ""):
    """Full-screen loading page. Returns a slot; call .empty() on it to remove the page."""
    slot = st.empty()
    slot.markdown(f"""
<div class="dg-loading">
  {LOGO_STACK}
  <div class="dg-spin"></div>
  <h3>{_html.escape(title)}</h3>
  <p>{_html.escape(note)}</p>
</div>""", unsafe_allow_html=True)
    return slot


def _img_args(rec):
    return rec["name"], rec["cuisine"], tuple(rec["ingredients"][:15]), rec.get("category", "")

def _pic_html(state, val):
    if state == "error":
        return (f'<div class="thumb-wait thumb-none"><div><b>Picture not available</b>'
                f'<div class="thumb-why">{_html.escape(val)}</div></div></div>')
    return '<div class="thumb-wait">Preparing picture...</div>'


@st.fragment(run_every=3)
def _picture_poll(args, caption):
    """Checks every 3 s whether the background painter has finished this picture."""
    state, val = image_state(*args)
    if state != "waiting":
        st.rerun()                     # redraw the page once: with the finished picture, or without the box if it failed
    st.markdown(_pic_html(state, val), unsafe_allow_html=True)


def dish_picture(rec, caption=None, first=False):
    """Shows the dish picture. Never waits for the picture model: it is painted in the background.
    If the picture model is not on this computer (or a picture fails), NOTHING is shown, so the card stays clean."""
    if rec.get("image"):                                   # the user's own photo
        st.image(rec["image"], caption=caption, **STRETCH)
        return
    args = _img_args(rec)
    state, val = image_state(*args)
    if state == "done":
        st.image(val, caption=caption, **STRETCH)
    elif state == "waiting" and pictures_available():
        request_image(*args, first=first)
        _picture_poll(args, caption)



st.markdown(f"""
<div class="topbar">
  <div class="brand">{LOGO_HEADER}</div>
</div>
<div class="hero hero-pro hero-light">
  <div>
    <h1>Turn your ingredients into <em>something delicious.</em></h1>
    <p>Tell us the ingredients in your kitchen. Smart DishGenie suggests dishes you can make, picks the ones that suit
       today's weather and time of day, and reads the recipe to you in English, Roman Urdu, Urdu, Sindhi or Pashto.</p>
    <div class="hero-stats">
      <div><b>{len(_lr_top()):,}</b><span>recipes</span></div>
      <div><b>6</b><span>cuisines</span></div>
      <div><b>5</b><span>languages</span></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_app, tab_add = st.tabs(["Find a recipe", "My dishes"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RECIPE FINDER
# ══════════════════════════════════════════════════════════════════════════════
with tab_app:

    def _add_to_box(items):
        """Append ingredients to the text box, skipping ones already there (must run BEFORE the text box is drawn)."""
        cur = st.session_state.get("ing_box", "").strip().strip(",").strip()
        have = {x.strip().lower() for x in cur.split(",") if x.strip()}
        new = []
        for chunk in items:
            for x in str(chunk).split(","):
                x = x.strip()
                if x and x.lower() not in have:
                    have.add(x.lower())
                    new.append(x)
        if new:
            add = ", ".join(new)
            st.session_state["ing_box"] = f"{cur}, {add}" if cur else add
        return new

    def _open_image(f):
        """Open any image format (jpg/jpeg/png/webp/bmp/jfif/heic...), fix phone rotation, return RGB PIL image."""
        from PIL import Image, ImageOps
        try:
            from pillow_heif import register_heif_opener   # optional: iPhone HEIC photos
            register_heif_opener()
        except Exception:
            pass
        f.seek(0)
        img = Image.open(f)
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")

    with st.expander("Add ingredients by voice or photo (optional)", expanded=False):
        v_tab, p_tab = st.tabs(["Voice", "Photo"])

        with v_tab:
            in_lang = st.radio("Language you will speak", INPUT_LANGUAGES, horizontal=True, key="voice_lang")
            rec_audio = st.audio_input("Press the microphone, say your ingredients, then press stop.", key="voice_rec")
            if st.button("Add spoken ingredients", key="voice_go"):
                src = rec_audio
                if src is None:
                    st.warning("Please record your voice first with the microphone above.")
                else:
                    try:
                        with st.spinner("Listening..."):
                            heard = transcribe(src.getvalue(), in_lang)
                        if not heard.strip():
                            st.warning("We could not hear any words. Please try again a little closer to the microphone.")
                        else:
                            with st.spinner("Understanding your ingredients..."):
                                known, unknown = ingredients_to_english_checked(heard, in_lang)
                            st.session_state["heard"] = (in_lang, heard, known, unknown)
                            _add_to_box(known)
                    except Exception as e:
                        st.error(f"Voice input is not available right now. ({e})")
            heard_info = st.session_state.get("heard")
            if heard_info:
                if heard_info[2]:
                    st.success(f"We heard ({heard_info[0]}): {heard_info[1]}\n\nAdded to your ingredients: **{', '.join(heard_info[2])}**")
                else:
                    st.info(f"We heard ({heard_info[0]}): {heard_info[1]}")
                if heard_info[3]:
                    st.warning("We did not recognise: **" + ", ".join(heard_info[3]) +
                               "**. Please say it again slowly, or type it in the box below.")

        with p_tab:
            mode = st.radio("Photo source", ["Upload photos", "Use camera"], horizontal=True, key="photo_mode")
            if mode == "Upload photos":
                img_files = st.file_uploader("Upload one or more photos of your ingredients (JPG, PNG, WEBP or HEIC)",
                                             accept_multiple_files=True, key="photo_files")
            else:
                shot = st.camera_input("Take a photo. You can take more photos; each one is added to your list.", key="photo_cam")
                img_files = [shot] if shot is not None else []

            if img_files:
                st.caption(f"{len(img_files)} photo(s) selected")
                tcols = st.columns(min(len(img_files), 6))
                for i, f in enumerate(img_files):
                    with tcols[i % len(tcols)]:
                        try:
                            st.image(_open_image(f), width=110)
                        except Exception:
                            st.warning(f"{getattr(f, 'name', 'This file')} could not be opened as an image.")

                if st.button(f"Find ingredients in {len(img_files)} photo(s)", key="photo_go", type="primary"):
                    report, merged = [], {}
                    bar = st.progress(0.0, text="Starting...")
                    try:
                        for i, f in enumerate(img_files):
                            bar.progress(i / len(img_files), text=f"Checking photo {i + 1} of {len(img_files)}...")
                            fname = getattr(f, "name", f"photo {i + 1}")
                            try:
                                img = _open_image(f)
                            except Exception:
                                report.append((fname + " (skipped: could not open this image)", []))
                                continue
                            found = detect_ingredients(img)
                            report.append((fname, found))
                            for name, prob in found:
                                merged[name] = max(merged.get(name, 0.0), prob)
                        bar.empty()
                        added = _add_to_box(sorted(merged, key=lambda n: -merged[n]))
                        st.session_state["photo_report"] = (report, added)
                    except Exception as e:
                        bar.empty()
                        st.error(f"Photo recognition is not available right now. ({e})")

            rep = st.session_state.get("photo_report")
            if rep:
                report, added = rep
                for name, found in report:
                    st.caption(f"{name}: " + (", ".join(n for n, p in found) or "no ingredient recognised"))
                if added:
                    st.success(f"Added to your ingredients: **{', '.join(added)}**. You can remove any wrong item in the box below.")
                else:
                    st.info("No new ingredient was found, or it is already in your list.")

    # ── Weather and time (module 2) ───────────────────────────────────────────
    from datetime import datetime as _dt
    st.markdown('<div class="panel-title">Weather and time of day</div>'
                '<div class="panel-sub">We check the weather in your city and the time of day, and suggest dishes that suit the moment.</div>',
                unsafe_allow_html=True)
    cw1, cw2 = st.columns([2, 3], gap="large")
    with cw1:
        city = st.text_input("City", value="Karachi", key="ctx_city")
    with cw2:
        st.markdown("<div class='btn-spacer' style='height:1.7rem'></div>", unsafe_allow_html=True)
        use_ctx = st.toggle("Match dishes to the weather and time", value=True, key="use_ctx")
    manual = st.checkbox("Set the weather and time manually", key="ctx_manual")

    ctx = None
    if use_ctx:
        live = None if manual else get_live_weather(city)
        if live:
            ctx = build_context(live["temp"], live["sky"], live["hour"], live["minute"], live["city"], "live")
        else:
            now = _dt.now()
            if not manual:
                st.info("We could not get the live weather (no internet connection, or the city was not found). "
                        "Please set the weather below. The time is taken from your device.")
            m1, m2, m3 = st.columns(3, gap="medium")
            t_in  = m1.slider("Temperature (°C)", -5, 48, 30, key="ctx_temp")
            sky   = m2.radio("Sky", ["clear", "cloudy", "rain"], horizontal=True, key="ctx_sky",
                             format_func=lambda x: {"clear": "☀️ Clear", "cloudy": "☁️ Cloudy", "rain": "🌧️ Rain"}[x])
            hr    = m3.slider("Hour of the day", 0, 23, now.hour, key="ctx_hour")
            ctx = build_context(t_in, sky, hr, now.minute if hr == now.hour else 0, "", "manual")
    st.session_state.ctx = ctx

    if ctx:
        icon = "🌧️" if ctx["rainy"] else "☀️" if ctx["temp_class"] == "hot" else "❄️" if ctx["temp_class"] == "cold" else "🌤️"
        st.markdown(f"""
        <div class="ctxbar"><div class="ctxicon">{icon}</div>
          <div><div class="ctxmain">{_html.escape(ctx['summary'])}</div>
          <div class="ctxsub">Suggesting dishes for <b>{ctx['weather_text']}</b> and <b>{ctx['meal_text']}</b>.</div></div></div>""",
            unsafe_allow_html=True)

    st.markdown('<div class="panel-title">Which ingredients do you have?</div>'
                '<div class="panel-sub">Separate them with commas. You can type in English or Roman Urdu, and small spelling mistakes are fine.</div>',
                unsafe_allow_html=True)
    user_input = st.text_area(
        label="ing", label_visibility="collapsed",
        placeholder="For example: rice, onion, tomato, chicken",
        height=130, key="ing_box",
    )
    col_cui, col_go = st.columns([3, 1], gap="large")
    with col_cui:
        st.markdown('<div class="panel-title" style="font-size:1.05rem">Cuisine</div>', unsafe_allow_html=True)
        cuisine = st.radio(
            label="cui", label_visibility="collapsed", horizontal=True,
            options=["Pakistani","Indian","Chinese","Italian","Mexican","Thai"],
            format_func=lambda x: f"{FLAGS[x]} {x}",
            key="cui_radio",
        )
        st.session_state.cuisine = cuisine
    with col_go:
        st.markdown("<div class='btn-spacer'></div>", unsafe_allow_html=True)
        go_btn  = st.button("Find dishes", type="primary", **STRETCH)
        now_btn = st.button("What should I cook now?", **STRETCH, disabled=not use_ctx,
                            help="No ingredients needed. We suggest dishes for the current weather and time." if use_ctx else "Turn on 'Match dishes to the weather and time' first.")

    # AUTOMATIC: when the page opens, suggest dishes for the weather + time right away (no button needed)
    if (ctx and not now_btn and not go_btn and st.session_state.step == "input"
            and not st.session_state.get("auto_ctx_done")):
        st.session_state.auto_ctx_done = True
        _auto = suggest_for_context(cuisine, ctx, top_n=6)
        if _auto:
            st.session_state.cuisine = cuisine
            st.session_state.matched = _auto
            st.session_state.step = "pick"
            st.rerun()

    if now_btn and ctx:
        for k, v in DEFAULTS.items():
            st.session_state[k] = v
        st.session_state.cuisine = cuisine
        drop_waiting()
        _load = show_loading("Finding dishes for you", f"Choosing {cuisine} dishes for {ctx['weather_text']} and {ctx['meal_text']}...")
        try:
            results = suggest_for_context(cuisine, ctx, top_n=6)
        finally:
            _load.empty()
        if not results:
            st.warning(f"We could not find a {cuisine} dish for this weather and time. Try another cuisine.")
            st.stop()
        st.session_state.matched = results
        st.session_state.step = "pick"
        st.rerun()

    if go_btn:
        if not user_input.strip():
            st.warning("Please add at least one ingredient.")
            st.stop()

        for k, v in DEFAULTS.items():
            st.session_state[k] = v
        st.session_state.cuisine = cuisine

        drop_waiting()
        _load = show_loading("Finding dishes for you", f"Searching {cuisine} recipes that use your ingredients...")
        try:
            clean, details = normalize_all(user_input)
            st.session_state.clean_ings   = clean
            st.session_state.norm_details = details
            results = find_best_recipes(clean, cuisine, top_n=5, context=ctx) if clean else []
            st.session_state.matched = results
        finally:
            _load.empty()

        if not clean:
            st.error("We could not recognise these ingredients. Please check the spelling or try the English name.")
            st.stop()

        if not results:
            st.warning(f"No {cuisine} dish uses these ingredients. Try another cuisine or add another ingredient.")
            st.stop()

        st.session_state.step = "pick"
        st.rerun()

    # Normalized chips
    if st.session_state.step != "input" and st.session_state.norm_details:
        st.markdown('<div class="fdiv">Your ingredients</div>', unsafe_allow_html=True)
        html = '<div class="chips">'
        for d in st.session_state.norm_details:
            m     = d["method"]
            cls   = "chip-green" if m == "hashmap" else "chip-blue" if m == "fuzzy" else "chip-gray"
            orig  = _html.escape(d["original"])
            match = _html.escape(d["matched"])
            note  = f' <span style="opacity:.6;font-size:.76rem">(you typed "{orig}")</span>' if orig != match else ""
            html += f'<span class="chip {cls}">{match}{note}</span>'
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

    # Pick recipe
    if st.session_state.step == "pick" and st.session_state.matched:
        st.markdown('<div class="fdiv">Suggested dishes</div>', unsafe_allow_html=True)
        _ctx_mode = not st.session_state.clean_ings
        _cx = st.session_state.get("ctx")
        if _ctx_mode and _cx:
            st.markdown(f'<div class="panel-title">{FLAGS[st.session_state.cuisine]} {st.session_state.cuisine} dishes for {_cx["weather_text"]} and {_cx["meal_text"]}</div>'
                        '<div class="panel-sub">Picked for the weather and time right now. Check again later for new ideas.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="panel-title">{FLAGS[st.session_state.cuisine]} {len(st.session_state.matched)} {st.session_state.cuisine} dishes that use your ingredients</div>'
                        '<div class="panel-sub">The bar shows how many of the dish\'s ingredients you already have.</div>', unsafe_allow_html=True)

        results = st.session_state.matched
        cols = st.columns(min(len(results), 3), gap="medium")
        for i, (score, rec) in enumerate(results):
            ms   = match_summary(st.session_state.clean_ings, rec)
            pct  = int(100 * ms["have"] / ms["total"]) if ms["total"] else 0
            bdg  = BADGES.get(rec["cuisine"], "badge-pk")
            miss_txt = ("Missing: " + ", ".join(_html.escape(m) for m in ms["missing"][:4])) if ms["missing"] else "You have everything"
            why = ctx_reason(rec, _cx) if _cx else ""
            if _ctx_mode:
                need = [i for i in rec["ingredients"] if i not in ("salt", "water", "oil", "pepper", "sugar")][:5]
                body = (f'<div class="sbar-note">{_html.escape(why or "A good choice right now")}</div>'
                        f'<div class="sbar-miss" style="color:var(--muted)">Main ingredients: {_html.escape(", ".join(need))}</div>')
            else:
                body = (f'<div class="sbar-note">You have {ms["have"]} of {ms["total"]} ingredients</div>'
                        f'<div class="sbar-wrap"><div class="sbar" style="width:{max(pct, 6)}%"></div></div>'
                        f'<div class="sbar-miss">{miss_txt}</div>'
                        + (f'<div class="ctxtag">{_html.escape(why)}</div>' if why else ""))
            with cols[i % 3]:
                dish_picture(rec)
                st.markdown(f"""
                <div class="rcard {bdg.replace('badge-', 'cu-')}">
                  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:8px">
                    <div class="rcard-title">{_html.escape(rec['name'])}</div>
                    <span class="badge {bdg}">{rec['cuisine']}</span>
                  </div>
                  <div class="rcard-cat">{_html.escape(rec['category'].title())}{' &nbsp;·&nbsp; <b style="color:#8A5A00">Your dish</b>' if rec.get('user') else ''}</div>
                  {body}
                </div>""", unsafe_allow_html=True)

                if st.button("View recipe", key=f"pick_{i}", **STRETCH):
                    st.session_state.chosen  = rec
                    st.session_state.missing = get_missing_ingredients(
                        st.session_state.clean_ings, rec["ingredients"]) if st.session_state.clean_ings else []
                    st.session_state.recipe_en = rec["instructions"]
                    st.session_state.step = "recipe"
                    st.session_state.pop("_shown_orig", None)
                    st.session_state.pop("_shown_updated", None)
                    st.rerun()

    # Render recipe helper
    def render_recipe(title, en, voice_key):
        # The full loading page is shown only when a recipe is opened. Changing the language
        # afterwards keeps the page in place and shows a small spinner next to the method instead.
        seen = f"_shown_{voice_key}"
        if st.session_state.get(seen) == title:
            _render_recipe(title, en, voice_key)
            return
        _load = show_loading(f"Preparing {title}", "Getting the ingredients, steps and voice ready...")
        try:
            _render_recipe(title, en, voice_key)
            st.session_state[seen] = title
        finally:
            _load.empty()

    def _render_recipe(title, en, voice_key):
        rec  = st.session_state.chosen
        cui  = st.session_state.cuisine
        bdg  = BADGES.get(cui, "badge-pk")
        flag = FLAGS.get(cui, "")

        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;margin:.6rem 0 1rem;flex-wrap:wrap">
          <h2 class="dish-title">{_html.escape(title)}</h2>
          <span class="badge {bdg}">{flag} {cui}</span>
        </div>""", unsafe_allow_html=True)

        user_ings = st.session_state.clean_ings
        used = matched_ingredients(user_ings, rec["ingredients"])
        missing_now = get_missing_ingredients(user_ings, rec["ingredients"])

        _st = image_state(*_img_args(rec))[0] if not rec.get("image") else "done"
        _has_pic = rec.get("image") or _st == "done" or (_st == "waiting" and pictures_available())
        if _has_pic:
            col_pic, col_ing = st.columns([1, 1], gap="large")
        else:                                    # no picture model: ingredients use the full width
            col_pic, col_ing = None, st.container()
        with col_ing:
            st.markdown('<div class="panel-title">Ingredients</div>', unsafe_allow_html=True)
            if user_ings:
                st.markdown('<div class="panel-sub">Green: you have it. Red: you still need it.</div>', unsafe_allow_html=True)
                chips = "".join(f'<span class="chip chip-green">✓ {_html.escape(u)}</span>' for u in used)
                chips += "".join(f'<span class="miss-chip">+ {_html.escape(m)}</span>' for m in missing_now)
            else:
                st.markdown('<div class="panel-sub">Everything you need for this dish.</div>', unsafe_allow_html=True)
                chips = "".join(f'<span class="chip chip-gray">{_html.escape(i)}</span>' for i in rec["ingredients"])
            st.markdown(f'<div class="chips">{chips}</div>', unsafe_allow_html=True)
        # The picture is drawn LAST so the recipe text is never blocked while the image is being generated
        if col_pic is not None:
          with col_pic:
            dish_picture(rec, caption=f"AI-generated picture of {rec['name']}" if not rec.get("image") else rec["name"], first=True)

        st.markdown('<div class="fdiv">Method</div>', unsafe_allow_html=True)
        lang = st.radio("Language", OUTPUT_LANGUAGES, horizontal=True, key=f"lang_{voice_key}",
                        format_func=lambda x: {"English": "English", "Roman Urdu": "Roman Urdu",
                                               "Urdu": "Urdu (اردو)", "Sindhi": "Sindhi (سنڌي)",
                                               "Pashto": "Pashto (پښتو)"}[x])

        text = ""
        try:
            with st.spinner(f"Showing the recipe in {lang}..."):
                text = translate_recipe(en, lang)
        except Exception as e:
            st.error(f"The translation is not available right now. ({e})")
        css = {"English": "rtab-en", "Roman Urdu": "rtab-ru"}.get(lang, "rtab-ur")
        if text:
            safe = _html.escape(text).replace("\n", "<br>")
            st.markdown(f'<div class="{css}">{safe}</div>', unsafe_allow_html=True)

        # Voice: generated automatically for every language that has an offline voice model
        if text and lang in VOICE_LANGUAGES:
            try:
                if lang == "English":
                    speak_text, speak_lang = text, "English"
                else:   # Roman Urdu is spoken using the Urdu-script text
                    speak_text, speak_lang = translate_recipe(en, "Urdu"), "Urdu"
                audio = text_to_speech(speak_text, speak_lang)
                st.audio(audio, format="audio/wav")
            except Exception as e:
                st.error(f"The {lang} voice is not available right now. ({e})")
        elif text:
            st.info(f"Voice is not available for {lang} yet, so the recipe is shown as text only. "
                    "Voice is available in English, Roman Urdu and Urdu.")

    # Show recipe
    if st.session_state.step in ("recipe", "updated") and st.session_state.chosen:
        st.markdown('<div class="fdiv">Recipe</div>', unsafe_allow_html=True)
        render_recipe(
            title=st.session_state.chosen['name'],
            en=st.session_state.recipe_en,
            voice_key="orig",
        )

        if st.session_state.step == "recipe" and st.session_state.missing:
            st.markdown('<div class="fdiv">Missing ingredients</div>', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">Add these for the complete dish</div>', unsafe_allow_html=True)
            miss_html = '<div class="chips">' + "".join(
                f'<span class="miss-chip">＋ {m}</span>' for m in st.session_state.missing) + '</div>'
            st.markdown(miss_html, unsafe_allow_html=True)

            sel = st.multiselect("Which ones will you add?",
                                 options=st.session_state.missing,
                                 default=st.session_state.missing, key="miss_multi")
            st.session_state.sel_missing = sel

            c1, c2 = st.columns([1, 2])
            with c1:
                if st.button("Update the recipe", type="primary", **STRETCH):
                    note   = f"(Note: Additional ingredients added: {', '.join(sel)})\n\n"
                    upd_en = note + st.session_state.chosen["instructions"]
                    st.session_state.upd_recipe_en = upd_en
                    st.session_state.step = "updated"
                    st.session_state.pop("_shown_updated", None)
                    st.rerun()
            with c2:
                if st.button("Continue without adding", **STRETCH):
                    st.session_state.step = "updated"
                    st.session_state.pop("_shown_updated", None)
                    st.rerun()

    if st.session_state.step == "updated" and st.session_state.upd_recipe_en:
        st.markdown('<div class="fdiv">Updated recipe</div>', unsafe_allow_html=True)
        if st.session_state.sel_missing:
            st.success(f"Added: **{', '.join(st.session_state.sel_missing)}**")
        render_recipe(
            title=f"Updated: {st.session_state.chosen['name']}",
            en=st.session_state.upd_recipe_en,
            voice_key="updated",
        )

    if st.session_state.step not in ("input",):
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Start over", **STRETCH):
            for k, v in DEFAULTS.items():
                st.session_state[k] = v
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ADD YOUR OWN DISH
# ══════════════════════════════════════════════════════════════════════════════
from services import user_recipes as UR

def _parse_ings(text: str) -> list[str]:
    """Roman Urdu or English, comma / new-line separated -> clean English ingredient names."""
    raw = ", ".join(line for line in text.replace("\n", ",").split(","))
    clean, _ = normalize_all(raw)
    out = []
    for c in clean:
        c = c.strip().lower()
        if c and c not in out:
            out.append(c)
    return out

with tab_add:
    st.markdown('<div class="panel-title">Add your own dish</div>'
                '<div class="panel-sub">Your dish is saved on this device and appears in search results with all other dishes. '
                'Ingredients can be typed in English or Roman Urdu.</div>', unsafe_allow_html=True)

    with st.form("add_dish_form", clear_on_submit=True):
        c1, c2, c3 = st.columns([2, 1, 1], gap="medium")
        new_name = c1.text_input("Dish name", placeholder="For example: Home-style Chicken Curry")
        new_cui  = c2.selectbox("Cuisine", UR.CUISINES, format_func=lambda x: f"{FLAGS[x]} {x}")
        new_cat  = c3.selectbox("Category", UR.CATEGORIES)
        new_ings = st.text_area("Ingredients (separate with commas)", height=100,
                                placeholder="For example: chicken, onion, tomato, ginger, garlic, salt, turmeric")
        new_steps = st.text_area("Cooking steps (in English)", height=190,
                                 placeholder="1. Heat oil and fry the onions until golden.\n2. Add the meat and spices and cook for 10 minutes.\n3. ...")
        st.caption("Write the steps in English. They are translated into Roman Urdu, Urdu, Sindhi and Pashto automatically.")
        new_photo = st.file_uploader("Photo of the dish (optional)", type=["png", "jpg", "jpeg", "webp"],
                                     help="Your photo is shown on the dish card.")
        saved = st.form_submit_button("Save dish", type="primary")

    if saved:
        ings = _parse_ings(new_ings)
        err = UR.validate(new_name, ings, new_steps)
        if err:
            st.error(err)
        else:
            UR.add_recipe(new_name, new_cui, new_cat, ings, new_steps,
                          new_photo.getvalue() if new_photo else None, new_photo.name if new_photo else "")
            chips = "".join(f'<span class="chip chip-green">{_html.escape(i)}</span>' for i in ings)
            st.success(f"“{new_name.strip()}” was saved under {new_cui}. You can now find it in the Find a recipe tab.")
            st.markdown(f'<div class="panel-sub">Ingredients saved:</div><div class="chips">{chips}</div>', unsafe_allow_html=True)

    mine = UR.load_user_recipes()
    st.markdown(f'<div class="fdiv">Your dishes ({len(mine)})</div>', unsafe_allow_html=True)
    if not mine:
        st.info("You have not added a dish yet. Fill in the form above and press Save dish.")
    for rec in mine:
        with st.expander(f"{FLAGS.get(rec['cuisine'], '')} {rec['name']}  ·  {rec['cuisine']}"):
            with st.form(f"edit_{rec['id']}"):
                e1, e2, e3 = st.columns([2, 1, 1], gap="medium")
                en_name = e1.text_input("Dish name", value=rec["name"])
                en_cui  = e2.selectbox("Cuisine", UR.CUISINES, index=UR.CUISINES.index(rec["cuisine"]) if rec["cuisine"] in UR.CUISINES else 0)
                cats = UR.CATEGORIES if rec["category"] in UR.CATEGORIES else UR.CATEGORIES + [rec["category"]]
                en_cat  = e3.selectbox("Category", cats, index=cats.index(rec["category"]) if rec["category"] in cats else 0)
                en_ings = st.text_area("Ingredients", value=", ".join(rec["ingredients"]), height=90)
                en_steps = st.text_area("Steps", value=rec["instructions"], height=170)
                if rec["image"]:
                    st.image(rec["image"], width=200)
                    en_rm = st.checkbox("Remove this photo", key=f"rm_{rec['id']}")
                else:
                    en_rm = False
                en_photo = st.file_uploader("New photo (optional)", type=["png", "jpg", "jpeg", "webp"], key=f"ph_{rec['id']}")
                if st.form_submit_button("Save changes", type="primary"):
                    ings = _parse_ings(en_ings)
                    err = UR.validate(en_name, ings, en_steps)
                    if err:
                        st.error(err)
                    else:
                        UR.update_recipe(rec["id"], en_name, en_cui, en_cat, ings, en_steps,
                                         en_photo.getvalue() if en_photo else None, en_photo.name if en_photo else "", en_rm)
                        st.rerun()
            if st.button("Delete this dish", key=f"del_{rec['id']}"):
                UR.delete_recipe(rec["id"])
                st.rerun()

# ── Footer ─────────────────────────────────────────────────────────────────────
from services.recipe_engine import load_recipes as _lr
_n_recipes = f"{len(_lr()):,}"
st.markdown(f"""
<div class="sitefoot">
  <div class="foot-brand">{LOGO_DARK}<div>AI recipe assistant</div></div>
  <div>{_n_recipes} recipes &nbsp;·&nbsp; 6 cuisines &nbsp;·&nbsp; 5 languages</div>
</div>""", unsafe_allow_html=True)
