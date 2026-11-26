"""
app.py — Smart DishGenie System
Run: streamlit run app.py

Structure:
  core/       — ingredient normalization (hashmap + fuzzy)
  services/   — recipe engine, weather and time model, translator, voice, pictures
  services/ui_helpers.py — display-only helpers for the interface
(Model evaluation is done in the training notebook, not in the app.)
"""
import os
# After `python download_models.py` finished, force every library to use ONLY local files (100% offline).
if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".models_ready")):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

import html as _html
import time as _time
import streamlit as st

st.set_page_config(
    page_title="Smart DishGenie",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&family=Inter:wght@400;500;600;700&family=Noto+Nastaliq+Urdu:wght@400;700&display=swap');
:root{
  --green:#176B4D; --green-dark:#123D30; --green-light:#E9F5EE; --bg:#FFFFFF; --soft:#F4FAF6; --white:#FFFFFF;
  --text:#1F2D26; --muted:#5F6F66; --line:#DCE9E1; --gold:#E8B85C; --gold-soft:#FBF1DC;
  --berry:#B0364F; --berry-soft:#FBEAEE;
  --r-lg:20px; --r-md:14px; --r-sm:10px;
  --shadow:0 1px 2px rgba(18,61,48,.05), 0 6px 20px rgba(18,61,48,.06);
  --shadow-hover:0 2px 4px rgba(18,61,48,.06), 0 14px 30px rgba(18,61,48,.12);
}
html, body, [class*="css"], .stApp { font-family:'Inter',system-ui,sans-serif; color:var(--text); }
.stApp { background:var(--bg); }
header[data-testid="stHeader"] { display:none!important; }
.block-container { max-width:1180px; padding:0 1.5rem 3rem!important; }
.stApp, [data-testid="stAppViewContainer"] { overflow-x:clip; }
h1,h2,h3,h4 { font-family:'Plus Jakarta Sans','Inter',sans-serif!important; color:var(--text); letter-spacing:-.015em; }
[data-stale="true"], .stale-element { opacity:1!important; filter:none!important; transition:none!important; }
* { box-sizing:border-box; }
img { max-width:100%; }

/* ---------- sticky navigation bar ---------- */
.st-key-topnav { position:sticky; top:0; z-index:1000; background:#FFFFFF; margin:0 0 0 calc(50% - 50vw)!important; width:100vw!important; max-width:100vw!important;
  padding:.6rem calc(50vw - 50%) .5rem; border-bottom:1px solid var(--line); box-shadow:0 2px 14px rgba(18,61,48,.07); }
.st-key-topnav div[data-baseweb="select"] > div { background:#fff!important; border:1.5px solid var(--line)!important; border-radius:999px!important; }
.st-key-topnav [data-testid="stHorizontalBlock"] { align-items:center; }
.dg-brand svg { height:46px!important; width:auto!important; display:block; }
.st-key-topnav [data-testid="stButtonGroup"] { justify-content:center; }
.st-key-topnav [data-testid="stButtonGroup"] button { border:none!important; background:transparent!important; color:var(--muted)!important;
  font-weight:600!important; border-radius:999px!important; padding:.45rem 1.05rem!important; min-height:38px; box-shadow:none!important; }
.st-key-topnav [data-testid="stButtonGroup"] button:hover { color:var(--green)!important; }
.st-key-topnav [data-testid="stButtonGroup"] button[aria-checked="true"],
.st-key-topnav [data-testid="stButtonGroup"] button[kind$="Active"] { background:var(--green-light)!important; color:var(--green)!important; }

/* ---------- hero ---------- */
.dg-hero { display:grid; grid-template-columns:1.1fr .9fr; gap:2.6rem; align-items:center;
  margin:0 calc(50% - 50vw) 2rem; padding:3.2rem calc(50vw - 50%) 3.4rem; background:var(--green); }
.dg-kicker { display:inline-flex; align-items:center; gap:.45rem; background:rgba(255,255,255,.14); color:#FFE3A3;
  font-weight:700; font-size:.78rem; letter-spacing:.06em; text-transform:uppercase; padding:.38rem .8rem; border-radius:999px; }
.dg-hero h1 { font-size:3.05rem!important; line-height:1.08!important; font-weight:800!important; margin:1rem 0 .9rem!important; padding:0!important; color:#FFFFFF!important; }
.dg-hero p.lead { font-size:1.1rem; line-height:1.65; color:#D6ECDF; margin:0; max-width:34rem; }
.dg-stats { display:flex; gap:.8rem; margin-top:1.6rem; flex-wrap:wrap; }
.dg-stat { background:rgba(255,255,255,.10); border:1px solid rgba(255,255,255,.22); border-radius:var(--r-md); padding:.75rem 1.2rem; min-width:120px; }
.dg-stat b { display:block; font-family:'Plus Jakarta Sans',sans-serif; font-size:1.5rem; color:#FFFFFF; line-height:1.15; }
.dg-stat span { font-size:.82rem; color:#CFE6D8; }
.dg-hero-cta { display:inline-block; margin-top:1.4rem; background:#FFFFFF; color:var(--green)!important; font-weight:700; padding:.75rem 1.4rem;
  border-radius:999px; text-decoration:none!important; box-shadow:0 6px 18px rgba(0,0,0,.15); }
.dg-pagehead { margin:0 calc(50% - 50vw) 1.8rem; padding:2rem calc(50vw - 50%) 2.1rem; background:var(--soft); border-bottom:1px solid var(--line); }
.dg-pagehead .t { font-family:'Plus Jakarta Sans',sans-serif; font-size:1.9rem; font-weight:800; color:var(--green-dark); }
.dg-pagehead .s { color:var(--muted); margin-top:.3rem; font-size:1rem; }
.dg-collage { display:grid; grid-template-columns:1fr 1fr; grid-template-rows:1fr 1fr; gap:.75rem; aspect-ratio:1.08/1; }
.dg-collage > div { border-radius:var(--r-lg); overflow:hidden; box-shadow:0 10px 30px rgba(0,0,0,.18); border:3px solid rgba(255,255,255,.18); }
.dg-collage > div:first-child { grid-row:span 2; }
.dg-collage img { width:100%; height:100%; object-fit:cover; display:block; }
.dg-tile { display:flex; flex-direction:column; align-items:center; justify-content:center; gap:.3rem; height:100%; font-size:3rem; }
.dg-tile small { font-size:.8rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase; }

/* ---------- cards / panels ---------- */
.st-key-wxcard, .st-key-kitchen, .st-key-addform {
  background:var(--white); border:1px solid var(--line); border-top:4px solid var(--green); border-radius:var(--r-lg);
  padding:1.4rem 1.5rem 1.3rem; box-shadow:var(--shadow); margin-bottom:1.1rem; }
.st-key-wxcard { background:var(--soft); }
.dg-sec { font-family:'Plus Jakarta Sans',sans-serif; font-size:1.35rem; font-weight:800; color:var(--green-dark); margin:0 0 .2rem; }
.dg-sub { color:var(--muted); font-size:.93rem; margin:0 0 .9rem; line-height:1.55; }
.dg-divider { display:flex; align-items:center; gap:12px; color:var(--green); font-size:.8rem; font-weight:700; margin:1.8rem 0 1rem; text-transform:uppercase; letter-spacing:.08em; }
.dg-divider::before,.dg-divider::after { content:""; flex:1; height:1px; background:var(--line); }

/* weather card */
.dg-wx { display:flex; gap:1rem; align-items:center; }
.dg-wx-icon { width:68px; height:68px; border-radius:18px; background:#FFFFFF; border:1px solid var(--line); display:flex; align-items:center; justify-content:center; font-size:2.2rem; flex:none; }
.dg-wx-temp { font-family:'Plus Jakarta Sans',sans-serif; font-size:2.1rem; font-weight:800; color:var(--green-dark); line-height:1; }
.dg-wx-cond { font-weight:600; color:var(--text); margin-top:.25rem; }
.dg-wx-meta { color:var(--muted); font-size:.88rem; margin-top:.15rem; }
.dg-wx-msg { margin-top:.9rem; margin-bottom:.8rem; background:var(--green); color:#FFFFFF; border-radius:var(--r-sm); padding:.6rem .85rem; font-size:.9rem; line-height:1.5; }
.dg-wx-off { color:var(--muted); font-size:.92rem; padding:.4rem 0; }

/* ingredient tags */
.st-key-tagrow [data-testid="stHorizontalBlock"] { flex-wrap:wrap!important; gap:.45rem!important; }
.st-key-tagrow div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] { width:auto!important; flex:0 0 auto!important; min-width:0!important; }
[class*="st-key-tagrm_"] button { background:var(--green-light)!important; color:var(--green)!important; border:1px solid #C9E1D1!important;
  border-radius:999px!important; padding:.15rem .8rem!important; min-height:32px!important; font-weight:600!important; }
[class*="st-key-tagrm_"] button p { font-size:.85rem!important; }
[class*="st-key-tagrm_"] button:hover { background:#D6EADC!important; }
.dg-tags-label { font-size:.8rem; color:var(--muted); font-weight:600; margin:.2rem 0 .35rem; }

/* chips */
.chips { display:flex; flex-wrap:wrap; gap:8px; margin:.4rem 0 .8rem; }
.chip { display:inline-flex; align-items:center; gap:5px; padding:5px 12px; border-radius:999px; font-size:.84rem; font-weight:600; }
.chip-green { background:var(--green-light); color:var(--green); border:1px solid #C9E1D1; }
.chip-blue  { background:var(--gold-soft); color:#7A5410; border:1px solid #F0DDB2; }
.chip-gray  { background:#F1F2EE; color:#55605A; border:1px solid #E0E3DC; }
.miss-chip  { background:var(--berry-soft); color:var(--berry); border:1px solid #F1C6D0; display:inline-flex; align-items:center;
              padding:5px 12px; border-radius:999px; font-size:.84rem; font-weight:600; }

/* recipe cards */
[class*="st-key-card_"] { background:var(--white); border:1px solid var(--line); border-radius:var(--r-lg); overflow:hidden;
  box-shadow:var(--shadow); transition:transform .18s ease, box-shadow .18s ease; gap:0!important; padding-bottom:1rem; height:100%; }
[class*="st-key-card_"]:hover { transform:translateY(-3px); box-shadow:var(--shadow-hover); }
[class*="st-key-card_"] .stButton, [class*="st-key-card_"] [data-testid="stHorizontalBlock"], [class*="st-key-card_"] [data-testid="stExpander"] { padding:0 1.1rem; }
[class*="st-key-card_"] [data-testid="stExpander"] { margin:.6rem 1.1rem 0; padding:0; }
.dg-pic { position:relative; width:100%; aspect-ratio:4/3; overflow:hidden; background:#EEF1EA; }
.dg-pic img.dg-img { width:100%; height:100%; object-fit:cover; display:block; }
.dg-pic-inner { position:absolute; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:.35rem; text-align:center; padding:1rem; }
.dg-pic-icon { font-size:3.2rem; line-height:1; }
.dg-pic-cuisine { font-weight:700; font-size:.8rem; letter-spacing:.08em; text-transform:uppercase; opacity:.85; }
.dg-pic-note { font-size:.78rem; color:var(--muted); font-weight:500; }
.dg-pic-loading { background:linear-gradient(110deg,#EEF1EA 8%,#F8F9F5 18%,#EEF1EA 33%); background-size:200% 100%; animation:dgshine 1.4s linear infinite; }
@keyframes dgshine { to { background-position-x:-200%; } }
.dg-dots { display:flex; gap:6px; } .dg-dots span { width:8px; height:8px; border-radius:50%; background:var(--green); opacity:.35; animation:dgdot 1s infinite; }
.dg-dots span:nth-child(2){animation-delay:.15s} .dg-dots span:nth-child(3){animation-delay:.3s}
@keyframes dgdot { 50% { opacity:1; transform:translateY(-3px); } }
.dg-card { padding:1rem 1.1rem 1.6rem; }
[class*="st-key-card_"] [data-testid="stMarkdownContainer"] { margin-bottom:0!important; }
.dg-card-top { display:flex; justify-content:space-between; align-items:flex-start; gap:.6rem; }
.dg-card-top .dg-title { font-family:'Plus Jakarta Sans',sans-serif; font-size:1.06rem; font-weight:700; line-height:1.3; color:var(--text); }
.dg-badge { display:inline-block; padding:3px 10px; border-radius:999px; font-size:.72rem; font-weight:700; white-space:nowrap; }
.dg-meta { display:flex; gap:.9rem; flex-wrap:wrap; color:var(--muted); font-size:.8rem; margin:.45rem 0 .6rem; }
.dg-desc { font-size:.88rem; line-height:1.5; color:var(--text); margin-bottom:.4rem; }
.dg-why { display:inline-block; background:var(--green-light); color:var(--green); border-radius:8px; padding:3px 9px; font-size:.76rem; font-weight:600; margin-bottom:.5rem; }
.dg-bar { background:#E8EDE5; border-radius:6px; height:7px; overflow:hidden; margin:.3rem 0 .45rem; }
.dg-bar > div { height:7px; border-radius:6px; background:var(--green); }
.dg-ing { font-size:.8rem; color:var(--muted); line-height:1.5; }
.dg-miss { font-size:.8rem; color:var(--berry); line-height:1.5; margin-top:.15rem; }

/* recipe page */
[class*="st-key-recipehead"] { background:var(--white); border:1px solid var(--line); border-radius:var(--r-lg); box-shadow:var(--shadow); overflow:hidden; margin-bottom:.6rem; }
[class*="st-key-recipehead"] .dg-pic { aspect-ratio:4/3; }
.dg-rh { padding:1.5rem 1.6rem 1.2rem; }
.dg-rh .dg-rtitle { font-family:'Plus Jakarta Sans',sans-serif; font-size:2rem; font-weight:800; margin:.7rem 0 .3rem; color:var(--green-dark); line-height:1.15; }
.dg-facts { display:flex; gap:.6rem; flex-wrap:wrap; margin:.9rem 0 .3rem; }
.dg-fact { background:var(--soft); border:1px solid var(--line); border-radius:var(--r-sm); padding:.5rem .8rem; font-size:.8rem; color:var(--muted); }
.dg-fact b { display:block; color:var(--text); font-size:.98rem; }
.dg-steps { list-style:none; counter-reset:s; padding:0; margin:.4rem 0 0; }
.dg-steps li { counter-increment:s; position:relative; padding:.85rem 1rem .85rem 3.4rem; background:var(--white); border:1px solid var(--line);
  border-radius:var(--r-md); margin-bottom:.6rem; line-height:1.7; font-size:.99rem; }
.dg-steps li::before { content:counter(s); position:absolute; left:1rem; top:.85rem; width:1.7rem; height:1.7rem; border-radius:50%;
  background:var(--green); color:#fff; font-weight:700; font-size:.85rem; display:flex; align-items:center; justify-content:center; }
.dg-rtl .dg-steps li { padding:.85rem 3.4rem .85rem 1rem; direction:rtl; text-align:right; font-family:'Noto Nastaliq Urdu',serif; line-height:2.4; font-size:1.12rem; }
.dg-rtl .dg-steps li::before { left:auto; right:1rem; font-family:'Inter',sans-serif; }
.dg-note { background:var(--gold-soft); border-radius:var(--r-sm); padding:.7rem .9rem; margin-bottom:.7rem; color:#6B4A0E; font-size:.92rem; }
.dg-rtl .dg-note { direction:rtl; text-align:right; font-family:'Noto Nastaliq Urdu',serif; line-height:2.2; }
.dg-h4 { font-family:'Plus Jakarta Sans',sans-serif; font-weight:700; font-size:1.15rem; color:var(--green-dark); margin:1.3rem 0 .5rem; }

/* my dishes */
.dg-empty { text-align:center; background:var(--white); border:1px dashed #CFD9CC; border-radius:var(--r-lg); padding:2.4rem 1rem; }
.dg-empty .ic { font-size:2.6rem; } .dg-empty b { display:block; font-family:'Plus Jakarta Sans',sans-serif; font-size:1.15rem; margin:.5rem 0 .2rem; color:var(--green-dark); }
.dg-empty span { color:var(--muted); font-size:.92rem; }

/* ---------- widgets ---------- */
.stButton>button, .stFormSubmitButton>button { border-radius:12px!important; font-weight:600!important;
  border:1.5px solid var(--green)!important; color:var(--green)!important; background:#FFFFFF!important; min-height:44px; transition:all .15s ease; }
.stButton>button:hover, .stFormSubmitButton>button:hover { border-color:var(--green)!important; color:var(--green)!important; background:var(--green-light)!important; }
.stButton>button:disabled { border-color:var(--line)!important; color:#9AA8A0!important; }
.stButton>button[kind="primary"], .stFormSubmitButton>button[kind="primary"] { background:var(--green)!important; border-color:var(--green)!important; color:#fff!important; }
.stButton>button[kind="primary"]:hover, .stFormSubmitButton>button[kind="primary"]:hover { background:var(--green-dark)!important; border-color:var(--green-dark)!important; color:#fff!important; }
.st-key-go_btn button { min-height:52px!important; box-shadow:0 6px 16px rgba(23,107,77,.25); }
.st-key-go_btn button p { font-size:1.02rem!important; }
.st-key-now_btn button { min-height:52px!important; }
div[data-testid="stTextArea"] textarea, div[data-testid="stTextInput"] input { border-radius:12px!important; font-size:16px!important; }
div[data-testid="stTextArea"] textarea { padding:.9rem 1rem!important; }
div[data-testid="stTextArea"] > div, div[data-testid="stTextInput"] > div > div { border-radius:12px!important; }
div[data-baseweb="textarea"], div[data-baseweb="textarea"] textarea, div[data-baseweb="input"], div[data-baseweb="input"] input { background:var(--white)!important; }
div[data-baseweb="textarea"], div[data-baseweb="input"] { border:1.5px solid var(--line)!important; }
[data-baseweb="base-input"], [data-baseweb="base-input"] > textarea, [data-baseweb="base-input"] > input { background:#FFFFFF!important; }
div[data-baseweb="textarea"]:focus-within, div[data-baseweb="input"]:focus-within { border-color:var(--green)!important; box-shadow:0 0 0 3px rgba(23,107,77,.12); }
div[data-testid="stExpander"] { background:var(--soft); border-radius:var(--r-md); border:1px solid var(--line); }
div[data-testid="stExpander"] details { border:none!important; }
div[data-testid="stExpander"] summary p { font-weight:600; }
.stButton>button:focus-visible, textarea:focus-visible { outline:3px solid var(--gold)!important; outline-offset:2px; }
div[data-testid="stAlert"] { border-radius:var(--r-md)!important; }
audio { width:100%; }
div[data-testid="stRadio"] [role="radiogroup"] { flex-wrap:wrap!important; gap:8px!important; }
div[data-testid="stRadio"] label[data-testid="stRadioOption"], div[data-testid="stRadio"] label[data-baseweb="radio"] {
  background:var(--white); border:1px solid var(--line); border-radius:999px; padding:8px 16px; margin:0!important; min-height:40px; align-items:center; cursor:pointer; transition:all .15s ease; }
div[data-testid="stRadio"] label[data-baseweb="radio"]:hover { border-color:var(--green); }
div[data-testid="stRadio"] label[data-testid="stRadioOption"][data-selected="true"],
div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) { background:var(--green); border-color:var(--green); }
div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) p { color:#fff!important; font-weight:600; }
div[data-testid="stRadio"] label[data-testid="stRadioOption"] > div > div:first-child,
div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child { display:none!important; }
.stTabs [data-baseweb="tab-list"] { background:var(--white)!important; border:1px solid var(--line)!important; border-radius:12px!important; padding:4px!important; gap:4px!important; }
.stTabs button[role="tab"] { border-radius:9px!important; padding:.45rem 1rem!important; }
.stTabs button[role="tab"][aria-selected="true"] { background:var(--green-light)!important; color:var(--green)!important; }
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display:none!important; }

/* loading overlay */
.dg-loading { position:fixed; inset:0; z-index:999999; background:rgba(255,255,255,.97); display:flex; flex-direction:column;
  align-items:center; justify-content:center; gap:1rem; text-align:center; padding:1.5rem; }
.dg-loading svg.dg-logo-stack { width:150px; height:auto; }
.dg-spin { width:44px; height:44px; border-radius:50%; border:4px solid #DCE5D8; border-top-color:var(--green); animation:dgspin .9s linear infinite; }
@keyframes dgspin { to { transform:rotate(360deg); } }
.dg-loading h3 { margin:0; font-size:1.25rem; } .dg-loading p { margin:0; color:var(--muted); font-size:.95rem; max-width:26rem; }

/* footer */
.dg-foot { margin:3.5rem calc(50% - 50vw) -3rem; background:var(--green-dark); color:#BFD3C7; padding:2.6rem calc(50vw - 50%) 1.6rem; }
.dg-foot-grid { display:grid; grid-template-columns:1.5fr 1fr 1fr 1fr; gap:1.6rem; }
.dg-foot svg.dg-logo-dark { width:210px!important; height:auto!important; display:block; }
.dg-foot .fh { color:#fff; font-family:'Plus Jakarta Sans',sans-serif; font-weight:700; font-size:.9rem; margin:0 0 .6rem; }
.dg-foot p, .dg-foot li { font-size:.86rem; line-height:1.7; margin:0; color:#BFD3C7; }
.dg-foot ul { list-style:none; padding:0; margin:0; }
.dg-foot-bottom { border-top:1px solid rgba(255,255,255,.12); margin-top:1.4rem; padding-top:1rem; font-size:.8rem; display:flex; justify-content:space-between; flex-wrap:wrap; gap:.5rem; }

/* ---------- responsive ---------- */
@media (max-width:1024px){
  .dg-hero { grid-template-columns:1fr; gap:1.4rem; } .dg-hero h1 { font-size:2.5rem!important; }
  .dg-collage { aspect-ratio:2/1; }
  .dg-foot-grid { grid-template-columns:1fr 1fr; }
  .st-key-results [data-testid="stHorizontalBlock"] { flex-wrap:wrap!important; gap:1rem!important; }
  .st-key-results [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { flex:1 1 calc(50% - 1rem)!important; min-width:calc(50% - 1rem)!important; }
}
@media (max-width:640px){
  .block-container { padding:0 .8rem 2rem!important; }
  .dg-hero { padding-top:1.8rem; padding-bottom:2rem; } .dg-hero h1 { font-size:1.95rem!important; } .dg-hero p.lead { font-size:.98rem; }
  .dg-stat { min-width:0; flex:1; padding:.6rem .7rem; } .dg-stat b { font-size:1.15rem; }
  .dg-collage { aspect-ratio:1.6/1; }
  .dg-brand svg { height:38px!important; }
  .st-key-wxcard, .st-key-kitchen, .st-key-addform { padding:1.1rem 1rem; }
  div[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { flex:1 1 100%!important; min-width:100%!important; }
  .st-key-tagrow div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] { min-width:0!important; flex:0 0 auto!important; width:auto!important; }
  .st-key-results div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] { flex:1 1 100%!important; min-width:100%!important; }
  .dg-rh .dg-rtitle { font-size:1.55rem; }
  .dg-foot-grid { grid-template-columns:1fr; }
}
</style>
""", unsafe_allow_html=True)


# ── Imports from modules ───────────────────────────────────────────────────────
from normalizer import normalize_all
from context_engine import get_live_weather, build_context, suggest_for_context, reason as ctx_reason
from recipe_engine import find_best_recipes, get_missing_ingredients, ingredient_coverage_score, matched_ingredients, match_summary
from config import OUTPUT_LANGUAGES, INPUT_LANGUAGES, VOICE_LANGUAGES
from translator import translate_recipe, ingredients_to_english_checked
from tts_engine import text_to_speech
from speech_to_text import transcribe
from image_recognizer import detect_ingredients
from image_generator import request_image, image_state, drop_waiting, pictures_available
from ui_helpers import (picture_html, img_tag, cook_minutes, time_label, split_steps, hero_pictures,
                                 CUISINE_COLORS, dish_icon)
from recipe_engine import load_recipes as _lr_top
from logo import LOGO_HEADER, LOGO_STACK, LOGO_DARK
from rapidfuzz import fuzz
import importlib.util as _ilu

# LITE mode = the online version (Streamlit Community Cloud): the large translation, voice, photo and picture
# models are not installed there. Search, Module 1, Module 2 labels, weather and ranking work exactly the same.
LITE = os.environ.get("DG_LITE") == "1" or _ilu.find_spec("transformers") is None
UI_LANGS = ["English"] if LITE else OUTPUT_LANGUAGES


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
CUISINES = ["Pakistani", "Indian", "Chinese", "Italian", "Mexican", "Thai"]
LANG_LABEL = {"English": "English", "Roman Urdu": "Roman Urdu", "Urdu": "Urdu (اردو)",
              "Sindhi": "Sindhi (سنڌي)", "Pashto": "Pashto (پښتو)"}
PIC_WAIT_SECONDS = 45          # after this a waiting picture shows the fallback art (the real picture still appears when ready)

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
st.session_state.setdefault("nav", "Home")
if st.session_state.get("_goto"):                   # e.g. "View recipe" on My Dishes opens Explore Recipes
    st.session_state.nav = st.session_state.pop("_goto")
st.session_state.setdefault("ui_lang", "English")
st.session_state.setdefault("pic_t0", {})


def esc(x) -> str:
    return _html.escape(str(x))


def badge(cuisine: str) -> str:
    soft, strong = CUISINE_COLORS.get(cuisine, ("#E7F2EA", "#176B4D"))
    return f'<span class="dg-badge" style="background:{soft};color:{strong}">{esc(cuisine)}</span>'


def show_loading(title: str, note: str = ""):
    """Full-screen loading page. Returns a slot; call .empty() on it to remove the page."""
    slot = st.empty()
    slot.markdown(f"""
<div class="dg-loading">
  {LOGO_STACK}
  <div class="dg-spin"></div>
  <h3>{esc(title)}</h3>
  <p>{esc(note)}</p>
</div>""", unsafe_allow_html=True)
    return slot


# ══════════════════════════════════════════════════════════════════════════════
# DISH PICTURES — always one picture box: the real picture, a loading box, or fallback art
# ══════════════════════════════════════════════════════════════════════════════
def _img_args(rec):
    return rec["name"], rec["cuisine"], tuple(rec["ingredients"][:15]), rec.get("category", "")


def _waited(args) -> float:
    t0 = st.session_state.pic_t0.setdefault(str(args), _time.time())
    return _time.time() - t0


@st.fragment(run_every=3)
def _picture_poll(args, rec_light):
    """Checks every 3 s whether the background painter has finished this picture."""
    state, val = image_state(*args)
    if state != "waiting":
        st.rerun()                     # redraw the page once with the finished picture (or the fallback)
    if _waited(args) < PIC_WAIT_SECONDS:
        st.markdown(picture_html(rec_light, "loading"), unsafe_allow_html=True)
    else:
        st.markdown(picture_html(rec_light, "fallback", note="Picture will appear when ready"), unsafe_allow_html=True)


def dish_picture(rec, first=False):
    """Never waits for the picture model: pictures are painted in the background."""
    if rec.get("image"):                                   # the user's own photo
        try:
            with open(rec["image"], "rb") as fh:
                st.markdown(picture_html(rec, "done", fh.read()), unsafe_allow_html=True)
            return
        except Exception:
            pass
    args = _img_args(rec)
    state, val = image_state(*args)
    if state == "done":
        st.markdown(picture_html(rec, "done", val), unsafe_allow_html=True)
    elif state == "waiting" and pictures_available():
        request_image(*args, first=first)
        _picture_poll(args, {"name": rec["name"], "cuisine": rec["cuisine"], "category": rec.get("category", "")})
    else:                                                   # no picture model, or this picture failed
        st.markdown(picture_html(rec, "fallback"), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# NAVIGATION BAR (sticky)
# ══════════════════════════════════════════════════════════════════════════════
def _sync_lang():
    for k in ("lang_orig", "lang_updated"):
        st.session_state[k] = st.session_state.ui_lang

with st.container(key="topnav"):
    n1, n2, n3 = st.columns([1.25, 1.6, .85], gap="medium", vertical_alignment="center")
    with n1:
        st.markdown(f'<div class="dg-brand">{LOGO_HEADER}</div>', unsafe_allow_html=True)
    with n2:
        nav = st.segmented_control("Navigation", ["Home", "Explore Recipes", "My Dishes"], key="nav",
                                   label_visibility="collapsed")
        if not nav:                         # clicking the selected item again must not leave the page empty
            nav = "Home"
    with n3:
        st.selectbox("Recipe language", UI_LANGS, key="ui_lang", format_func=lambda x: "🌐 " + LANG_LABEL[x],
                     on_change=_sync_lang, label_visibility="collapsed", help="Language of the recipes")

# ══════════════════════════════════════════════════════════════════════════════
# HERO (Home)
# ══════════════════════════════════════════════════════════════════════════════
if nav == "Home":
    pics = hero_pictures(3)
    if len(pics) >= 3:
        collage = "".join(f"<div>{img_tag(p, 'Dish picture made by Smart DishGenie')}</div>" for p in pics[:3])
    else:
        tiles = [("🍛", "Pakistani"), ("🍜", "Thai"), ("🍝", "Italian")]
        collage = "".join(
            f'<div><div class="dg-tile" style="background:{CUISINE_COLORS[c][0]};color:{CUISINE_COLORS[c][1]}">{ic}<small>{c}</small></div></div>'
            for ic, c in tiles)
    st.markdown(f"""
<div class="dg-hero">
  <div>
    <span class="dg-kicker">✦ AI Recipe Assistant</span>
    <h1>Cook Something Amazing With What You Have.</h1>
    <p class="lead">Discover delicious recipes using ingredients already available in your kitchen.
       Let AI help you cook smarter and eat better.</p>
    <div class="dg-stats">
      <div class="dg-stat"><b>{len(_lr_top()):,}</b><span>Recipes</span></div>
      <div class="dg-stat"><b>6</b><span>Cuisines</span></div>
      <div class="dg-stat"><b>5</b><span>Languages</span></div>
    </div>
  </div>
  <div class="dg-collage">{collage}</div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# RECIPE FINDER (Home and Explore Recipes)
# ══════════════════════════════════════════════════════════════════════════════
if nav in ("Home", "Explore Recipes"):
    if nav == "Explore Recipes":
        st.markdown('<div class="dg-pagehead"><div class="t">Explore Recipes</div>'
                    '<div class="s">Tell us what you have, and we will find dishes that suit your kitchen, the weather and the time of day.</div></div>',
                    unsafe_allow_html=True)

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

    def _remove_tag(i):
        items = [x.strip() for x in st.session_state.get("ing_box", "").replace("\n", ",").split(",") if x.strip()]
        if 0 <= i < len(items):
            items.pop(i)
        st.session_state["ing_box"] = ", ".join(items)

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

    # ── Weather and time card (module 2) ──────────────────────────────────────
    from datetime import datetime as _dt
    col_k, col_w = st.columns([1.7, 1], gap="large")
    with col_w:
        with st.container(key="wxcard"):
            st.markdown('<div class="dg-sec">Weather &amp; time</div>'
                        '<div class="dg-sub">We suggest dishes that suit the weather in your city and the time of day.</div>',
                        unsafe_allow_html=True)
            wx_left = st.container()                      # the weather summary is drawn here (filled below)
            city = st.text_input("City", value="Karachi", key="ctx_city")
            use_ctx = st.toggle("Match dishes to the weather and time", value=True, key="use_ctx")
            manual = st.checkbox("Set the weather and time manually", key="ctx_manual")

            ctx = None
            live_failed = False
            if use_ctx:
                live = None if manual else get_live_weather(city)
                if live:
                    ctx = build_context(live["temp"], live["sky"], live["hour"], live["minute"], live["city"], "live")
                else:
                    now = _dt.now()
                    live_failed = not manual
                    m1 = m2 = m3 = st
                    t_in  = m1.slider("Temperature (°C)", -5, 48, 30, key="ctx_temp")
                    sky   = m2.radio("Sky", ["clear", "cloudy", "rain"], horizontal=True, key="ctx_sky",
                                     format_func=lambda x: {"clear": "☀️ Clear", "cloudy": "☁️ Cloudy", "rain": "🌧️ Rain"}[x])
                    hr    = m3.slider("Hour of the day", 0, 23, now.hour, key="ctx_hour")
                    ctx = build_context(t_in, sky, hr, now.minute if hr == now.hour else 0, "", "manual")
            st.session_state.ctx = ctx

            with wx_left:
                if ctx:
                    icon = "🌧️" if ctx["rainy"] else "☀️" if ctx["temp_class"] == "hot" else "❄️" if ctx["temp_class"] == "cold" else "🌤️"
                    sky_txt = {"clear": "Clear sky", "cloudy": "Cloudy", "rain": "Rain"}.get(ctx["sky"], str(ctx["sky"]).title())
                    place = ctx["city"] or ("Manual setting" if ctx["source"] == "manual" else city)
                    clock = ctx["summary"].split("·")[-1].strip()
                    st.markdown(f"""
    <div class="dg-wx">
      <div class="dg-wx-icon">{icon}</div>
      <div>
        <div class="dg-wx-temp">{ctx['temp']:.0f}°C</div>
        <div class="dg-wx-cond">{esc(sky_txt)} · {esc(ctx['weather_text'].capitalize())}</div>
        <div class="dg-wx-meta">📍 {esc(place)} &nbsp;·&nbsp; 🕒 {esc(clock)}</div>
      </div>
    </div>
    <div class="dg-wx-msg">A good moment for dishes suited to <b>{esc(ctx['weather_text'])}</b> and <b>{esc(ctx['meal_text'])}</b>.</div>""",
                                unsafe_allow_html=True)
                    if live_failed:
                        st.caption("We could not get the live weather (no internet connection, or the city was not found). "
                                   "Set it below; the time is taken from your device.")
                else:
                    st.markdown('<div class="dg-wx-off">Weather and time matching is off. Turn it on to get dishes that suit the moment.</div>',
                                unsafe_allow_html=True)

    with col_k:
        # ── Ingredient panel ──────────────────────────────────────────────────────
        with st.container(key="kitchen"):
            st.markdown('<div class="dg-sec">What\'s in Your Kitchen?</div>'
                        '<div class="dg-sub">Type the ingredients you have, separated by commas. English and Roman Urdu both work, '
                        'and small spelling mistakes are fine. You can also add them by voice or photo.</div>',
                        unsafe_allow_html=True)

            if LITE:
                st.caption("Voice and photo input, translation and dish pictures are available in the full desktop version.")
            with (st.container() if LITE else st.expander("🎤  Add ingredients by voice or photo", expanded=False)):
              if not LITE:
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
                              st.success(f"Added to your ingredients: **{', '.join(added)}**. You can remove any wrong item below.")
                          else:
                              st.info("No new ingredient was found, or it is already in your list.")

            user_input = st.text_area(
                label="Your ingredients", label_visibility="collapsed",
                placeholder="Example: Rice, Chicken, Onion, Tomato...",
                height=120, key="ing_box",
            )

            # removable ingredient tags (the text box stays the single source of truth)
            _items = [x.strip() for x in (user_input or "").replace("\n", ",").split(",") if x.strip()]
            if _items:
                st.markdown('<div class="dg-tags-label">Your ingredients · click a tag to remove it</div>', unsafe_allow_html=True)
                with st.container(key="tagrow"):
                    shown = _items[:15]
                    tcols = st.columns(len(shown))
                    for i, it in enumerate(shown):
                        with tcols[i]:
                            st.button(f"{it}  ✕", key=f"tagrm_{i}", on_click=_remove_tag, args=(i,), help=f"Remove {it}")

            st.markdown('<div class="dg-tags-label" style="margin-top:.7rem">Choose a cuisine</div>', unsafe_allow_html=True)
            cuisine = st.radio(label="Cuisine", label_visibility="collapsed", horizontal=True,
                               options=CUISINES, key="cui_radio")
            st.session_state.cuisine = cuisine

            st.markdown('<div style="height:.4rem"></div>', unsafe_allow_html=True)
            b1, b2 = st.columns([1.4, 1], gap="medium")
            with b1:
                go_btn = st.button("Find My Recipes", type="primary", key="go_btn", **STRETCH)
            with b2:
                now_btn = st.button("What Should I Cook Now?", key="now_btn", **STRETCH, disabled=not use_ctx,
                                    help="No ingredients needed. We suggest dishes for the current weather and time." if use_ctx
                                    else "Turn on 'Match dishes to the weather and time' first.")

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

    # How the ingredients were understood (Module 1)
    if st.session_state.step != "input" and st.session_state.norm_details:
        st.markdown('<div class="dg-divider">How we understood your ingredients</div>', unsafe_allow_html=True)
        chips = '<div class="chips">'
        for d in st.session_state.norm_details:
            m     = d["method"]
            cls   = "chip-green" if m == "hashmap" else "chip-blue" if m == "fuzzy" else "chip-gray"
            orig  = esc(d["original"])
            match = esc(d["matched"])
            note  = f' <span style="opacity:.6;font-size:.76rem">(you typed "{orig}")</span>' if orig != match else ""
            chips += f'<span class="chip {cls}">{match}{note}</span>'
        chips += '</div>'
        st.markdown(chips, unsafe_allow_html=True)

    # ── Recipe cards ──────────────────────────────────────────────────────────
    if st.session_state.step == "pick" and st.session_state.matched:
        st.markdown('<div class="dg-divider">Recommended for you</div>', unsafe_allow_html=True)
        _ctx_mode = not st.session_state.clean_ings
        _cx = st.session_state.get("ctx")
        cname = st.session_state.cuisine
        if _ctx_mode and _cx:
            st.markdown(f'<div class="dg-sec">{esc(cname)} dishes for {esc(_cx["weather_text"])} and {esc(_cx["meal_text"])}</div>'
                        '<div class="dg-sub">Picked for the weather and time right now. Check again later for new ideas.</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="dg-sec">{len(st.session_state.matched)} {esc(cname)} dishes you can make</div>'
                        '<div class="dg-sub">Ranked by how well they use your ingredients and suit the moment.</div>',
                        unsafe_allow_html=True)

        results = st.session_state.matched
        with st.container(key="results"):
            for row in range(0, len(results), 3):
                cols = st.columns(3, gap="medium")
                for j, (score, rec) in enumerate(results[row:row + 3]):
                    i = row + j
                    ms   = match_summary(st.session_state.clean_ings, rec)
                    pct  = int(100 * ms["have"] / ms["total"]) if ms["total"] else 0
                    why  = ctx_reason(rec, _cx) if _cx else ""
                    mins = time_label(cook_minutes(rec.get("instructions", "")))
                    need = [x for x in rec["ingredients"] if x not in ("salt", "water", "oil", "pepper", "sugar")][:5]
                    meta = f'<span>{dish_icon(rec)} {esc(rec["category"].title())}</span>'
                    if mins:
                        meta += f'<span>⏱ {esc(mins)}</span>'
                    meta += f'<span>🧺 {len(rec["ingredients"])} ingredients</span>'
                    if rec.get("user"):
                        meta += '<span style="color:#7A5410;font-weight:700">★ Your dish</span>'
                    if _ctx_mode:
                        body = (f'<div class="dg-desc">{esc(why or "A good choice for right now.")}</div>'
                                f'<div class="dg-ing">Main ingredients: {esc(", ".join(need))}</div>')
                    else:
                        miss_txt = ("Missing: " + ", ".join(esc(m) for m in ms["missing"][:4])) if ms["missing"] else "You have everything you need"
                        body = ((f'<div class="dg-why">{esc(why)}</div>' if why else "")
                                + f'<div class="dg-desc">You have <b>{ms["have"]} of {ms["total"]}</b> ingredients</div>'
                                f'<div class="dg-bar"><div style="width:{max(pct, 6)}%"></div></div>'
                                f'<div class="dg-ing">Main ingredients: {esc(", ".join(need))}</div>'
                                f'<div class="dg-miss">{miss_txt}</div>')
                    with cols[j]:
                        with st.container(key=f"card_{i}"):
                            dish_picture(rec)
                            st.markdown(f"""
<div class="dg-card">
  <div class="dg-card-top"><div class="dg-title">{esc(rec['name'])}</div>{badge(rec['cuisine'])}</div>
  <div class="dg-meta">{meta}</div>
  {body}
</div>""", unsafe_allow_html=True)
                            if st.button("View Recipe →", key=f"pick_{i}", type="primary" if i == 0 else "secondary", **STRETCH):
                                st.session_state.chosen  = rec
                                st.session_state.missing = get_missing_ingredients(
                                    st.session_state.clean_ings, rec["ingredients"]) if st.session_state.clean_ings else []
                                st.session_state.recipe_en = rec["instructions"]
                                st.session_state.step = "recipe"
                                st.rerun()

    # ── Recipe detail ─────────────────────────────────────────────────────────
    def render_recipe(title, en, voice_key):
        _load = show_loading(f"Preparing {title}", "Getting the ingredients, steps and voice ready...")
        try:
            _render_recipe(title, en, voice_key)
        finally:
            _load.empty()

    def _render_recipe(title, en, voice_key):
        rec  = st.session_state.chosen
        cui  = rec.get("cuisine") or st.session_state.cuisine
        user_ings = st.session_state.clean_ings
        used = matched_ingredients(user_ings, rec["ingredients"])
        missing_now = get_missing_ingredients(user_ings, rec["ingredients"])
        mins = time_label(cook_minutes(en))

        with st.container(key=f"recipehead_{voice_key}"):
            c_pic, c_info = st.columns([1, 1.15], gap="small")
            with c_info:
                facts = f'<div class="dg-fact"><b>{len(rec["ingredients"])}</b>ingredients</div>'
                if mins:
                    facts += f'<div class="dg-fact"><b>{esc(mins)}</b>cooking time</div>'
                if user_ings:
                    facts += f'<div class="dg-fact"><b>{len(used)} of {len(used) + len(missing_now)}</b>you have</div>'
                facts += f'<div class="dg-fact"><b>{esc(rec["category"].title())}</b>category</div>'
                _cx = st.session_state.get("ctx")
                why = ctx_reason(rec, _cx) if _cx else ""
                main = [x for x in rec["ingredients"] if x not in ("salt", "water", "oil", "pepper", "sugar")][:6]
                st.markdown(f"""
<div class="dg-rh">
  {badge(cui)}
  <div class="dg-rtitle">{esc(title)}</div>
  {f'<div class="dg-why" style="margin-top:.3rem">{esc(why)}</div>' if why else ''}
  <div class="dg-facts">{facts}</div>
  <div class="dg-ing" style="margin-top:.8rem">Main ingredients: {esc(", ".join(main))}</div>
  <div class="dg-ing" style="margin-top:.3rem">Choose a language below to read or listen to the recipe.</div>
</div>""", unsafe_allow_html=True)
            with c_pic:
                dish_picture(rec, first=True)

        # ingredients
        st.markdown('<div class="dg-h4">Ingredients</div>', unsafe_allow_html=True)
        if user_ings:
            st.caption("Green: you have it · Red: you still need it")
            chips = "".join(f'<span class="chip chip-green">✓ {esc(u)}</span>' for u in used)
            chips += "".join(f'<span class="miss-chip">+ {esc(m)}</span>' for m in missing_now)
        else:
            chips = "".join(f'<span class="chip chip-gray">{esc(i)}</span>' for i in rec["ingredients"])
        st.markdown(f'<div class="chips">{chips}</div>', unsafe_allow_html=True)

        # language, voice and steps
        st.markdown('<div class="dg-h4">Method</div>', unsafe_allow_html=True)
        lkey = f"lang_{voice_key}"
        if lkey not in st.session_state:
            st.session_state[lkey] = st.session_state.get("ui_lang", "English")
        if st.session_state.get(lkey) not in UI_LANGS:
            st.session_state[lkey] = UI_LANGS[0]
        lang = st.radio("Language", UI_LANGS, horizontal=True, key=lkey, format_func=lambda x: LANG_LABEL[x])

        text = ""
        try:
            text = translate_recipe(en, lang)
        except Exception as e:
            st.error(f"The translation is not available right now. ({e})")

        if LITE:
            pass                                         # no voice models in the online version
        elif text and lang in VOICE_LANGUAGES:
            try:
                if lang == "English":
                    speak_text, speak_lang = text, "English"
                else:   # Roman Urdu is spoken using the Urdu-script text
                    speak_text, speak_lang = translate_recipe(en, "Urdu"), "Urdu"
                audio = text_to_speech(speak_text, speak_lang)
                st.markdown('<div class="dg-tags-label">🔊 Listen to the recipe</div>', unsafe_allow_html=True)
                st.audio(audio, format="audio/wav")
            except Exception as e:
                st.error(f"The {lang} voice is not available right now. ({e})")
        elif text:
            st.caption(f"Voice is not available for {lang} yet. Voice is available in English, Roman Urdu and Urdu.")

        if text:
            note, steps = split_steps(text)
            rtl = lang in ("Urdu", "Sindhi", "Pashto")
            items = "".join(f"<li>{esc(s)}</li>" for s in steps)
            st.markdown(f'<div class="{"dg-rtl" if rtl else ""}">'
                        + (f'<div class="dg-note">{esc(note)}</div>' if note else "")
                        + f'<ol class="dg-steps">{items}</ol></div>', unsafe_allow_html=True)

    if st.session_state.step in ("recipe", "updated") and st.session_state.chosen:
        st.markdown('<div class="dg-divider">Recipe</div>', unsafe_allow_html=True)
        if st.session_state.matched and st.button("← Back to suggestions", key="back_btn"):
            st.session_state.step = "pick"
            st.rerun()
        render_recipe(
            title=st.session_state.chosen['name'],
            en=st.session_state.recipe_en,
            voice_key="orig",
        )

        if st.session_state.step == "recipe" and st.session_state.missing:
            st.markdown('<div class="dg-divider">Missing ingredients</div>', unsafe_allow_html=True)
            st.markdown('<div class="dg-sec" style="font-size:1.1rem">Add these for the complete dish</div>', unsafe_allow_html=True)
            miss_html = '<div class="chips">' + "".join(
                f'<span class="miss-chip">＋ {esc(m)}</span>' for m in st.session_state.missing) + '</div>'
            st.markdown(miss_html, unsafe_allow_html=True)

            sel = st.multiselect("Which ones will you add?",
                                 options=st.session_state.missing,
                                 default=st.session_state.missing, key="miss_multi")
            st.session_state.sel_missing = sel

            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button("Update the recipe", type="primary", **STRETCH):
                    note   = f"(Note: Additional ingredients added: {', '.join(sel)})\n\n"
                    upd_en = note + st.session_state.chosen["instructions"]
                    st.session_state.upd_recipe_en = upd_en
                    st.session_state.step = "updated"
                    st.rerun()
            with c2:
                if st.button("Continue without adding", **STRETCH):
                    st.session_state.step = "updated"
                    st.rerun()

    if st.session_state.step == "updated" and st.session_state.upd_recipe_en:
        st.markdown('<div class="dg-divider">Updated recipe</div>', unsafe_allow_html=True)
        if st.session_state.sel_missing:
            st.success(f"Added: **{', '.join(st.session_state.sel_missing)}**")
        render_recipe(
            title=f"Updated: {st.session_state.chosen['name']}",
            en=st.session_state.upd_recipe_en,
            voice_key="updated",
        )

    if st.session_state.step not in ("input",):
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Start over", key="start_over", **STRETCH):
            for k, v in DEFAULTS.items():
                st.session_state[k] = v
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# MY DISHES
# ══════════════════════════════════════════════════════════════════════════════
import user_recipes as UR

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

if nav == "My Dishes":
    st.markdown('<div class="dg-pagehead"><div class="t">My Dishes</div>'
                '<div class="s">Add your own family recipes. They are saved on this device and included in every search.</div></div>',
                unsafe_allow_html=True)
    with st.container(key="addform"):
        st.markdown('<div class="dg-sec">Add your own dish</div>'
                    '<div class="dg-sub">Your dish is saved on this device and appears in search results with all other dishes. '
                    'Ingredients can be typed in English or Roman Urdu.</div>', unsafe_allow_html=True)

        with st.form("add_dish_form", clear_on_submit=True, border=False):
            c1, c2, c3 = st.columns([2, 1, 1], gap="medium")
            new_name = c1.text_input("Dish name", placeholder="For example: Home-style Chicken Curry")
            new_cui  = c2.selectbox("Cuisine", UR.CUISINES)
            new_cat  = c3.selectbox("Category", UR.CATEGORIES)
            new_ings = st.text_area("Ingredients (separate with commas)", height=100,
                                    placeholder="For example: chicken, onion, tomato, ginger, garlic, salt, turmeric")
            new_steps = st.text_area("Cooking steps (in English)", height=170,
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
                chips = "".join(f'<span class="chip chip-green">{esc(i)}</span>' for i in ings)
                st.success(f"“{new_name.strip()}” was saved under {new_cui}. You can now find it in Explore Recipes.")
                st.markdown(f'<div class="dg-sub">Ingredients saved:</div><div class="chips">{chips}</div>', unsafe_allow_html=True)

    mine = UR.load_user_recipes()
    st.markdown(f'<div class="dg-divider">Your dishes ({len(mine)})</div>', unsafe_allow_html=True)
    if not mine:
        st.markdown('<div class="dg-empty"><div class="ic">🍲</div><b>No dishes yet</b>'
                    '<span>Add your family recipes with the form above. They will appear here and in your searches.</span></div>',
                    unsafe_allow_html=True)
    for row in range(0, len(mine), 3):
        cols = st.columns(3, gap="medium")
        for j, rec in enumerate(mine[row:row + 3]):
            with cols[j]:
                with st.container(key=f"card_mine_{rec['id']}"):
                    dish_picture(rec)
                    st.markdown(f"""
<div class="dg-card">
  <div class="dg-card-top"><div class="dg-title">{esc(rec['name'])}</div>{badge(rec['cuisine'])}</div>
  <div class="dg-meta"><span>{dish_icon(rec)} {esc(rec['category'].title())}</span><span>🧺 {len(rec['ingredients'])} ingredients</span></div>
  <div class="dg-ing">{esc(", ".join(rec['ingredients'][:6]))}</div>
</div>""", unsafe_allow_html=True)
                    v1, v2 = st.columns(2, gap="small")
                    with v1:
                        if st.button("View recipe", key=f"view_{rec['id']}", type="primary", **STRETCH):
                            for k, v in DEFAULTS.items():
                                st.session_state[k] = v
                            st.session_state.cuisine = rec["cuisine"]
                            st.session_state.chosen = rec
                            st.session_state.recipe_en = rec["instructions"]
                            st.session_state.step = "recipe"
                            st.session_state.auto_ctx_done = True
                            st.session_state["_goto"] = "Explore Recipes"
                            st.rerun()
                    with v2:
                        if st.button("Remove", key=f"del_{rec['id']}", **STRETCH):
                            UR.delete_recipe(rec["id"])
                            st.rerun()
                    with st.expander("Edit dish"):
                        with st.form(f"edit_{rec['id']}", border=False):
                            en_name = st.text_input("Dish name", value=rec["name"])
                            en_cui  = st.selectbox("Cuisine", UR.CUISINES, index=UR.CUISINES.index(rec["cuisine"]) if rec["cuisine"] in UR.CUISINES else 0)
                            cats = UR.CATEGORIES if rec["category"] in UR.CATEGORIES else UR.CATEGORIES + [rec["category"]]
                            en_cat  = st.selectbox("Category", cats, index=cats.index(rec["category"]) if rec["category"] in cats else 0)
                            en_ings = st.text_area("Ingredients", value=", ".join(rec["ingredients"]), height=90)
                            en_steps = st.text_area("Steps", value=rec["instructions"], height=170)
                            if rec["image"]:
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

# ── Footer ─────────────────────────────────────────────────────────────────────
_n_recipes = f"{len(_lr_top()):,}"
st.markdown(f"""
<div class="dg-foot">
  <div class="dg-foot-grid">
    <div><div>{LOGO_DARK}</div><p style="margin-top:.6rem">An AI recipe assistant that turns the ingredients in your kitchen into dishes
      that suit the weather and the time of day.</p></div>
    <div><div class="fh">Features</div><ul><li>Ingredient search</li><li>Weather &amp; time matching</li><li>Voice and photo input</li><li>Your own dishes</li></ul></div>
    <div><div class="fh">Cuisines</div><ul><li>Pakistani · Indian</li><li>Chinese · Thai</li><li>Italian · Mexican</li></ul></div>
    <div><div class="fh">Languages</div><ul><li>English · Roman Urdu</li><li>Urdu · Sindhi</li><li>Pashto</li></ul></div>
  </div>
  <div class="dg-foot-bottom"><span>{_n_recipes} recipes · 6 cuisines · 5 languages</span><span>Smart DishGenie System · Final Year Project</span></div>
</div>""", unsafe_allow_html=True)
