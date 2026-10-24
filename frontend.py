"""
ScamShield v2 — Threat Intelligence Dashboard (Streamlit)
Same look and layout as v1, wired to the v2 multi-task multimodal API (app.py).
Left : live scan feed, system telemetry, multimodal inspection suite (text / call audio / URL / batch)
Right: holographic risk gauge + uncertainty, fraud-taxonomy breakdown,
       Integrated-Gradients attribution matrix, inference console
Run  : uvicorn app:app --port 8000      then      streamlit run frontend.py
"""
import html
import io
import os
import tempfile
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st

API = os.getenv("SCAMSHIELD_API", "http://localhost:8000")
MC_SAMPLES = int(os.getenv("SCAMSHIELD_MC_SAMPLES", "30"))
START_HINT = "Start it with `uvicorn app:app --port 8000`."

st.set_page_config(page_title="ScamShield", page_icon="🛡️", layout="wide", initial_sidebar_state="collapsed")

# --------------------------------------------------------------------------- #
# Design system — every CSS override lives here
# --------------------------------------------------------------------------- #
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

@property --p{syntax:'<number>';inherits:false;initial-value:0}
:root{
  --bg:#030806; --glass:rgba(7,22,16,.78); --glass2:rgba(3,12,8,.9);
  --line:rgba(34,229,138,.15); --line2:rgba(34,229,138,.32); --hair:rgba(255,255,255,.07);
  --txt:#E6F4EC; --mut:#7FA391; --dim:#4F7563;
  --cyan:#22E58A; --red:#EF4444; --green:#10B981; --amber:#F59E0B;
}

/* ===== base ===== */
html,body,[class*="css"],.stApp,.stMarkdown,button,input,textarea{font-family:'Plus Jakarta Sans',sans-serif!important;}
.stApp{color:var(--txt);background:
  radial-gradient(1000px 560px at 88% -8%,rgba(34,229,138,.13),transparent 60%),
  radial-gradient(800px 520px at -6% 18%,rgba(5,150,105,.09),transparent 60%),
  radial-gradient(700px 400px at 50% 110%,rgba(16,185,129,.09),transparent 60%),var(--bg);}
.stApp::before{content:"";position:fixed;inset:0;pointer-events:none;z-index:0;
  background-image:linear-gradient(rgba(34,229,138,.03) 1px,transparent 1px),linear-gradient(90deg,rgba(34,229,138,.03) 1px,transparent 1px);
  background-size:42px 42px;-webkit-mask-image:radial-gradient(ellipse at 50% 0%,#000 25%,transparent 78%);mask-image:radial-gradient(ellipse at 50% 0%,#000 25%,transparent 78%);}
#MainMenu,header,footer,[data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="stStatusWidget"],[data-testid="stHeader"]{visibility:hidden;height:0;}
.block-container{padding:.7rem 1.4rem 1.2rem 1.4rem!important;max-width:1560px;position:relative;z-index:1;}
[data-testid="stVerticalBlock"]{gap:.8rem;}
::-webkit-scrollbar{width:7px;height:7px}::-webkit-scrollbar-thumb{background:#14301f;border-radius:8px}

/* ===== top bar ===== */
.topbar{display:flex;align-items:center;justify-content:space-between;gap:14px;flex-wrap:wrap;padding:11px 18px;border-radius:14px;
  background:var(--glass);border:1px solid var(--line);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);
  box-shadow:0 8px 34px rgba(0,0,0,.5),0 0 0 1px rgba(34,229,138,.04),inset 0 1px 0 rgba(255,255,255,.05);}
.topbar{position:relative;overflow:hidden;padding:13px 20px;}
.topbar::after{content:"";position:absolute;left:8%;right:8%;bottom:0;height:1px;background:linear-gradient(90deg,transparent,rgba(34,229,138,.75),transparent);}
.brand{display:flex;align-items:center;gap:15px;}
.logo{width:46px;height:46px;border-radius:13px;display:grid;place-items:center;
  background:linear-gradient(145deg,rgba(34,229,138,.24),rgba(5,150,105,.24));border:1px solid rgba(34,229,138,.5);
  box-shadow:0 0 28px rgba(34,229,138,.35),inset 0 1px 0 rgba(255,255,255,.2);}
.logo svg{width:25px;height:25px;filter:drop-shadow(0 0 6px rgba(34,229,138,.85));}
.brand h1{font-size:26px;font-weight:800;letter-spacing:-.8px;margin:0;line-height:1;color:#F1F5F9;}
.brand h1 span{background:linear-gradient(90deg,#22E58A,#6EE7B7);-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;}
.brand p{margin:7px 0 0 0;font-size:10px;font-weight:600;letter-spacing:2.4px;text-transform:uppercase;color:var(--mut);}
.status{display:flex;align-items:center;gap:8px;flex-wrap:wrap;font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:.5px;}
.chip{display:inline-flex;align-items:center;gap:7px;padding:6px 11px;border-radius:8px;background:rgba(34,229,138,.05);border:1px solid var(--line);color:var(--mut);}
.chip b{color:var(--txt);font-weight:500;}
.led{width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 10px var(--green);animation:pulse 1.8s infinite;}
.led.off{background:var(--red);box-shadow:0 0 10px var(--red);animation:none;}
.led.warn{background:var(--amber);box-shadow:0 0 10px var(--amber);}
.led.cy{background:var(--cyan);box-shadow:0 0 10px var(--cyan);}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}

/* ===== glass cards (HTML cards + native bordered containers) ===== */
.card,[data-testid="stVerticalBlockBorderWrapper"]{background:var(--glass)!important;border:1px solid var(--line)!important;border-radius:16px!important;
  backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);
  box-shadow:0 10px 36px rgba(0,0,0,.55),0 0 34px rgba(34,229,138,.07),inset 0 1px 0 rgba(255,255,255,.05);}
.card{padding:16px 18px;}
[data-testid="stVerticalBlockBorderWrapper"]{padding:6px 12px 12px 12px;}
.sec{display:flex;align-items:center;gap:9px;margin:0 0 13px 0;font-size:10.5px;font-weight:700;letter-spacing:1.7px;text-transform:uppercase;color:var(--mut);}
.sec::before{content:"";width:3px;height:13px;border-radius:2px;background:linear-gradient(var(--cyan),#059669);box-shadow:0 0 9px var(--cyan);}
.sec .r{margin-left:auto;font-family:'JetBrains Mono',monospace;letter-spacing:.4px;text-transform:none;font-weight:500;color:var(--dim);font-size:10.5px;}

/* ===== threat feed ===== */
.feed{position:relative;height:178px;overflow:hidden;
  -webkit-mask-image:linear-gradient(transparent,#000 12%,#000 88%,transparent);mask-image:linear-gradient(transparent,#000 12%,#000 88%,transparent);}
.ftrack{animation:fscroll var(--d,30s) linear infinite;}
.feed:hover .ftrack{animation-play-state:paused;}
@keyframes fscroll{to{transform:translateY(-50%)}}
.fi{display:grid;grid-template-columns:62px 74px 1fr auto;align-items:center;gap:10px;padding:7px 4px;border-bottom:1px solid var(--hair);font-size:12px;}
.fi .t{font-family:'JetBrains Mono',monospace;color:var(--dim);font-size:11px;}
.fi .v{font-weight:600;color:var(--txt);} .fi .v small{display:block;font-weight:400;color:var(--mut);font-size:11px;margin-top:1px;}
.sev{font-family:'JetBrains Mono',monospace;font-size:9.5px;font-weight:700;letter-spacing:.8px;padding:3px 7px;border-radius:5px;text-align:center;}
.sv-CRITICAL{color:#FF6B6B;background:rgba(239,68,68,.12);border:1px solid rgba(239,68,68,.45);box-shadow:0 0 12px rgba(239,68,68,.25);}
.sv-ELEVATED{color:#FBBF24;background:rgba(245,158,11,.10);border:1px solid rgba(245,158,11,.45);}
.sv-LOW{color:var(--cyan);background:rgba(34,229,138,.09);border:1px solid rgba(34,229,138,.4);}
.src{font-family:'JetBrains Mono',monospace;font-size:9.5px;letter-spacing:1px;color:var(--dim);}
.src.live{color:var(--green);}

/* ===== telemetry tiles ===== */
.tgrid{display:grid;grid-template-columns:1fr 1fr;gap:.8rem;}
.tile{padding:13px 15px;border-radius:14px;background:var(--glass);border:1px solid var(--line);backdrop-filter:blur(14px);
  box-shadow:0 8px 28px rgba(0,0,0,.45),inset 0 1px 0 rgba(255,255,255,.05);position:relative;overflow:hidden;}
.tile::after{content:"";position:absolute;right:-30px;top:-30px;width:90px;height:90px;border-radius:50%;background:radial-gradient(var(--tc,rgba(34,229,138,.18)),transparent 70%);}
.tl{font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:var(--mut);font-weight:700;}
.tv{font-family:'JetBrains Mono',monospace;font-size:23px;font-weight:700;margin:7px 0 3px 0;color:var(--cyan);text-shadow:0 0 18px rgba(34,229,138,.4);}
.tv small{font-size:12px;color:var(--mut);font-weight:500;margin-left:3px;text-shadow:none;}
.tv.g{color:var(--green);text-shadow:0 0 18px rgba(16,185,129,.4);} .tv.a{color:var(--amber);text-shadow:0 0 18px rgba(245,158,11,.4);} .tv.r{color:var(--red);text-shadow:0 0 18px rgba(239,68,68,.4);}
.ts{font-family:'JetBrains Mono',monospace;font-size:10.5px;color:var(--dim);}
.meter{display:grid;grid-template-columns:34px 1fr 40px;align-items:center;gap:8px;margin-top:7px;font-family:'JetBrains Mono',monospace;font-size:10.5px;color:var(--mut);}
.meter b{text-align:right;color:var(--txt);font-weight:500;} .meter em{grid-column:2/4;font-style:normal;color:var(--dim);}
svg.spark{display:block;margin-top:4px;}

/* ===== inputs ===== */
.stTextArea textarea,.stTextInput input{background:rgba(2,10,6,.9)!important;border:1px solid var(--line2)!important;border-radius:12px!important;color:var(--txt)!important;
  font-size:14px!important;transition:all .2s;}
.stTextArea textarea{line-height:1.6!important;}
.stTextArea textarea::placeholder,.stTextInput input::placeholder{color:var(--dim)!important;}
.stTextArea textarea:focus,.stTextInput input:focus{border-color:var(--cyan)!important;box-shadow:0 0 0 1px var(--cyan),0 0 24px rgba(34,229,138,.32)!important;}
[data-testid="stFileUploader"] section{background:rgba(2,10,6,.9);border:1px dashed rgba(34,229,138,.35);border-radius:12px;transition:all .2s;}
[data-testid="stFileUploader"] section:hover{border-color:var(--cyan);box-shadow:0 0 20px rgba(34,229,138,.18);}
[data-testid="stFileUploader"] small,[data-testid="stFileUploader"] span{color:var(--mut)!important;}
.stTabs [data-baseweb="tab-list"]{gap:4px;border-bottom:1px solid var(--line);}
.stTabs [data-baseweb="tab"]{background:transparent;color:var(--mut);font-weight:600;font-size:13px;padding:9px 15px;border-radius:8px 8px 0 0;}
.stTabs [data-baseweb="tab"]:hover{color:var(--txt);}
.stTabs [aria-selected="true"]{color:var(--cyan)!important;}
.stTabs [data-baseweb="tab-highlight"]{background:var(--cyan)!important;box-shadow:0 0 10px var(--cyan);}
.stTabs [data-baseweb="tab-border"]{background:transparent!important;}
[data-testid="stCaptionContainer"]{color:var(--dim)!important;font-size:10.5px!important;letter-spacing:1px;text-transform:uppercase;}
[data-testid="stAlert"]{border-radius:12px;border:1px solid var(--line);background:rgba(239,68,68,.08);}
[data-testid="stAudio"] audio,audio{width:100%;filter:invert(.92) hue-rotate(180deg) saturate(.7);}

/* segmented filter (st.radio) */
div[role="radiogroup"]{gap:.4rem;flex-wrap:wrap;}
label[data-baseweb="radio"]{background:rgba(255,255,255,.03);border:1px solid var(--hair);border-radius:8px;padding:3px 12px;margin:0;transition:all .15s;}
label[data-baseweb="radio"] > div:first-child{display:none;}
label[data-baseweb="radio"] p{font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:.8px;color:var(--mut);}
label[data-baseweb="radio"]:has(input:checked){border-color:var(--cyan);background:rgba(34,229,138,.10);box-shadow:0 0 14px rgba(34,229,138,.2);}
label[data-baseweb="radio"]:has(input:checked) p{color:var(--cyan);}

/* ===== buttons ===== */
.stButton>button{width:100%;border-radius:10px;font-weight:600;font-size:12.5px;padding:.5rem .8rem;color:var(--txt);
  background:rgba(34,229,138,.04);border:1px solid var(--line2);transition:all .18s ease;}
.stButton>button:hover{border-color:var(--cyan);color:var(--cyan);background:rgba(34,229,138,.10);transform:translateY(-1px);box-shadow:0 6px 20px rgba(34,229,138,.18);}
.stButton>button:active{transform:translateY(0);}
.stButton>button[kind="primary"]{color:#02130a;font-weight:800;letter-spacing:.6px;border:1px solid rgba(255,255,255,.3);padding:.68rem 1rem;
  background:linear-gradient(135deg,#22E58A 0%,#10B981 50%,#059669 100%);box-shadow:0 8px 26px rgba(34,229,138,.35),inset 0 1px 0 rgba(255,255,255,.5);}
.stButton>button[kind="primary"]:hover{color:#02130a;transform:translateY(-2px);filter:brightness(1.08);box-shadow:0 14px 36px rgba(34,229,138,.5),inset 0 1px 0 rgba(255,255,255,.55);}

/* ===== waveform ===== */
.wave{display:flex;align-items:center;gap:2px;height:76px;padding:8px 10px;border-radius:12px;background:rgba(2,10,6,.9);border:1px solid var(--line);}
.wave i{flex:1;min-width:2px;border-radius:2px;background:linear-gradient(var(--cyan),#059669);box-shadow:0 0 8px rgba(34,229,138,.45);transform-origin:center;animation:wv .9s ease-out both;}
@keyframes wv{from{transform:scaleY(.05);opacity:0}}
.wmeta{display:flex;justify-content:space-between;font-family:'JetBrains Mono',monospace;font-size:10.5px;color:var(--dim);margin:6px 2px 2px 2px;}

/* ===== badges ===== */
.badge{display:inline-flex;align-items:center;gap:8px;padding:6px 14px;border-radius:8px;font-weight:800;font-size:12px;letter-spacing:1.6px;font-family:'JetBrains Mono',monospace;}
.badge i{width:8px;height:8px;border-radius:50%;background:currentColor;box-shadow:0 0 10px currentColor;animation:pulse 1.6s infinite;}
.b-crit{color:#FF6B6B;background:rgba(239,68,68,.12);border:1px solid rgba(239,68,68,.6);box-shadow:0 0 26px rgba(239,68,68,.4),inset 0 0 16px rgba(239,68,68,.12);}
.b-warn{color:#FBBF24;background:rgba(245,158,11,.12);border:1px solid rgba(245,158,11,.55);box-shadow:0 0 24px rgba(245,158,11,.3);}
.b-low{color:#34D399;background:rgba(16,185,129,.12);border:1px solid rgba(16,185,129,.55);box-shadow:0 0 26px rgba(16,185,129,.38),inset 0 0 16px rgba(16,185,129,.10);}
.b-idle{color:var(--mut);background:rgba(255,255,255,.04);border:1px solid var(--hair);}
.b-idle i{animation:none;}

/* ===== holographic gauge ===== */
.gwrap{display:flex;align-items:center;gap:28px;flex-wrap:wrap;}
.gauge{position:relative;width:200px;height:200px;flex:none;border-radius:50%;display:grid;place-items:center;
  background:conic-gradient(var(--c) calc(var(--p)*1%),rgba(255,255,255,.06) 0);box-shadow:0 0 50px var(--g),0 0 0 1px rgba(255,255,255,.04);
  animation:fillg 1.5s cubic-bezier(.2,.8,.2,1);}
@keyframes fillg{from{--p:0}}
.gauge .ticks{position:absolute;inset:-15px;border-radius:50%;background:repeating-conic-gradient(rgba(34,229,138,.5) 0 1deg,transparent 1deg 6deg);
  -webkit-mask:radial-gradient(farthest-side,transparent calc(100% - 7px),#000 calc(100% - 6px));mask:radial-gradient(farthest-side,transparent calc(100% - 7px),#000 calc(100% - 6px));animation:spin 50s linear infinite;}
.gauge .ring2{position:absolute;inset:-26px;border-radius:50%;border:1px dashed rgba(34,229,138,.28);animation:spin 36s linear infinite reverse;}
@keyframes spin{to{transform:rotate(360deg)}}
.gauge .in{position:relative;overflow:hidden;width:164px;height:164px;border-radius:50%;background:radial-gradient(circle at 50% 30%,#0b2418,#030c07);
  display:flex;flex-direction:column;align-items:center;justify-content:center;border:1px solid var(--line);}
.gauge .in::after{content:"";position:absolute;left:0;right:0;height:38%;top:-40%;background:linear-gradient(transparent,rgba(34,229,138,.14),transparent);animation:scan 3.2s linear infinite;}
@keyframes scan{to{top:110%}}
.gauge .v{font-family:'JetBrains Mono',monospace;font-size:50px;font-weight:700;line-height:1;color:var(--c);text-shadow:0 0 26px var(--g);}
.gauge .v small{font-size:20px;opacity:.7;}
.gauge .l{font-family:'JetBrains Mono',monospace;font-size:9.5px;letter-spacing:2.2px;color:var(--mut);margin-top:9px;}
.vd{flex:1;min-width:230px;}
.cat{font-size:27px;font-weight:800;letter-spacing:-.6px;margin:12px 0 4px 0;}
.vec{font-family:'JetBrains Mono',monospace;font-size:11.5px;color:var(--cyan);letter-spacing:.3px;}
.vec span{color:var(--dim);margin-right:6px;}
.act{margin-top:16px;padding:12px 14px;border-radius:10px;background:rgba(2,10,6,.7);border:1px solid var(--hair);border-left:3px solid var(--c);font-size:12.5px;line-height:1.55;color:var(--txt);}
.act b{display:block;font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:1.6px;color:var(--c);margin-bottom:4px;}
.mgrid{display:grid;grid-template-columns:repeat(4,1fr);gap:9px;margin-top:12px;}
.mg{padding:10px 12px;border-radius:10px;background:rgba(2,10,6,.7);border:1px solid var(--hair);}
.mg b{display:block;font-family:'JetBrains Mono',monospace;font-size:15px;font-weight:700;color:var(--txt);margin-top:3px;}
.mg span{font-size:9.5px;letter-spacing:1.3px;text-transform:uppercase;color:var(--mut);}

/* ===== neon probability bars ===== */
.prow{display:grid;grid-template-columns:150px 1fr 60px;align-items:center;gap:14px;margin:12px 0;}
.prow .n{font-size:12.5px;font-weight:600;white-space:nowrap;} .prow .n.top{color:var(--c,var(--cyan));}
.track{height:9px;border-radius:99px;background:rgba(255,255,255,.06);overflow:hidden;box-shadow:inset 0 1px 3px rgba(0,0,0,.6);}
.nbar{height:100%;border-radius:99px;background:linear-gradient(90deg,#047857,#22E58A);box-shadow:0 0 14px rgba(34,229,138,.65);animation:grow 1.1s cubic-bezier(.2,.8,.2,1) both;}
.nbar.dim{background:linear-gradient(90deg,#143024,#1f4a35);box-shadow:none;}
.nbar.top{background:linear-gradient(90deg,var(--c2,#047857),var(--c));box-shadow:0 0 16px var(--g);}
@keyframes grow{from{width:0}}
.prow .val{text-align:right;font-family:'JetBrains Mono',monospace;font-size:12.5px;}

/* ===== SHAP matrix ===== */
.shead,.srow{display:grid;grid-template-columns:minmax(120px,1.25fr) 2fr 66px;gap:12px;align-items:center;}
.shead{padding:0 8px 7px 8px;font-size:9.5px;letter-spacing:1.5px;text-transform:uppercase;color:var(--dim);border-bottom:1px solid var(--hair);}
.srow{padding:8px 8px;border-radius:8px;margin-top:3px;transition:background .15s;}
.srow:hover{filter:brightness(1.25);}
.srow .tk{font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--txt);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.srow .tk small{font-size:9px;color:var(--dim);letter-spacing:.8px;margin-left:7px;text-transform:uppercase;}
.div{display:grid;grid-template-columns:1fr 1fr;height:9px;position:relative;}
.div::after{content:"";position:absolute;left:50%;top:-3px;bottom:-3px;width:1px;background:rgba(255,255,255,.22);}
.div .l,.div .r{display:flex;height:100%;background:rgba(255,255,255,.04);}
.div .l{justify-content:flex-end;border-radius:99px 0 0 99px;} .div .r{border-radius:0 99px 99px 0;}
.div .fp,.div .fn{height:100%;animation:grow 1s cubic-bezier(.2,.8,.2,1) both;}
.fp{background:linear-gradient(90deg,#B91C1C,#EF4444);box-shadow:0 0 12px rgba(239,68,68,.7);border-radius:0 99px 99px 0;}
.fn{background:linear-gradient(270deg,#047857,#10B981);box-shadow:0 0 12px rgba(16,185,129,.65);border-radius:99px 0 0 99px;}
.pc{text-align:right;font-family:'JetBrains Mono',monospace;font-size:12.5px;font-weight:700;}
.pos{color:#FF6B6B;text-shadow:0 0 12px rgba(239,68,68,.45);} .neg{color:#34D399;text-shadow:0 0 12px rgba(16,185,129,.4);}
.note{margin-top:10px;font-size:11px;color:var(--dim);line-height:1.5;} .note code{font-family:'JetBrains Mono',monospace;color:var(--mut);}

/* ===== terminal console ===== */
.term{border-radius:12px;overflow:hidden;background:#020A06;border:1px solid var(--line2);box-shadow:0 0 30px rgba(34,229,138,.08),inset 0 0 40px rgba(34,229,138,.03);}
.term .bar{display:flex;align-items:center;gap:7px;padding:9px 13px;background:rgba(255,255,255,.035);border-bottom:1px solid var(--hair);}
.term .bar i{width:10px;height:10px;border-radius:50%;} .term .bar i:nth-child(1){background:#EF4444}.term .bar i:nth-child(2){background:#F59E0B}.term .bar i:nth-child(3){background:#10B981}
.term .bar span{margin-left:8px;font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--dim);}
.term .body{padding:12px 15px;max-height:270px;overflow:auto;font-family:'JetBrains Mono',monospace;font-size:11.5px;line-height:1.8;}
.ln{white-space:pre-wrap;word-break:break-word;opacity:0;animation:tin .35s forwards;color:#B7D4C4;}
@keyframes tin{from{opacity:0;transform:translateX(-8px)}to{opacity:1;transform:none}}
.ln .ts{color:var(--dim);margin-right:9px;}
.lv{font-weight:700;margin-right:8px;} .lv.INFO{color:var(--cyan)} .lv.SUCCESS{color:var(--green)} .lv.WARN{color:var(--amber)} .lv.ERROR{color:var(--red)}
.cur::after{content:"█";color:var(--cyan);margin-left:2px;animation:blink 1s steps(2) infinite;} @keyframes blink{50%{opacity:0}}

/* ===== inspected content ===== */
.insp{background:rgba(2,10,6,.9);border:1px solid var(--line);border-radius:12px;padding:13px 15px;font-size:13.5px;line-height:1.75;max-height:200px;overflow:auto;word-break:break-word;}
.insp mark{background:rgba(239,68,68,.16);color:#FFB4B4;padding:1px 6px;border-radius:5px;border:1px solid rgba(239,68,68,.42);box-shadow:0 0 10px rgba(239,68,68,.2);}
.insp mark.safe{background:rgba(16,185,129,.14);color:#9BF0CE;border-color:rgba(16,185,129,.4);box-shadow:none;}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:9px;margin-top:11px;}
.stat{background:rgba(2,10,6,.8);border:1px solid var(--hair);border-radius:10px;padding:10px 12px;}
.stat b{display:block;font-family:'JetBrains Mono',monospace;font-size:16px;color:var(--cyan);} .stat span{font-size:9.5px;color:var(--mut);letter-spacing:1.2px;text-transform:uppercase;}
.pipe{display:flex;align-items:center;justify-content:center;gap:8px;flex-wrap:wrap;margin-top:16px;font-family:'JetBrains Mono',monospace;font-size:11px;}
.pipe span{padding:6px 11px;border-radius:8px;border:1px solid var(--line2);color:var(--mut);background:rgba(34,229,138,.04);} .pipe em{color:var(--cyan);font-style:normal;}
.sub{color:var(--mut);font-size:12.5px;line-height:1.55;}

@media(max-width:900px){.mgrid{grid-template-columns:repeat(2,1fr)}.tgrid{grid-template-columns:1fr}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
LEVELS = {  # level -> (colour, dark shade, glow, badge css, badge text)
    "CRITICAL": ("#EF4444", "#991B1B", "rgba(239,68,68,.5)", "b-crit", "CRITICAL THREAT"),
    "ELEVATED": ("#F59E0B", "#92400E", "rgba(245,158,11,.4)", "b-warn", "ELEVATED RISK"),
    "LOW": ("#10B981", "#065F46", "rgba(16,185,129,.45)", "b-low", "LOW RISK"),
}

SAMPLES = {
    "HDFC Phishing": "HDFC Bank Alert: Your account will be BLOCKED today due to incomplete KYC. Verify your login and PAN immediately at http://hdfc-kyc-update.xyz/secure or your account will be frozen within 2 hours!",
    "Microsoft Support Scam": "This is Microsoft Tech Support. We detected a virus on your computer and your account has been compromised. Do not tell anyone. Call immediately and give us remote access, or pay the fee with gift cards to avoid legal action.",
    "OTP Verification": "Your OTP for the transaction of Rs 49,999 is 482913. Share this OTP with our bank executive immediately to cancel the payment, or your account will be suspended. Do not ignore this message!",
}
SAMPLE_ICONS = {"HDFC Phishing": "🏦", "Microsoft Support Scam": "🖥️", "OTP Verification": "🔑"}


def esc(s) -> str:
    """HTML-escape (and neutralise '$' so Streamlit never renders user text as LaTeX)."""
    return html.escape(str(s)).replace("$", "&#36;")


def H(s: str):
    """Render HTML; strip indentation/blank lines so Markdown never treats it as a code block."""
    st.markdown("\n".join(l.strip() for l in s.splitlines() if l.strip()), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# API layer
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=4, show_spinner=False)
def fetch_telemetry():
    try:
        r = requests.get(f"{API}/telemetry", timeout=2)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


# ---- v2 response -> dashboard schema ------------------------------------- #
VECTORS = {
    "Benign": "No malicious vector identified",
    "Phishing": "Deceptive link / spoofed brand notice",
    "Credential Harvesting": "Password / OTP / PIN capture",
    "Tech Support Impersonation": "Fake support · remote-access lure",
    "Financial Coercion": "Urgency + threat · payment demand",
    "Crypto Fraud": "Wallet drain / investment bait",
}


def risk_level(pct: float) -> str:
    return "CRITICAL" if pct >= 70 else "ELEVATED" if pct >= 35 else "LOW"


def display_class(name: str) -> str:
    return "Safe" if name == "Benign" else name


def indicators(res: dict) -> list[dict]:
    """Attribution rows for the contribution matrix (IG when available, else span-head evidence)."""
    exp = res.get("explanation")
    rows = []
    if exp:
        for key, name, kind in (("top_text_tokens", "token", "text"), ("top_url_segments", "segment", "url"),
                                ("top_acoustic_features", "feature", "acoustic")):
            rows += [{"token": r[name], "impact": r["attribution"], "type": kind} for r in exp.get(key, [])]
    else:
        spans = res["suspicious_spans"]
        rows = [{"token": s["text"], "impact": s["score"], "type": kind}
                for kind in ("text", "url") for s in spans[kind]]
    rows = [r for r in rows if r["token"].strip() and r["impact"] != 0]
    rows.sort(key=lambda r: -abs(r["impact"]))
    rows = rows[:12]
    total = sum(abs(r["impact"]) for r in rows) or 1.0
    for r in rows:
        r["impact_pct"] = round(r["impact"] / total * 100, 1)
        r["impact"] = round(r["impact"], 4)
    return rows


def build_log(res: dict, source: str, latency_ms: float) -> list[dict]:
    stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    u = res["uncertainty"]
    present = res["modalities_present"]
    lines = [("INFO", f"POST /v1/predict · source={source} · modalities: {', '.join(present) or 'none'}")]
    if "text" in present:
        lines.append(("INFO", "Text → frozen transformer · 768-d token states"))
    if res.get("url_analyzed"):
        lines.append(("INFO", f"URL → Char-CNN (k=3/5/7) · {res['url_analyzed'][:60]}"))
    if "acoustic" in present:
        lines.append(("INFO", "Audio → Librosa · MFCC / pitch / jitter / shimmer"))
    lines.append(("INFO", "Gated fusion · " + " · ".join(f"{k} {v:.2f}" for k, v in res["modality_gates"].items())))
    lines.append(("INFO", f"MC dropout · {u['mc_samples']} stochastic passes"))
    if u["is_out_of_distribution"]:
        lines.append(("WARN", f"OOD score {u['ood_score']:.4f} > threshold {u['ood_threshold']:.4f} · flag for human review"))
    elif u["ood_threshold"] is not None:
        lines.append(("INFO", f"OOD score {u['ood_score']:.4f} ≤ threshold {u['ood_threshold']:.4f}"))
    n_spans = len(res["suspicious_spans"]["text"]) + len(res["suspicious_spans"]["url"])
    lines.append(("INFO", f"Span head · {n_spans} suspicious span(s)"))
    if res.get("explanation"):
        lines.append(("INFO", f"Integrated Gradients · completeness Δ {res['explanation']['convergence_delta']}"))
    lvl = "WARN" if res["verdict"] == "scam" else "SUCCESS"
    lines.append((lvl, f"Verdict: {res['verdict'].upper()} · {res['category']} · risk {res['risk_score']:.0%} · {latency_ms:.0f}ms"))
    return [{"ts": stamp, "level": l, "msg": m} for l, m in lines]


def adapt(res: dict, source: str, latency_ms: float, engine: str | None = None) -> dict:
    """Map a v2 API response onto the fields the dashboard renderers use (raw v2 result kept under 'v2')."""
    pct = res["risk_score"] * 100
    level = risk_level(pct)
    tax = res["taxonomy"]
    cat = res["category"]
    review = res["needs_review"]
    latency = res.get("latency_ms", latency_ms)
    steps = ("Do not click links, reply, or share OTP / PIN / passwords. Block the sender and report it "
             "via cybercrime.gov.in or helpline 1930.")
    if level == "LOW":
        action = "No strong scam indicators. Stay cautious with unexpected links or payment requests."
        if review:
            action += " This message is unlike the training data, so double-check it before trusting it."
    else:
        verdict = "LIKELY SCAM" if level == "CRITICAL" else "SUSPICIOUS"
        action = f"{verdict}: {steps}"
        if review:
            # the risk verdict stands; the flag only says the exact pattern is new to the model
            action += " Note: this wording is new to the model, so the scam type may be less certain."
    return {
        "risk_score": round(pct, 1),
        "risk_level": level,
        "threat_category": display_class(cat),
        "confidence_score": round(max(res["risk_score"], 1 - res["risk_score"]) * 100, 1),
        "category_confidence": round(tax[cat] * 100, 1),
        "attack_vector": VECTORS.get(cat, cat) + (" · OOD" if review else ""),
        "recommended_action": action,
        "class_probabilities": {display_class(k): v for k, v in tax.items()},
        "shap_tokens": indicators(res),
        "engine": engine or "multitask",
        "source": source,
        "latency_ms": latency,
        "execution_log": build_log(res, source, latency),
        "transcript": res.get("transcript") or "",
        "v2": res,
    }


def _post(path: str, source: str, **kwargs) -> dict:
    t0 = time.perf_counter()
    r = requests.post(f"{API}{path}", **kwargs)
    r.raise_for_status()
    return adapt(r.json(), source, (time.perf_counter() - t0) * 1000, (fetch_telemetry() or {}).get("model_kind"))


def call_text(text: str):
    return _post("/v1/predict", "text", json={"text": text, "mc_samples": MC_SAMPLES, "explain": True}, timeout=120)


def call_url(url: str):
    return _post("/v1/predict", "url", json={"url": url, "mc_samples": MC_SAMPLES, "explain": True}, timeout=60)


def call_audio(f):
    return _post("/v1/predict/audio", "audio", files={"file": (f.name, f.getvalue())},
                 data={"mc_samples": str(MC_SAMPLES), "explain": "true"}, timeout=300)


def flash(kind: str, msg: str):
    st.session_state["flash"] = (kind, msg)


def run(fn, *args, source: str, shown: str | None = None):
    try:
        with st.spinner("Running inference pipeline…"):
            res = fn(*args)
        st.session_state["result"] = res
        st.session_state["source"] = source
        st.session_state["shown"] = shown if shown is not None else res.get("transcript", "")
    except requests.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        if e.response is not None and e.response.status_code == 404:
            detail += (" — the API on this port has no v2 endpoints. It is probably the old v1 backend "
                       "(backend.main). Stop it and start v2 from the scamshield-v2 folder: uvicorn app:app --port 8000")
        flash("error", f"API error: {detail}")
    except requests.RequestException:
        flash("error", f"Cannot reach the ScamShield API at {API}. {START_HINT}")


def queue(kind: str):
    st.session_state["pending"] = kind


def queue_sample(label: str):
    """Autofill the text area and immediately trigger a scan on the next run."""
    st.session_state["text_input"] = SAMPLES[label]
    st.session_state["pending"] = "text"


def call_combined(text: str, url: str, f):
    """One scan over every input that was filled in: the model fuses them in a single pass."""
    if f is not None:
        data = {"mc_samples": str(MC_SAMPLES), "explain": "true"}
        if text:
            data["text"] = text
        if url:
            data["url"] = url
        return _post("/v1/predict/audio", "combined", files={"file": (f.name, f.getvalue())}, data=data, timeout=300)
    body = {"mc_samples": MC_SAMPLES, "explain": True, "text": text or None, "url": url or None}
    return _post("/v1/predict", "combined", json=body, timeout=120)


def process_pending():
    kind = st.session_state.pop("pending", None)
    if not kind:
        return
    if kind == "text":
        txt = st.session_state.get("text_input", "").strip()
        if len(txt) < 3:
            flash("warning", "Enter some text first.")
        else:
            run(call_text, txt, source="text", shown=txt)
    elif kind == "url":
        url = st.session_state.get("url_input", "").strip()
        if len(url) < 3:
            flash("warning", "Enter a URL first.")
        else:
            run(call_url, url, source="url", shown=url)
    elif kind == "audio":
        f = st.session_state.get("audio_file")
        if not f:
            flash("warning", "Upload an audio file first.")
        else:
            run(call_audio, f, source="audio")
    elif kind == "combined":
        txt = st.session_state.get("combo_text", "").strip()
        url = st.session_state.get("combo_url", "").strip()
        f = st.session_state.get("combo_audio")
        if sum([len(txt) >= 3, len(url) >= 3, f is not None]) < 1:
            flash("warning", "Fill in at least one of message, link or call recording.")
        else:
            run(call_combined, txt, url, f, source="combined", shown=txt or None)
    elif kind == "batch":
        run_batch()


# --------------------------------------------------------------------------- #
# Audio waveform (computed locally for the preview)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, max_entries=8)
def audio_envelope(data: bytes, suffix: str, bars: int = 96):
    import librosa

    try:
        y, sr = librosa.load(io.BytesIO(data), sr=8000, mono=True)
    except Exception:  # some decoders need a real file path
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            path = tmp.name
        try:
            y, sr = librosa.load(path, sr=8000, mono=True)
        finally:
            os.unlink(path)
    if len(y) == 0:
        return None
    env = np.array([c.mean() if len(c) else 0.0 for c in np.array_split(np.abs(y), bars)])
    env = env / (env.max() or 1.0)
    return env.tolist(), len(y) / sr


def render_waveform(f):
    try:
        out = audio_envelope(f.getvalue(), os.path.splitext(f.name)[1])
    except Exception:
        out = None
    if not out:
        st.caption("Waveform preview unavailable for this file")
        return
    env, dur = out
    bars = "".join(f'<i style="height:{max(6, int(v * 100))}%;animation-delay:{i * 8}ms"></i>' for i, v in enumerate(env))
    H(f'<div class="wave">{bars}</div><div class="wmeta"><span>{esc(f.name)}</span><span>SPECTRAL ENVELOPE · {dur:.1f}s</span></div>')


# --------------------------------------------------------------------------- #
# Renderers — telemetry
# --------------------------------------------------------------------------- #
def spark(vals: list[float], w=150, h=26) -> str:
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1
    pts = " ".join(f"{i * w / (len(vals) - 1):.1f},{h - 3 - (v - lo) / rng * (h - 6):.1f}" for i, v in enumerate(vals))
    return (f'<svg class="spark" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><polyline points="{pts}" fill="none" '
            f'stroke="#22E58A" stroke-width="1.6" stroke-linejoin="round" style="filter:drop-shadow(0 0 4px #22E58A)"/></svg>')


def meter(label: str, pct) -> str:
    if pct is None:
        return f'<div class="meter"><span>{label}</span><em>N/A</em></div>'
    return (f'<div class="meter"><span>{label}</span><div class="track"><div class="nbar" style="width:{min(pct, 100):.0f}%"></div></div>'
            f'<b>{pct:.0f}%</b></div>')


def render_feed(t):
    if not t:
        H('<div class="card"><div class="sec">Threat Intelligence Feed<span class="r">offline</span></div><div class="sub">Feed unavailable — backend not reachable.</div></div>')
        return
    if not t.get("feed"):
        H('<div class="card"><div class="sec">Threat Intelligence Feed<span class="r">live scans on this server</span></div><div class="sub">No scans yet. Results of every scan served by the API appear here.</div></div>')
        return
    items = "".join(
        f'<div class="fi"><span class="t">{esc(e["ts"])}</span><span class="sev sv-{e["severity"]}">{e["severity"]}</span>'
        f'<span class="v">{esc(e["vector"])}<small>{esc(e["detail"])}</small></span>'
        f'<span class="src{" live" if e["source"] == "LIVE" else ""}">{e["source"]}</span></div>'
        for e in t["feed"]
    )
    dur = max(18, len(t["feed"]) * 2.6)
    H(f'<div class="card"><div class="sec">Threat Intelligence Feed<span class="r">live scans on this server</span></div>'
      f'<div class="feed"><div class="ftrack" style="--d:{dur:.0f}s">{items}{items}</div></div></div>')


def render_tiles(t):
    if not t:
        lat = rules = cpu = wh = '<div class="tv r">—</div>'
        H(f'<div class="tgrid"><div class="tile"><div class="tl">Model Latency</div>{lat}<div class="ts">backend offline</div></div>'
          f'<div class="tile"><div class="tl">Multi-Task Model</div>{rules}<div class="ts">backend offline</div></div>'
          f'<div class="tile"><div class="tl">CPU / GPU Load</div>{cpu}<div class="ts">backend offline</div></div>'
          f'<div class="tile"><div class="tl">Whisper STT</div>{wh}<div class="ts">backend offline</div></div></div>')
        return
    avg = t.get("latency_ms_avg")
    lat_v = f'{avg:.0f}<small>ms avg</small>' if avg is not None else '—<small>no scans yet</small>'
    gpu = t["gpu"]
    gpu_meter = meter("GPU", gpu.get("load_percent")) if gpu.get("available") else ""  # hide GPU row when no GPU
    wh = t["whisper"]
    wh_cls = {"ready": "g", "loading": "a", "error": "r", "idle": ""}.get(wh["status"], "")
    wh_txt = {"ready": "LOADED", "loading": "LOADING", "error": "ERROR", "idle": "STANDBY"}.get(wh["status"], wh["status"].upper())
    ms = t["model_status"]
    eng_cls = {"ready": "g", "loading": "a", "error": "r"}.get(ms, "a")
    params = t.get("parameters")
    params_v = f'{params / 1e6:.1f}<small>M params</small>' if params else '—<small>no model</small>'
    kind = (t.get("model_kind") or "not loaded").upper()
    thr = t.get("ood_threshold")
    thr_txt = f"OOD τ = {thr:.4f}" if thr is not None else "OOD threshold not calibrated"
    H(f"""<div class="tgrid">
    <div class="tile"><div class="tl">Model Latency</div><div class="tv">{lat_v}</div>{spark(t.get("latency_history", []))}<div class="ts">{t["scans_total"]} scans served</div></div>
    <div class="tile" style="--tc:rgba(16,185,129,.18)"><div class="tl">Multi-Task Model</div><div class="tv {eng_cls}">{params_v}</div><div class="ts">{esc(kind)} · 3 heads · MC dropout</div><div class="ts" style="margin-top:3px">{thr_txt}</div></div>
    <div class="tile" style="--tc:rgba(5,150,105,.2)"><div class="tl">System Load</div>{meter("CPU", t.get("cpu_percent"))}{gpu_meter}<div class="ts" style="margin-top:7px">{esc(gpu["name"]) if gpu.get("available") else "Inference runs on CPU"}</div></div>
    <div class="tile" style="--tc:rgba(245,158,11,.16)"><div class="tl">Whisper STT</div><div class="tv {wh_cls}">{wh_txt}</div><div class="ts">model: {esc(wh["model"])}</div><div class="ts" style="margin-top:3px">uptime {t["uptime_sec"] // 60}m {t["uptime_sec"] % 60}s</div></div>
    </div>""")


def live_fragment(fn):
    """Auto-refresh telemetry every 5s without rerunning the whole app (Streamlit >= 1.37)."""
    frag = getattr(st, "fragment", None)
    try:
        return frag(run_every=5)(fn) if frag else fn
    except TypeError:
        return fn


@live_fragment
def telemetry_panel():
    t = fetch_telemetry()
    render_feed(t)
    render_tiles(t)


# --------------------------------------------------------------------------- #
# Renderers — results
# --------------------------------------------------------------------------- #
def render_topbar():
    t = fetch_telemetry()
    res = st.session_state.get("result")
    if t is None:
        led, state, engine = "led off", "OFFLINE", "—"
    else:
        ms = t["model_status"]
        led = "led" if ms == "ready" else "led warn"
        state = {"ready": "ONLINE", "loading": "ML LOADING", "error": "DEGRADED", "starting": "STARTING"}.get(ms, "ONLINE")
        engine = f'MULTITASK {(t.get("model_kind") or "").upper()} + MC DROPOUT' if ms == "ready" else "NO MODEL LOADED"
    if res and res.get("latency_ms") is not None:
        lat = f'{res["latency_ms"]:.0f}ms'
    elif t and t.get("latency_ms_avg") is not None:
        lat = f'{t["latency_ms_avg"]:.0f}ms'
    else:
        lat = "—"
    scans = t["scans_total"] if t else "—"
    H(f"""<div class="topbar">
    <div class="brand"><div class="logo"><svg viewBox="0 0 24 24" fill="none" stroke="#22E58A" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l7.5 3v5.5c0 4.6-3.1 8.2-7.5 9.5-4.4-1.3-7.5-4.9-7.5-9.5V6L12 3z"/><path d="M8.8 12.2l2.2 2.2 4.2-4.6"/></svg></div><div><h1>Scam<span>Shield</span></h1><p>Multimodal Threat Intelligence</p></div></div>
    <div class="status">
    <span class="chip"><span class="led cy"></span>ENGINE: <b>{engine}</b></span>
    <span class="chip">LATENCY: <b>{lat}</b></span>
    <span class="chip">SCANS: <b>{scans}</b></span>
    <span class="chip"><span class="{led}"></span>STATUS: <b>{state}</b></span>
    </div></div>""")


def highlight_spans(text: str, spans: list[dict]) -> str:
    """Mark the exact character ranges returned by the span-extraction head."""
    out, last = [], 0
    for sp in sorted(spans, key=lambda s: s["start"]):
        start, end = max(sp["start"], last), sp["end"]
        if start >= end:
            continue
        out.append(esc(text[last:start]))
        out.append(f'<mark title="span score {sp["score"]:.2f}">{esc(text[start:end])}</mark>')
        last = end
    out.append(esc(text[last:]))
    return "".join(out).replace("\n", "<br>")


def render_verdict(res: dict):
    c, c2, g, bcls, btxt = LEVELS[res["risk_level"]]
    risk = res["risk_score"]
    src = res.get("source", "text").upper()
    eng = f'{str(res.get("engine") or "").upper()} MULTITASK'
    v2 = res.get("v2")
    unc_row = ""
    if v2:
        u, g = v2["uncertainty"], v2["modality_gates"]
        ood_col = "#FBBF24" if u["is_out_of_distribution"] else "var(--txt)"
        unc_row = f"""<div class="mgrid" style="grid-template-columns:repeat(2,1fr)">
        <div class="mg"><span>OOD score</span><b style="color:{ood_col}">{u["ood_score"]:.4f}</b></div>
        <div class="mg"><span>Gates T·A·U</span><b>{g["text"]:.2f}·{g["acoustic"]:.2f}·{g["url"]:.2f}</b></div></div>"""
    H(f"""<div class="card" style="--c:{c};--c2:{c2};--g:{g}">
    <div class="sec">Threat Verdict<span class="r">{src} SCAN · {eng}</span></div>
    <div class="gwrap">
    <div class="gauge" style="--p:{risk}"><div class="ticks"></div><div class="ring2"></div><div class="in"><span class="v">{risk:.0f}<small>%</small></span><span class="l">RISK INDEX</span></div></div>
    <div class="vd"><span class="badge {bcls}"><i></i>{btxt}</span>
    <div class="cat">{esc(res["threat_category"])}</div>
    <div class="vec"><span>ATTACK VECTOR</span>{esc(res["attack_vector"])}</div>
    <div class="act"><b>RECOMMENDED ACTION</b>{esc(res["recommended_action"])}</div></div></div>
    <div class="mgrid">
    <div class="mg"><span>Confidence</span><b>{res["confidence_score"]:.1f}%</b></div>
    <div class="mg"><span>Class conf.</span><b>{res.get("category_confidence", 0):.1f}%</b></div>
    <div class="mg"><span>Latency</span><b>{res["latency_ms"]:.0f} ms</b></div>
    <div class="mg"><span>Indicators</span><b>{len(res["shap_tokens"])}</b></div></div>{unc_row}</div>""")


def render_probs(res: dict):
    c, c2, g, _, _ = LEVELS[res["risk_level"]]
    probs = sorted(res["class_probabilities"].items(), key=lambda kv: -kv[1])
    rows = "".join(
        f'<div class="prow"><div class="n{" top" if i == 0 else ""}">{esc(k)}</div>'
        f'<div class="track"><div class="nbar{" top" if i == 0 else (" dim" if v < .05 else "")}" style="width:{max(v * 100, 1.2):.1f}%;animation-delay:{i * 90}ms"></div></div>'
        f'<div class="val">{v * 100:.1f}%</div></div>'
        for i, (k, v) in enumerate(probs)
    )
    H(f'<div class="card" style="--c:{c};--c2:{c2};--g:{g}"><div class="sec">Probability Breakdown<span class="r">{"share of items per class" if res.get("source") == "batch" else "taxonomy head · MC-dropout mean"}</span></div>{rows}</div>')


def render_shap(res: dict):
    toks = res["shap_tokens"]
    with st.container(border=True):
        H('<div class="sec" style="margin-top:6px">Feature Attribution Matrix<span class="r">Integrated Gradients · signed share</span></div>')
        mode = st.radio("Filter", ["ALL", "RISK ▲", "SAFE ▼"], horizontal=True, key="shap_filter", label_visibility="collapsed")
        shown = [t for t in toks if mode == "ALL" or (mode == "RISK ▲" and t["impact"] > 0) or (mode == "SAFE ▼" and t["impact"] < 0)]
        if not shown:
            H('<div class="sub" style="padding:10px 4px">No indicators in this class for the current scan.</div>')
            return
        mx = max(abs(t["impact_pct"]) for t in shown) or 1
        rows = []
        for t in shown:
            pos = t["impact"] > 0
            w = abs(t["impact_pct"]) / mx * 100
            tint = .06 + .20 * abs(t["impact_pct"]) / mx
            bg = f"linear-gradient(90deg,rgba({'239,68,68' if pos else '16,185,129'},{tint:.2f}),transparent 75%)"
            # grid order: left half (negative) then right half (positive)
            bar = (f'<div class="l"></div><div class="r"><div class="fp" style="width:{w:.1f}%"></div></div>' if pos
                   else f'<div class="l"><div class="fn" style="width:{w:.1f}%"></div></div><div class="r"></div>')
            rows.append(
                f'<div class="srow" style="background:{bg}"><div class="tk" title="{esc(t["token"])}">{esc(t["token"])}<small>{esc(t["type"])}</small></div>'
                f'<div class="div">{bar}</div><div class="pc {"pos" if pos else "neg"}">{t["impact_pct"]:+.1f}%</div></div>')
        H('<div class="shead"><span>Indicator</span><span>◀ reduces risk · increases risk ▶</span><span style="text-align:right">Weight</span></div>' + "".join(rows)
          + '<div class="note">Percent = each indicator\'s share of the total absolute attribution. Types: <code>text</code> token · <code>url</code> sub-domain / path segment · <code>acoustic</code> Librosa feature.</div>')


def console_html(lines: list[dict], title="inference.log") -> str:
    body = "".join(
        f'<div class="ln" style="animation-delay:{i * 110}ms"><span class="ts">{esc(l["ts"])}</span>'
        f'<span class="lv {l["level"]}">[{l["level"]}]</span>{esc(l["msg"])}</div>'
        for i, l in enumerate(lines)
    )
    return (f'<div class="term"><div class="bar"><i></i><i></i><i></i><span>scamshield@soc:~$ tail -f {title}</span></div>'
            f'<div class="body">{body}<div class="ln cur" style="animation-delay:{len(lines) * 110}ms"></div></div></div>')


def render_console(res: dict):
    H(f'<div class="card"><div class="sec">Inference Console<span class="r">{len(res["execution_log"])} events · {res["latency_ms"]:.0f}ms</span></div>{console_html(res["execution_log"])}</div>')


def render_inspected(res: dict):
    shown = st.session_state.get("shown", "")
    if not shown:
        return
    src = st.session_state.get("source", "text")
    label = {"audio": "Transcript · Whisper STT", "url": "Inspected URL",
             "combined": "Combined Input · message + link + call"}.get(src, "Inspected Content")
    v2 = res.get("v2") or {}
    spans = v2.get("suspicious_spans", {})
    body = highlight_spans(shown, spans.get("url" if src == "url" else "text", []))
    extra = ""
    if src != "url" and v2.get("url_analyzed") and spans.get("url"):
        extra = f'<div class="insp" style="margin-top:9px;font-family:JetBrains Mono,monospace;font-size:12px">{highlight_spans(v2["url_analyzed"], spans["url"])}</div>'
    H(f'<div class="card"><div class="sec">{label}<span class="r">span-head evidence highlighted</span></div><div class="insp">{body}</div>{extra}</div>')


def render_idle():
    t = fetch_telemetry()
    mode = "no model loaded" if (not t or t["model_status"] != "ready") else f'multi-task {t.get("model_kind") or ""}'
    params = f'{t["parameters"]:,}' if t and t.get("parameters") else "—"
    boot = [
        {"ts": "--:--:--.---", "level": "INFO", "msg": "ScamShield inspection engine initialised"},
        {"ts": "--:--:--.---", "level": "INFO" if t else "ERROR", "msg": f"Backend link {'established' if t else 'FAILED — start uvicorn on :8000'} · engine: {mode}"},
        {"ts": "--:--:--.---", "level": "INFO", "msg": f"Model armed · {params} parameters · 3 heads · MC dropout"},
        {"ts": "--:--:--.---", "level": "INFO", "msg": "Awaiting input · text, call audio or URL"},
    ]
    H("""<div class="card"><div class="sec">Threat Verdict<span class="r">awaiting input</span></div>
    <div class="gwrap" style="justify-content:center;flex-direction:column;gap:18px">
    <div class="gauge" style="--p:0;--c:#22E58A;--g:rgba(34,229,138,.18);animation:none"><div class="ticks"></div><div class="ring2"></div><div class="in"><span class="v" style="color:#1d4a33;text-shadow:none">--</span><span class="l">RISK INDEX</span></div></div>
    <span class="badge b-idle"><i></i>NO SCAN</span></div>
    <div class="sub" style="text-align:center;margin-top:12px">Pick a quick sample or submit content on the left to see the risk gauge, class probabilities, uncertainty, attribution matrix and live inference log.</div>
    <div class="pipe"><span>Char-CNN</span><em>›</em><span>Gated Fusion</span><em>›</em><span>Multi-Task</span><em>›</em><span>MC Dropout</span><em>›</em><span>IG</span></div></div>""")
    H(f'<div class="card"><div class="sec">Inference Console<span class="r">idle</span></div>{console_html(boot, "boot.log")}</div>')


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
TEXT_COLS = ("text", "message", "body", "content", "sms", "msg", "email", "transcript")


def batch_row(source: str, label: str, o: dict) -> dict:
    """Normalise one API result (batch or single-audio) into a results-table row."""
    if "error" in o:
        return {"source": source, "input": label[:110], "risk_score": None, "risk_level": "—",
                "threat_category": o["error"], "confidence": None, "attack_vector": "—", "top_indicators": "—"}
    ind = o.get("top_indicators")
    if ind is None:
        ind = ", ".join(t["token"] for t in o.get("shap_tokens", []) if t["impact"] > 0)[:120] or "-"
    return {"source": source, "input": label[:110], "risk_score": o["risk_score"], "risk_level": o["risk_level"],
            "threat_category": o["threat_category"], "confidence": o["confidence_score"],
            "attack_vector": o["attack_vector"], "top_indicators": ind}


CLASSES = ["Safe", "Phishing", "Credential Harvesting", "Tech Support Impersonation", "Financial Coercion", "Crypto Fraud"]


def aggregate_batch(df: pd.DataFrame, toks: list, blog: list, elapsed: float, engine) -> dict | None:
    """Fold many scored items into ONE result dict with the same schema as a single scan,
    so the right-hand panel (gauge, probabilities, SHAP, console) renders for batches too."""
    ok = df[df["risk_score"].notna()]
    n = len(ok)
    if n == 0:
        return None

    def log(level, msg):
        blog.append({"ts": datetime.now().strftime("%H:%M:%S.%f")[:-3], "level": level, "msg": msg})

    avg = float(ok["risk_score"].mean())
    level = "CRITICAL" if avg >= 70 else "ELEVATED" if avg >= 35 else "LOW"
    n_c = int((ok["risk_level"] == "CRITICAL").sum())
    n_e = int((ok["risk_level"] == "ELEVATED").sum())
    flagged = ok[ok["risk_level"] != "LOW"]
    category = flagged["threat_category"].mode().iloc[0] if len(flagged) else "Safe"
    vector = flagged["attack_vector"].mode().iloc[0] if len(flagged) else "No malicious vector identified"

    log("INFO", f"Aggregating attributions across {n} items")
    sums, types = {}, {}
    for tl in toks:
        for t in tl:
            sums[t["token"]] = sums.get(t["token"], 0.0) + t["impact"]
            types[t["token"]] = t.get("type", "term")
    ranked = sorted(sums.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]
    total = sum(abs(v) for _, v in ranked) or 1.0
    shap_tokens = [{"token": k, "impact": round(v / n, 4), "impact_pct": round(v / total * 100, 1),
                    "direction": "increases risk" if v > 0 else "reduces risk", "type": types.get(k, "term")}
                   for k, v in ranked]

    badge = {"CRITICAL": "CRITICAL THREAT", "ELEVATED": "ELEVATED RISK", "LOW": "LOW RISK"}[level]
    log("SUCCESS", f"Batch verdict: {badge} · avg risk {avg:.1f} · {n_c} critical / {n_e} elevated / {n - n_c - n_e} low")
    action = (f"{n_c} of {n} items are CRITICAL and {n_e} ELEVATED. Review the flagged rows in the report below, block the senders, and never reply "
              "or share OTP/PIN/passwords. Report via cybercrime.gov.in or helpline 1930."
              if n_c + n_e else f"All {n} items look low risk. No action required.")
    cat_conf = ((flagged["threat_category"] == category).mean() if len(flagged) else (ok["risk_level"] == "LOW").mean()) * 100
    return {
        "risk_score": round(avg, 1), "risk_level": level, "threat_category": category,
        "confidence_score": round(float(ok["confidence"].mean()), 1), "category_confidence": round(float(cat_conf), 1),
        "attack_vector": vector, "recommended_action": action,
        "class_probabilities": {c: round(float((ok["threat_category"] == c).mean()), 4) for c in CLASSES},
        "shap_tokens": shap_tokens, "engine": engine or "multitask", "source": "batch",
        "execution_log": blog, "latency_ms": round(elapsed * 1000 / n, 1),
    }


def run_batch():
    """Analyse an uploaded CSV/TSV (auto-detected text column) and/or a set of call recordings."""
    csv_file = st.session_state.get("batch_csv")
    audio_files = st.session_state.get("batch_audio") or []
    if not csv_file and not audio_files:
        flash("warning", "Upload a CSV and/or audio files first.")
        return
    rows, toks, blog, engine = [], [], [], None
    t_start = time.perf_counter()

    def log(level, msg):
        blog.append({"ts": datetime.now().strftime("%H:%M:%S.%f")[:-3], "level": level, "msg": msg})

    log("INFO", f"Batch job started · {'1 CSV' if csv_file else 'no CSV'} + {len(audio_files)} recording(s)")
    try:
        if csv_file:
            try:
                df = pd.read_csv(csv_file, sep=None, engine="python")
            except UnicodeDecodeError:
                csv_file.seek(0)
                df = pd.read_csv(csv_file, sep=None, engine="python", encoding="latin-1")
            col = next((c for c in df.columns if str(c).strip().lower() in TEXT_COLS), None)
            if col is None:  # fall back to the text-like column with the longest average content
                obj = [c for c in df.columns if df[c].dtype == object]
                col = max(obj, key=lambda c: df[c].astype(str).str.len().mean()) if obj else None
            if col is None:
                flash("error", "Could not find a text column in the CSV. Name it 'text' or 'message'.")
                return
            texts = df[col].fillna("").astype(str).str.strip().tolist()
            if len(texts) > 500:
                flash("warning", f"CSV has {len(texts)} rows — only the first 500 were analysed.")
                texts = texts[:500]
            log("INFO", f"File parsed · text column '{col}' · {len(texts)} rows")
            log("INFO", f"POST /v1/predict/batch · {len(texts)} messages")
            with st.spinner(f"Analysing {len(texts)} messages from column '{col}'…"):
                r = requests.post(f"{API}/v1/predict/batch", json={"texts": texts, "mc_samples": min(MC_SAMPLES, 10)}, timeout=900)
                r.raise_for_status()
            data = r.json()
            engine = data.get("engine")
            for t, o in zip(texts, data["results"]):
                o = o if "error" in o else adapt(o, "csv", o.get("latency_ms", 0.0), engine)
                rows.append(batch_row("csv", t, o))
                toks.append(o.get("shap_tokens", []))
            skipped = sum(1 for o in data["results"] if "error" in o)
            log("SUCCESS", f"Scored {len(texts) - skipped} messages · engine: {engine}" + (f" · {skipped} skipped (empty)" if skipped else ""))
        if audio_files:
            prog = st.progress(0.0, text="Transcribing call recordings…")
            for i, f in enumerate(audio_files):
                try:
                    o = call_audio(f)
                    rows.append(batch_row("audio", f"{f.name} — {o.get('transcript', '')}", o))
                    toks.append(o.get("shap_tokens", []))
                    engine = engine or o.get("engine")
                    log("SUCCESS", f"Transcribed + scored {f.name} · risk {o['risk_score']:.0f}/100")
                except requests.HTTPError as e:
                    try:
                        msg = e.response.json().get("detail", str(e))
                    except Exception:
                        msg = str(e)
                    rows.append(batch_row("audio", f.name, {"error": msg[:80]}))
                prog.progress((i + 1) / len(audio_files), text=f"Transcribing {i + 1}/{len(audio_files)}…")
            prog.empty()
    except requests.RequestException:
        flash("error", f"Cannot reach the ScamShield API at {API}. {START_HINT}")
        return
    df_out = pd.DataFrame(rows)
    st.session_state["batch"] = df_out
    agg = aggregate_batch(df_out, toks, blog, time.perf_counter() - t_start, engine)
    if agg:  # fill the right-hand panel exactly like a single scan does
        st.session_state["result"] = agg
        st.session_state["source"] = "batch"
        st.session_state["shown"] = ""


def clear_batch():
    st.session_state.pop("batch", None)


def render_batch(df: pd.DataFrame):
    n = len(df)
    crit = int((df["risk_level"] == "CRITICAL").sum())
    elev = int((df["risk_level"] == "ELEVATED").sum())
    low = int((df["risk_level"] == "LOW").sum())
    avg = df["risk_score"].mean()
    H(f"""<div class="card"><div class="sec">Batch Analysis Report<span class="r">{n} items analysed</span></div>
    <div class="mgrid">
    <div class="mg"><span>Total</span><b>{n}</b></div>
    <div class="mg"><span>Critical</span><b style="color:#FF6B6B">{crit}</b></div>
    <div class="mg"><span>Elevated</span><b style="color:#FBBF24">{elev}</b></div>
    <div class="mg"><span>Low risk</span><b style="color:#34D399">{low}</b></div></div>
    <div class="note">Average risk index: {0 if pd.isna(avg) else avg:.1f}/100 · sorted by highest risk first</div></div>""")
    ordered = df.sort_values("risk_score", ascending=False, na_position="last")
    c1, c2 = st.columns([1, 2.2], gap="medium")
    with c1:
        with st.container(border=True):
            st.caption("Threat categories")
            st.bar_chart(df["threat_category"].value_counts(), color="#22E58A")
    with c2:
        st.dataframe(
            ordered, hide_index=True, height=330,
            column_config={
                "risk_score": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%.0f"),
                "confidence": st.column_config.NumberColumn("Conf %", format="%.0f"),
                "input": st.column_config.TextColumn("Input", width="large"),
            },
        )
    b1, b2 = st.columns([1, 1], gap="medium")
    b1.download_button("⬇  DOWNLOAD RESULTS (CSV)", ordered.to_csv(index=False).encode("utf-8"),
                       "scamshield_batch_results.csv", "text/csv", use_container_width=True)
    b2.button("✕  CLEAR RESULTS", key="clr_batch", on_click=clear_batch)


process_pending()      # run any queued scan first so the whole page reflects it
render_topbar()
if st.session_state.get("batch") is not None:  # batch report sits at the TOP so it is impossible to miss
    render_batch(st.session_state["batch"])

left, right = st.columns([1, 1.12], gap="medium")

with left:
    telemetry_panel()

    with st.container(border=True):  # Multimodal inspection suite (no empty header card)
        fl = st.session_state.pop("flash", None)
        if fl:
            (st.error if fl[0] == "error" else st.warning)(fl[1])
        tab_text, tab_audio, tab_url, tab_combo, tab_batch = st.tabs(
            ["✉  Text", "🎙  Call Audio", "🔗  URL", "🧩  Combined", "📂  Batch Upload"])

        with tab_text:
            st.text_area("Message", key="text_input", height=150, label_visibility="collapsed",
                         placeholder="Paste a suspicious SMS, email or call script…")
            st.button("▶  ANALYZE TEXT", type="primary", key="go_text", on_click=queue, args=("text",))
            st.caption("Live samples · autofill + instant scan")
            cols = st.columns(len(SAMPLES))
            for col, key in zip(cols, SAMPLES):
                col.button(f"{SAMPLE_ICONS[key]} {key}", key=f"s_{key}", on_click=queue_sample, args=(key,))

        with tab_audio:
            up = st.file_uploader("Upload .wav or .mp3", type=["wav", "mp3"], label_visibility="collapsed", key="audio_file")
            if up:
                render_waveform(up)
                st.audio(up)
            st.button("▶  TRANSCRIBE & ANALYZE", type="primary", key="go_audio", on_click=queue, args=("audio",))

        with tab_url:
            st.text_input("URL", key="url_input", label_visibility="collapsed", placeholder="https://secure-login.example-bank.xyz/verify")
            st.button("▶  SCAN URL", type="primary", key="go_url", on_click=queue, args=("url",))
            st.caption("Static lexical analysis · the URL is never fetched or visited")

        with tab_combo:
            st.caption("Fill in any combination · the model fuses them in one scan")
            st.text_area("Message", key="combo_text", height=110, label_visibility="collapsed",
                         placeholder="Message or call script (optional)")
            st.text_input("Link", key="combo_url", label_visibility="collapsed",
                          placeholder="Link (optional) e.g. http://bescom-billpay.top/pay")
            st.file_uploader("Call recording (optional)", type=["wav", "mp3"], key="combo_audio")
            st.button("▶  ANALYZE TOGETHER", type="primary", key="go_combo", on_click=queue, args=("combined",))

        with tab_batch:
            st.caption("1 · Messages file (CSV / TSV / TXT with a text column)")
            st.file_uploader("Messages file", type=["csv", "tsv", "txt"], key="batch_csv", label_visibility="collapsed")
            st.caption("2 · Call recordings (optional, select many)")
            st.file_uploader("Call recordings", type=["wav", "mp3"], accept_multiple_files=True, key="batch_audio", label_visibility="collapsed")
            st.button("▶  ANALYZE UPLOADED DATA", type="primary", key="go_batch", on_click=queue, args=("batch",))
            st.caption("Auto-detects the text column · up to 500 rows per file · report appears below")

    res = st.session_state.get("result")
    if res:
        render_inspected(res)

with right:
    if res:
        render_verdict(res)
        render_probs(res)
        render_shap(res)
        render_console(res)
    else:
        render_idle()
