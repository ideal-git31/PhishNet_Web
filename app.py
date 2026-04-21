"""
PhishNet — Hybrid Phishing Detection System
6th Sem College Project | Adarsh Singh

Explainable AI Edition:
  - WHY flagged: feature contribution panel
  - Confidence breakdown: BERT score vs RF score
  - Attack pattern explanation: contextual tips per trigger
"""

import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
import pandas as pd
import numpy as np
import re
import os
from urllib.parse import urlparse
import time

# ─────────────────────────────────────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────────────────────────────────────
device = torch.device("cpu")

# ─────────────────────────────────────────────────────────────────────────────
# ATTACK PATTERN TIPS — shown when a feature fires
# ─────────────────────────────────────────────────────────────────────────────
ATTACK_TIPS = {
    "url_length": (
        "Long URL Obfuscation",
        "Attackers pad URLs with extra paths and parameters to hide the real "
        "domain and confuse victims who don't read the full address."
    ),
    "count_at": (
        "@ Symbol Trick",
        "Browsers ignore everything before '@' in a URL. "
        "'http://paypal.com@evil.com' actually takes you to evil.com."
    ),
    "count_hyphen": (
        "Hyphen Stuffing",
        "Phishers use hyphens to make fake domains look legitimate — "
        "e.g. 'paypal-secure-login.xyz' mimics PayPal's brand."
    ),
    "count_double_slash": (
        "Double-Slash Redirect",
        "Extra slashes can redirect browsers to attacker-controlled paths, "
        "bypassing simple URL filters that check only the domain."
    ),
    "count_percent": (
        "Percent Encoding",
        "Encoding characters as %XX (e.g. %2F for /) hides the true URL "
        "from security scanners that don't decode before checking."
    ),
    "count_digits": (
        "Digit Substitution",
        "Phishers replace letters with digits — 'paypa1' looks like 'paypal' "
        "at a glance. Always read domain character by character."
    ),
    "count_dots": (
        "Subdomain Abuse",
        "Extra dots create deep subdomains — 'paypal.com.evil.xyz' looks "
        "legitimate but the actual domain is evil.xyz."
    ),
    "digit_letter_ratio": (
        "High Digit Density",
        "Legitimate domains rarely have many digits. A high digit ratio "
        "often signals random/generated domains used in phishing campaigns."
    ),
    "has_ip": (
        "Raw IP Address",
        "Legitimate websites use domain names, not raw IPs. "
        "An IP in the URL almost always means the site has no registered domain."
    ),
    "is_shortened": (
        "URL Shortener",
        "Shorteners hide the real destination. Phishers use them to mask "
        "malicious URLs that would otherwise be obviously suspicious."
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# LEXICAL FEATURE META — thresholds for flagging
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_META = {
    "url_length"         : {"label": "URL Length",           "threshold": 75,   "high_is_bad": True,  "max": 200},
    "count_at"           : {"label": "@ Symbol",             "threshold": 0,    "high_is_bad": True,  "max": 3  },
    "count_hyphen"       : {"label": "Hyphens",              "threshold": 3,    "high_is_bad": True,  "max": 10 },
    "count_double_slash" : {"label": "Double Slashes",       "threshold": 0,    "high_is_bad": True,  "max": 3  },
    "count_percent"      : {"label": "% Encoding",           "threshold": 1,    "high_is_bad": True,  "max": 5  },
    "count_digits"       : {"label": "Digit Count",          "threshold": 8,    "high_is_bad": True,  "max": 20 },
    "count_dots"         : {"label": "Dot Count",            "threshold": 4,    "high_is_bad": True,  "max": 8  },
    "digit_letter_ratio" : {"label": "Digit/Letter Ratio",   "threshold": 0.15, "high_is_bad": True,  "max": 1.0},
    "has_ip"             : {"label": "Raw IP Address",       "threshold": 0,    "high_is_bad": True,  "max": 1  },
    "is_shortened"       : {"label": "URL Shortener",        "threshold": 0,    "high_is_bad": True,  "max": 1  },
}

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADER — single model, no double-load segfault
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    model_path = "./bert_phishing_5k_benchmark"
    tokenizer  = AutoTokenizer.from_pretrained("bert-base-uncased")
    dl_model   = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    dl_model.eval()
    rf_model   = joblib.load("random_forest_model.joblib")
    pca        = joblib.load("pca_compressor.joblib")
    return tokenizer, dl_model, rf_model, pca

tokenizer, dl_model, rf_model, pca = load_models()

# ─────────────────────────────────────────────────────────────────────────────
# LEXICAL EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────
def extract_url_math(user_url: str) -> dict:
    url           = str(user_url)
    url_for_parse = url if url.startswith(("http://", "https://")) else "http://" + url
    domain        = urlparse(url_for_parse).netloc
    num_digits    = sum(c.isdigit() for c in url)
    num_letters   = sum(c.isalpha() for c in url)
    return {
        "url_length"         : len(url),
        "count_at"           : url.count("@"),
        "count_hyphen"       : url.count("-"),
        "count_double_slash" : max(0, url.count("//") - 1),
        "count_percent"      : url.count("%"),
        "count_digits"       : num_digits,
        "count_dots"         : url.count("."),
        "digit_letter_ratio" : num_digits / num_letters if num_letters > 0 else num_digits,
        "has_ip"             : 1 if re.match(
                                   r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$",
                                   domain) else 0,
        "is_shortened"       : 1 if any(s in domain for s in [
                                   "bit.ly", "qrco.de", "t.co", "tinyurl.com",
                                   "l.ead.me", "goo.gl", "ow.ly"]) else 0,
    }

# ─────────────────────────────────────────────────────────────────────────────
# UI CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "PhishNet AI",
    page_icon  = "🛡️",
    layout     = "wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.header-box {
    background: linear-gradient(135deg, #0f172a, #1e3a5f);
    border-radius: 14px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    border-left: 5px solid #3b82f6;
}
.header-box h1 { color: #f8fafc; font-size: 2.2rem; margin: 0; font-weight: 800; }
.header-box p  { color: #94a3b8; font-family: 'JetBrains Mono', monospace;
                 font-size: 0.75rem; margin: 0.3rem 0 0; letter-spacing: 1px; }

.verdict-phish {
    background: linear-gradient(135deg, #1a0a0a, #2d0f0f);
    border: 1px solid #7f1d1d; border-left: 5px solid #ef4444;
    border-radius: 12px; padding: 1.5rem 2rem; margin: 1rem 0;
}
.verdict-safe {
    background: linear-gradient(135deg, #0a1a0f, #0f2d1a);
    border: 1px solid #14532d; border-left: 5px solid #22c55e;
    border-radius: 12px; padding: 1.5rem 2rem; margin: 1rem 0;
}
.verdict-title { font-size: 1.3rem; font-weight: 800; color: #f8fafc; margin: 0 0 0.2rem; }
.verdict-score { font-family: 'JetBrains Mono', monospace;
                 font-size: 3rem; font-weight: 700; line-height: 1; }
.verdict-sub   { font-family: 'JetBrains Mono', monospace;
                 font-size: 0.7rem; color: #94a3b8; letter-spacing: 1.5px;
                 text-transform: uppercase; margin-top: 0.3rem; }

.panel {
    background: #0f172a; border: 1px solid #1e3a5f;
    border-radius: 12px; padding: 1.25rem 1.5rem; height: 100%;
}
.panel-title {
    font-family: 'JetBrains Mono', monospace; font-size: 0.68rem;
    letter-spacing: 2px; text-transform: uppercase;
    color: #3b82f6; border-bottom: 1px solid #1e3a5f;
    padding-bottom: 0.5rem; margin-bottom: 1rem;
}

.feat-row { display: flex; align-items: center; gap: 8px;
            margin: 6px 0; font-size: 0.8rem; }
.feat-name { color: #94a3b8; width: 145px; flex-shrink: 0; }
.feat-track { flex: 1; height: 7px; background: #1e293b;
              border-radius: 4px; overflow: hidden; }
.feat-fill  { height: 100%; border-radius: 4px; }
.feat-val   { font-family: 'JetBrains Mono', monospace;
              color: #cbd5e1; font-size: 0.75rem;
              width: 48px; text-align: right; flex-shrink: 0; }
.feat-flag  { width: 18px; text-align: center; font-size: 0.85rem; }

.score-ring {
    text-align: center; padding: 1rem 0;
}
.score-val  { font-family: 'JetBrains Mono', monospace;
              font-size: 2.2rem; font-weight: 700; }
.score-lbl  { font-size: 0.7rem; color: #64748b;
              letter-spacing: 1px; text-transform: uppercase; }

.tip-box {
    background: #0c1829; border: 1px solid #1e3a5f;
    border-left: 3px solid #f59e0b; border-radius: 8px;
    padding: 0.75rem 1rem; margin: 0.5rem 0;
}
.tip-title { color: #fbbf24; font-weight: 700;
             font-size: 0.82rem; margin-bottom: 0.25rem; }
.tip-body  { color: #94a3b8; font-size: 0.78rem; line-height: 1.5; }

.safe-tip-box {
    background: #0a1a0f; border: 1px solid #14532d;
    border-left: 3px solid #22c55e; border-radius: 8px;
    padding: 0.75rem 1rem; margin: 0.5rem 0;
}
.safe-tip-title { color: #4ade80; font-weight: 700; font-size: 0.82rem; }
.safe-tip-body  { color: #94a3b8; font-size: 0.78rem; line-height: 1.5; }

.mono { font-family: 'JetBrains Mono', monospace; }

[data-testid="stSidebar"] { background: #080c14 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <h1>🛡️ PhishNet</h1>
    <p>EXPLAINABLE HYBRID PHISHING DETECTION · BERT + PCA + RANDOM FOREST · 96.45% ACCURACY</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Engine")
    model_choice = st.selectbox(
        "Intelligence Engine",
        ["Fine-Tuned BERT (96.45%)", "Hybrid PCA-Random Forest"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("### 📐 Architecture")
    st.markdown("""
| Layer | Detail |
|---|---|
| BERT | bert-base-uncased |
| Fine-tuned | 10K URLs |
| Embedding | 768-dim CLS |
| PCA | 768 → 20 |
| Lexical | 10 features |
| RF input | 30 features |
| Device | CPU |
""")
    st.markdown("---")
    st.info("**Adarsh Singh**\n\n6th Sem · College Project\n\nBERT → PCA (768→20)\n+ 10 Lexical Features\n→ Random Forest")

# ─────────────────────────────────────────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────────────────────────────────────────
col_in, col_btn = st.columns([6, 1])
with col_in:
    user_url = st.text_input(
        "URL", placeholder="https://example.com/path",
        label_visibility="collapsed"
    )
with col_btn:
    scan = st.button("🔍 Scan", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE + DISPLAY
# ─────────────────────────────────────────────────────────────────────────────
if scan:
    if not user_url.strip():
        st.warning("Please enter a URL.")
        st.stop()

    with st.spinner("Running inference..."):
        t0 = time.time()

        url_clean = user_url.strip()
        if not url_clean.startswith(("http://", "https://")):
            url_clean = "https://" + url_clean

        inputs = tokenizer(
            url_clean, return_tensors="pt",
            truncation=True, padding=True, max_length=128
        ).to(device)

        # ── Always run BERT for both engines ──────────────────────────────────
        with torch.no_grad():
            outputs = dl_model(**inputs, output_hidden_states=True)

        bert_probs  = F.softmax(outputs.logits, dim=1).squeeze().tolist()
        bert_score  = bert_probs[1]   # P(phishing) from fine-tuned BERT

        # ── Hybrid: BERT embedding → PCA → RF ─────────────────────────────────
        cls_emb      = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
        pca_feats    = pca.transform(cls_emb)
        lex          = extract_url_math(url_clean)
        feat_dict    = {f"pca_feature_{i}": pca_feats[0][i] for i in range(20)}
        feat_dict.update(lex)
        df           = pd.DataFrame([feat_dict])
        df_aligned   = df[rf_model.feature_names_in_]
        rf_proba     = rf_model.predict_proba(df_aligned)[0]
        rf_score     = float(rf_proba[1])

        # Primary score depends on engine choice
        if model_choice == "Fine-Tuned BERT (96.45%)":
            threat_score = bert_score
        else:
            threat_score = rf_score

        elapsed = (time.time() - t0) * 1000

    is_phish = threat_score > 0.5

    # ─────────────────────────────────────────────────────────────────────────
    # VERDICT CARD
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    card_cls   = "verdict-phish" if is_phish else "verdict-safe"
    icon       = "🚨" if is_phish else "✅"
    verdict    = "PHISHING DETECTED" if is_phish else "SAFE DOMAIN"
    clr        = "#ef4444" if is_phish else "#22c55e"

    st.markdown(f"""
    <div class="{card_cls}">
        <div class="verdict-title">{icon} {verdict}</div>
        <div class="verdict-score" style="color:{clr}">{threat_score:.1%}</div>
        <div class="verdict-sub">Phishing probability · {model_choice} · {elapsed:.0f}ms</div>
    </div>
    """, unsafe_allow_html=True)

    st.progress(float(threat_score))

    # ─────────────────────────────────────────────────────────────────────────
    # THREE PANELS: Scores | Features | Attack Patterns
    # ─────────────────────────────────────────────────────────────────────────
    p1, p2, p3 = st.columns([1, 1.6, 1.8])

    # ── PANEL 1: Confidence Breakdown ────────────────────────────────────────
    with p1:
        bert_clr = "#ef4444" if bert_score > 0.5 else "#22c55e"
        rf_clr   = "#ef4444" if rf_score   > 0.5 else "#22c55e"
        agree    = (bert_score > 0.5) == (rf_score > 0.5)
        agree_bg = "#0a1a0f" if agree else "#1a0a0a"
        agree_cl = "#4ade80" if agree else "#f87171"
        agree_bd = "#14532d" if agree else "#7f1d1d"
        agree_tx = "✅ Both agree" if agree else "⚡ Models disagree"

        components.html(f"""
        <style>
          body {{ margin:0; padding:0; background:transparent; font-family:'Inter',sans-serif; }}
        </style>
        <div style="background:#0f172a;border:1px solid #1e3a5f;
                    border-radius:12px;padding:1.25rem 1.5rem;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
                        letter-spacing:2px;text-transform:uppercase;color:#3b82f6;
                        border-bottom:1px solid #1e3a5f;padding-bottom:0.5rem;
                        margin-bottom:1rem;">📊 Confidence Breakdown</div>

            <div style="text-align:center;padding:0.8rem 0;">
                <div style="font-family:'JetBrains Mono',monospace;
                            font-size:2.2rem;font-weight:700;
                            color:{bert_clr}">{bert_score:.1%}</div>
                <div style="font-size:0.7rem;color:#94a3b8;
                            letter-spacing:1px;text-transform:uppercase;">BERT Score</div>
            </div>

            <div style="text-align:center;color:#475569;
                        font-size:1.2rem;margin:0.2rem 0;">↕</div>

            <div style="text-align:center;padding:0.8rem 0;">
                <div style="font-family:'JetBrains Mono',monospace;
                            font-size:2.2rem;font-weight:700;
                            color:{rf_clr}">{rf_score:.1%}</div>
                <div style="font-size:0.7rem;color:#94a3b8;
                            letter-spacing:1px;text-transform:uppercase;">RF Score</div>
            </div>

            <div style="margin-top:0.8rem;text-align:center;font-size:0.78rem;
                        padding:0.4rem 0.8rem;border-radius:6px;
                        background:{agree_bg};color:{agree_cl};
                        border:1px solid {agree_bd};">{agree_tx}</div>

            <div style="margin-top:1rem;">
                <div style="font-size:0.7rem;color:#475569;margin-bottom:0.3rem;">
                    BERT confidence</div>
                <div style="background:#1e293b;border-radius:4px;
                            height:6px;overflow:hidden;margin-bottom:0.6rem;">
                    <div style="width:{bert_score*100:.1f}%;height:100%;
                                background:{bert_clr};border-radius:4px;"></div>
                </div>
                <div style="font-size:0.7rem;color:#475569;margin-bottom:0.3rem;">
                    RF confidence</div>
                <div style="background:#1e293b;border-radius:4px;
                            height:6px;overflow:hidden;">
                    <div style="width:{rf_score*100:.1f}%;height:100%;
                                background:{rf_clr};border-radius:4px;"></div>
                </div>
            </div>
        </div>
        """, height=380)

    # ── PANEL 2: Feature Contributions ───────────────────────────────────────
    with p2:
        rows_html = ""
        triggered = []

        for key, meta in FEATURE_META.items():
            val     = lex[key]
            thresh  = meta["threshold"]
            max_v   = meta["max"]
            is_sus  = val > thresh if meta["high_is_bad"] else val < thresh
            norm    = min(val / max_v, 1.0) if max_v > 0 else 0
            bar_clr = "#ef4444" if is_sus else "#22c55e"
            flag    = "⚠️" if is_sus else "✅"
            bar_pct = int(norm * 100)

            if key == "digit_letter_ratio":
                disp = f"{val:.2f}"
            elif key in ("has_ip", "is_shortened"):
                disp = "YES" if val == 1 else "NO"
            else:
                disp = str(int(val))

            if is_sus:
                triggered.append(key)

            rows_html += f"""
            <div style="display:flex;align-items:center;gap:8px;
                        margin:6px 0;font-size:0.8rem;">
                <div style="width:18px;text-align:center;
                            font-size:0.85rem;">{flag}</div>
                <div style="color:#94a3b8;width:140px;
                            flex-shrink:0;">{meta['label']}</div>
                <div style="flex:1;height:7px;background:#1e293b;
                            border-radius:4px;overflow:hidden;">
                    <div style="width:{bar_pct}%;height:100%;
                                background:{bar_clr};border-radius:4px;"></div>
                </div>
                <div style="font-family:'JetBrains Mono',monospace;
                            color:#cbd5e1;font-size:0.75rem;
                            width:44px;text-align:right;
                            flex-shrink:0;">{disp}</div>
            </div>"""

        summary_bg  = "#1a0a0a" if triggered else "#0a1a0f"
        summary_bdr = "#7f1d1d" if triggered else "#14532d"
        summary_clr = "#fca5a5" if triggered else "#86efac"
        summary_txt = (
            f"⚠️ {len(triggered)} suspicious signal{'s' if len(triggered)!=1 else ''} detected"
            if triggered else "✅ No suspicious signals detected"
        )

        components.html(f"""
        <style>
          body {{ margin:0; padding:0; background:transparent; font-family:'Inter',sans-serif; }}
        </style>
        <div style="background:#0f172a;border:1px solid #1e3a5f;
                    border-radius:12px;padding:1.25rem 1.5rem;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
                        letter-spacing:2px;text-transform:uppercase;color:#3b82f6;
                        border-bottom:1px solid #1e3a5f;padding-bottom:0.5rem;
                        margin-bottom:1rem;">🔬 Lexical Feature Analysis</div>
            {rows_html}
            <div style="margin-top:0.8rem;padding:0.6rem 0.8rem;
                        background:{summary_bg};border:1px solid {summary_bdr};
                        border-radius:6px;font-size:0.76rem;color:{summary_clr};
                        font-family:'JetBrains Mono',monospace;">
                {summary_txt}
            </div>
        </div>
        """, height=420)

    # ── PANEL 3: Attack Pattern Explanations ─────────────────────────────────
    with p3:
        tips_html = ""

        if triggered:
            for key in triggered:
                if key in ATTACK_TIPS:
                    atk_name, atk_body = ATTACK_TIPS[key]
                    tips_html += f"""
                    <div style="background:#0c1829;border:1px solid #1e3a5f;
                                border-left:3px solid #f59e0b;border-radius:8px;
                                padding:0.75rem 1rem;margin:0.5rem 0;">
                        <div style="color:#fbbf24;font-weight:700;
                                    font-size:0.82rem;margin-bottom:0.25rem;">
                            ⚡ {atk_name}</div>
                        <div style="color:#94a3b8;font-size:0.78rem;
                                    line-height:1.5;">{atk_body}</div>
                    </div>"""
        else:
            tips_html = """
            <div style="background:#0a1a0f;border:1px solid #14532d;
                        border-left:3px solid #22c55e;border-radius:8px;
                        padding:0.75rem 1rem;margin:0.5rem 0;">
                <div style="color:#4ade80;font-weight:700;font-size:0.82rem;">
                    ✅ No attack patterns detected</div>
                <div style="color:#94a3b8;font-size:0.78rem;line-height:1.5;">
                    All lexical features are within normal ranges.
                    No known URL manipulation techniques were identified.
                </div>
            </div>"""

        bert_note_clr = "#fbbf24" if bert_score > 0.5 else "#4ade80"
        bert_note_bg  = "#1a1200" if bert_score > 0.5 else "#0a1a0f"
        bert_note_bdr = "#78350f" if bert_score > 0.5 else "#14532d"
        bert_note_txt = (
            "BERT's semantic analysis flagged this URL's token patterns "
            "as matching known phishing page structures."
            if bert_score > 0.5 else
            "BERT's semantic analysis found no suspicious token patterns "
            "matching known phishing structures."
        )

        components.html(f"""
        <style>
          body {{ margin:0; padding:0; background:transparent; font-family:'Inter',sans-serif; }}
        </style>
        <div style="background:#0f172a;border:1px solid #1e3a5f;
                    border-radius:12px;padding:1.25rem 1.5rem;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
                        letter-spacing:2px;text-transform:uppercase;color:#3b82f6;
                        border-bottom:1px solid #1e3a5f;padding-bottom:0.5rem;
                        margin-bottom:1rem;">💡 Attack Pattern Explanations</div>
            {tips_html}
            <div style="margin-top:0.8rem;background:{bert_note_bg};
                        border:1px solid {bert_note_bdr};
                        border-left:3px solid {bert_note_clr};
                        border-radius:8px;padding:0.75rem 1rem;">
                <div style="color:{bert_note_clr};font-weight:700;
                            font-size:0.82rem;margin-bottom:0.25rem;">
                    🧠 BERT Semantic Score: {bert_score:.1%}</div>
                <div style="color:#94a3b8;font-size:0.78rem;line-height:1.5;">
                    {bert_note_txt}</div>
            </div>
        </div>
        """, height=420)

    # ─────────────────────────────────────────────────────────────────────────
    # DIAGNOSTICS (collapsed)
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("")
    with st.expander("🔧 Engineering Diagnostic Logs"):
        st.code(f"""
URL             : {url_clean}
Engine          : {model_choice}
BERT Score      : {bert_score:.6f}
RF Score        : {rf_score:.6f}
Primary Score   : {threat_score:.6f}
Verdict         : {"PHISHING" if is_phish else "SAFE"}
Inference Time  : {elapsed:.1f}ms
Device          : CPU
Triggered       : {[FEATURE_META[k]['label'] for k in triggered]}
        """, language="yaml")
        st.write(f"**Lexical features:** {lex}")
        st.write(f"**BERT tokens:** {tokenizer.tokenize(url_clean)}")
        st.write(f"**CLS embedding:** mean={cls_emb.mean():.4f}  std={cls_emb.std():.4f}")

elif not scan:
    st.markdown("""
    <div style="text-align:center; padding:4rem 2rem;
                border:1px dashed #1e3a5f; border-radius:12px; margin-top:1rem;
                background:#080c14;">
        <div style="font-size:3.5rem; margin-bottom:1rem;">🛡️</div>
        <div style="font-size:1.1rem; color:#64748b; font-weight:600;">
            Enter a URL above and click Scan
        </div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.7rem;
                    color:#334155; margin-top:0.5rem; letter-spacing:1px;">
            BERT · PCA · RANDOM FOREST · EXPLAINABLE AI
        </div>
    </div>
    """, unsafe_allow_html=True)