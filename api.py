"""
PhishNet — FastAPI Backend
Serves phishing detection as a REST API.
Downloads BERT from HuggingFace on first startup.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
import numpy as np
import pandas as pd
import re
import os
import time
from urllib.parse import urlparse

app = FastAPI(title="PhishNet API", version="1.0.0")

# ── CORS — allow Chrome extension to call this API ────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # extension can call from any origin
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── DEVICE ────────────────────────────────────────────────────────────────────
device = torch.device("cpu")

# ── MODEL LOADING ─────────────────────────────────────────────────────────────
# BERT is downloaded from HuggingFace Hub at startup if not cached locally.
# .joblib files are loaded from disk (included in the repo).

HF_MODEL_ID  = os.getenv("HF_MODEL_ID", "YOUR_HF_USERNAME/phishnet-bert")
MODEL_CACHE  = "./bert_phishing_5k_benchmark"

print(f"Loading BERT from: {HF_MODEL_ID if not os.path.isdir(MODEL_CACHE) else MODEL_CACHE}")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_CACHE if os.path.isdir(MODEL_CACHE) else HF_MODEL_ID
)
dl_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CACHE if os.path.isdir(MODEL_CACHE) else HF_MODEL_ID
).to(device)
dl_model.eval()

rf_model  = joblib.load("random_forest_model.joblib")
pca       = joblib.load("pca_compressor.joblib")

print("✅ All models loaded")

# ── ATTACK TIPS ───────────────────────────────────────────────────────────────
ATTACK_TIPS = {
    "url_length"         : ("Long URL Obfuscation",   "Attackers pad URLs to hide the real domain."),
    "count_at"           : ("@ Symbol Trick",         "Browsers ignore everything before @ in a URL."),
    "count_hyphen"       : ("Hyphen Stuffing",        "Hyphens make fake domains look legitimate."),
    "count_double_slash" : ("Double-Slash Redirect",  "Extra slashes bypass simple URL filters."),
    "count_percent"      : ("Percent Encoding",       "Encoding hides the true URL from scanners."),
    "count_digits"       : ("Digit Substitution",     "Replacing letters with digits — paypa1 = paypal."),
    "count_dots"         : ("Subdomain Abuse",        "Extra dots create deep subdomains to fool users."),
    "digit_letter_ratio" : ("High Digit Density",     "Legitimate domains rarely have many digits."),
    "has_ip"             : ("Raw IP Address",         "Legitimate sites use domain names, not raw IPs."),
    "is_shortened"       : ("URL Shortener",          "Shorteners hide the real malicious destination."),
}

FEATURE_META = {
    "url_length"         : {"label": "URL Length",         "threshold": 75,   "high_is_bad": True},
    "count_at"           : {"label": "@ Symbol",           "threshold": 0,    "high_is_bad": True},
    "count_hyphen"       : {"label": "Hyphens",            "threshold": 3,    "high_is_bad": True},
    "count_double_slash" : {"label": "Double Slashes",     "threshold": 0,    "high_is_bad": True},
    "count_percent"      : {"label": "% Encoding",         "threshold": 1,    "high_is_bad": True},
    "count_digits"       : {"label": "Digit Count",        "threshold": 8,    "high_is_bad": True},
    "count_dots"         : {"label": "Dot Count",          "threshold": 4,    "high_is_bad": True},
    "digit_letter_ratio" : {"label": "Digit/Letter Ratio", "threshold": 0.15, "high_is_bad": True},
    "has_ip"             : {"label": "Raw IP Address",     "threshold": 0,    "high_is_bad": True},
    "is_shortened"       : {"label": "URL Shortener",      "threshold": 0,    "high_is_bad": True},
}

# ── LEXICAL EXTRACTOR ─────────────────────────────────────────────────────────
def extract_url_math(url: str) -> dict:
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
        "has_ip"             : 1 if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", domain) else 0,
        "is_shortened"       : 1 if any(s in domain for s in [
                                   "bit.ly", "qrco.de", "t.co", "tinyurl.com",
                                   "l.ead.me", "goo.gl", "ow.ly"]) else 0,
    }

# ── REQUEST / RESPONSE MODELS ─────────────────────────────────────────────────
class ScanRequest(BaseModel):
    url: str

class FeatureResult(BaseModel):
    label:      str
    value:      float
    suspicious: bool
    tip_title:  str
    tip_body:   str

class ScanResponse(BaseModel):
    url:              str
    verdict:          str        # "phishing" | "safe"
    threat_score:     float      # 0.0 – 1.0
    bert_score:       float
    rf_score:         float
    models_agree:     bool
    triggered_count:  int
    features:         list[FeatureResult]
    inference_ms:     float

# ── ENDPOINTS ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "PhishNet API running", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/scan", response_model=ScanResponse)
def scan(req: ScanRequest):
    t0  = time.time()
    url = req.url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    # ── Tokenise ──────────────────────────────────────────────────────────────
    inputs = tokenizer(
        url, return_tensors="pt",
        truncation=True, padding=True, max_length=128
    ).to(device)

    # ── BERT inference ────────────────────────────────────────────────────────
    with torch.no_grad():
        outputs = dl_model(**inputs, output_hidden_states=True)

    bert_probs  = F.softmax(outputs.logits, dim=1).squeeze().tolist()
    bert_score  = float(bert_probs[1])

    # ── Hybrid: CLS → PCA → RF ────────────────────────────────────────────────
    cls_emb      = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
    pca_feats    = pca.transform(cls_emb)
    lex          = extract_url_math(url)
    feat_dict    = {f"pca_feature_{i}": pca_feats[0][i] for i in range(20)}
    feat_dict.update(lex)
    df           = pd.DataFrame([feat_dict])
    df_aligned   = df[rf_model.feature_names_in_]
    rf_proba     = rf_model.predict_proba(df_aligned)[0]
    rf_score     = float(rf_proba[1])

    # ── Primary verdict: average of both ─────────────────────────────────────
    threat_score  = (bert_score + rf_score) / 2
    verdict       = "phishing" if threat_score > 0.5 else "safe"
    models_agree  = (bert_score > 0.5) == (rf_score > 0.5)
    bert_triggered = bert_score > 0.7

    # ── Feature breakdown ─────────────────────────────────────────────────────
    features = []
    triggered = 0
    for key, meta in FEATURE_META.items():
        val    = lex[key]
        is_sus = val > meta["threshold"] if meta["high_is_bad"] else val < meta["threshold"]
        if is_sus:
            triggered += 1
        tip_title, tip_body = ATTACK_TIPS.get(key, ("", ""))
        features.append(FeatureResult(
            label      = meta["label"],
            value      = float(val),
            suspicious = is_sus,
            tip_title  = tip_title if is_sus else "",
            tip_body   = tip_body  if is_sus else "",
        ))

    # If no lexical features fired but BERT is highly confident,
    # add a semantic trigger so the extension shows an explanation
    if triggered == 0 and bert_triggered:
        features.append(FeatureResult(
            label      = "BERT Semantic Pattern",
            value      = float(bert_score),
            suspicious = True,
            tip_title  = "Semantic Phishing Pattern",
            tip_body   = "BERT detected token patterns in this URL that "
                         "strongly match known phishing page structures, "
                         "even though individual lexical features appear normal.",
        ))
        triggered = 1

    elapsed = (time.time() - t0) * 1000

    return ScanResponse(
        url             = url,
        verdict         = verdict,
        threat_score    = round(threat_score, 4),
        bert_score      = round(bert_score,   4),
        rf_score        = round(rf_score,     4),
        models_agree    = models_agree,
        triggered_count = triggered,
        features        = features,
        inference_ms    = round(elapsed, 1),
    )