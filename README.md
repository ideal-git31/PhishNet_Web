# PhishNet 🛡️

A hybrid phishing detection system combining Fine-Tuned BERT and 
PCA-Random Forest for explainable URL classification.

## Architecture
- **BERT** (`bert-base-uncased`) fine-tuned on 10K URLs — 96.45% accuracy
- **PCA** compresses 768-dim CLS embeddings → 20 components
- **Random Forest** trained on 30 features (20 PCA + 10 lexical)
- **Chrome Extension** with real-time explainability panel
- **FastAPI** backend serving predictions via REST API

## Key Features
- Explainable AI — shows WHY a URL is flagged (feature breakdown + attack patterns)
- Dual-branch confidence — BERT score vs RF score displayed separately
- 10 lexical features — URL length, digit ratio, IP detection, shortener detection, etc.
- Dataset bias fix — corrected bare domain vs full-path URL distribution

## Results
| Model | Accuracy | AUC-ROC |
|---|---|---|
| Fine-Tuned BERT | 96.45% | — |
| PCA + Random Forest | 96.30% | 0.9907 |
| PCA + LightGBM | 96.40% | 0.9912 |

## Project Structure

    PhishNet_Web/
    ├── app.py              # Streamlit UI
    ├── api.py              # FastAPI backend
    ├── extension/          # Chrome extension
    │   ├── manifest.json
    │   ├── popup.html
    │   └── popup.js
    ├── requirements.txt
    └── render.yaml         # Render deployment config
