// PhishNet Extension — popup.js
// Grabs current tab URL, calls PhishNet API, renders result

const API_BASE = "http://localhost:8000";

// Feature max values for bar normalisation
const FEAT_MAX = {
  "URL Length": 200, "@ Symbol": 3, "Hyphens": 10,
  "Double Slashes": 3, "% Encoding": 5, "Digit Count": 20,
  "Dot Count": 8, "Digit/Letter Ratio": 1.0,
  "Raw IP Address": 1, "URL Shortener": 1
};

// ── Helpers ──────────────────────────────────────────────────────────────────
function clr(score) {
  return score > 0.5 ? "#ef4444" : "#22c55e";
}

function pct(val, max) {
  return Math.min((val / max) * 100, 100).toFixed(1);
}

function fmtVal(label, value) {
  if (label === "Digit/Letter Ratio") return value.toFixed(2);
  if (label === "Raw IP Address" || label === "URL Shortener")
    return value === 1 ? "YES" : "NO";
  return String(Math.round(value));
}

// ── Render result ─────────────────────────────────────────────────────────────
function render(data) {
  const isPhish   = data.verdict === "phishing";
  const scoreClr  = clr(data.threat_score);
  const bertClr   = clr(data.bert_score);
  const rfClr     = clr(data.rf_score);
  const agreeClr  = data.models_agree ? "#4ade80" : "#f87171";
  const agreeBg   = data.models_agree ? "#0a1a0f"  : "#1a0a0a";
  const agreeBdr  = data.models_agree ? "#14532d"  : "#7f1d1d";
  const agreeTxt  = data.models_agree ? "✅ Models agree" : "⚡ Models disagree";

  // Feature rows HTML
  let featRows = "";
  for (const f of data.features) {
    const max    = FEAT_MAX[f.label] ?? 10;
    const norm   = pct(f.value, max);
    const fclr   = f.suspicious ? "#ef4444" : "#22c55e";
    const flag   = f.suspicious ? "⚠" : "✓";
    const dispV  = fmtVal(f.label, f.value);
    featRows += `
      <div class="feat-row">
        <div class="feat-flag" style="color:${fclr}">${flag}</div>
        <div class="feat-name">${f.label}</div>
        <div class="feat-track">
          <div class="feat-fill" style="width:${norm}%;background:${fclr};"></div>
        </div>
        <div class="feat-val">${dispV}</div>
      </div>`;
  }

  // Attack tips HTML (only triggered ones)
  let tipsHtml = "";
  const triggered = data.features.filter(f => f.suspicious && f.tip_title);
  if (triggered.length > 0) {
    for (const f of triggered) {
      tipsHtml += `
        <div class="tip-box">
          <div class="tip-name">⚡ ${f.tip_title}</div>
          <div class="tip-body">${f.tip_body}</div>
        </div>`;
    }
  } else {
    tipsHtml = `
      <div class="safe-msg">
        ✅ No known attack patterns detected in this URL structure.
      </div>`;
  }

  // BERT note
  const bertNoteBg  = data.bert_score > 0.5 ? "#1a1200" : "#051a0a";
  const bertNoteBdr = data.bert_score > 0.5 ? "#f59e0b" : "#22c55e";
  const bertNoteClr = data.bert_score > 0.5 ? "#fbbf24" : "#4ade80";
  const bertNoteTxt = data.bert_score > 0.5
    ? "BERT flagged token patterns matching phishing pages."
    : "BERT found no suspicious semantic patterns.";

  document.getElementById("content").innerHTML = `

    <!-- Verdict -->
    <div class="verdict ${isPhish ? 'verdict-phish' : 'verdict-safe'}">
      <div class="verdict-title">${isPhish ? "🚨 PHISHING DETECTED" : "✅ SAFE DOMAIN"}</div>
      <div class="verdict-score" style="color:${scoreClr}">${(data.threat_score * 100).toFixed(1)}%</div>
      <div class="verdict-sub">PHISHING PROBABILITY · BERT + RF AVERAGE</div>
      <div class="prog-wrap">
        <div class="prog-fill" style="width:${(data.threat_score*100).toFixed(1)}%;background:${scoreClr};"></div>
      </div>
    </div>

    <!-- Score breakdown -->
    <div class="scores">
      <div class="score-box">
        <div class="score-val" style="color:${bertClr}">${(data.bert_score*100).toFixed(1)}%</div>
        <div class="score-lbl">BERT Score</div>
      </div>
      <div class="score-box">
        <div class="score-val" style="color:${rfClr}">${(data.rf_score*100).toFixed(1)}%</div>
        <div class="score-lbl">RF Score</div>
      </div>
      <div class="score-box">
        <div class="agree-badge" style="color:${agreeClr};background:${agreeBg};border:1px solid ${agreeBdr};">
          ${agreeTxt}
        </div>
        <div class="score-lbl" style="margin-top:4px;">Consensus</div>
      </div>
    </div>

    <!-- Feature analysis -->
    <div class="section-title">🔬 Lexical Feature Analysis</div>
    <div class="feat-list">${featRows}</div>

    <!-- Attack patterns -->
    <div class="section-title">💡 Attack Patterns</div>
    <div class="tips">${tipsHtml}</div>

    <!-- BERT semantic note -->
    <div class="bert-note" style="background:${bertNoteBg};border-left:2px solid ${bertNoteBdr};">
      <span style="color:${bertNoteClr};font-weight:700;">🧠 BERT:</span>
      <span style="color:#94a3b8;"> ${bertNoteTxt}</span>
    </div>
  `;

  document.getElementById("inferenceTime").textContent =
    `${data.inference_ms.toFixed(0)}ms inference`;
}

// ── Error state ───────────────────────────────────────────────────────────────
function renderError(msg) {
  document.getElementById("content").innerHTML = `
    <div class="error-box">
      <div style="font-size:24px;margin-bottom:8px;">⚠️</div>
      <div>${msg}</div>
      <div style="margin-top:8px;font-size:10px;color:#475569;">
        Make sure the PhishNet API is running.
      </div>
    </div>`;
}

// ── Main ──────────────────────────────────────────────────────────────────────
chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
  const tab = tabs[0];
  const url = tab?.url ?? "";

  // Show URL
  document.getElementById("urlDisplay").textContent =
    url.length > 60 ? url.slice(0, 60) + "…" : url;

  // Skip chrome:// and extension pages
  if (!url.startsWith("http://") && !url.startsWith("https://")) {
    document.getElementById("content").innerHTML = `
      <div class="error-box" style="color:#64748b;">
        <div style="font-size:24px;margin-bottom:8px;">ℹ️</div>
        <div>Navigate to a website to scan it.</div>
      </div>`;
    return;
  }

  try {
    const res  = await fetch(`${API_BASE}/scan`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ url })
    });

    if (!res.ok) throw new Error(`API error: ${res.status}`);
    const data = await res.json();
    render(data);

  } catch (err) {
    renderError("Could not reach PhishNet API.<br>" + err.message);
  }
});