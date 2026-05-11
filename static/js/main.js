// ============================================================
// TruthLens — main.js
// Content Reliability Scoring System
// ============================================================

// ── Tab switching ─────────────────────────────────────────
function switchTab(name, btn) {
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('panel-' + name).classList.add('active');
  btn.classList.add('active');
}

// ── Upload & preview ──────────────────────────────────────
const zone = document.getElementById('upload-zone');
zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
zone.addEventListener('drop', e => {
  e.preventDefault(); zone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f) { document.getElementById('image-input').files = e.dataTransfer.files; showPreview(f); }
});
function previewImage(input) { if (input.files?.[0]) showPreview(input.files[0]); }
function showPreview(file) {
  const r = new FileReader();
  r.onload = e => {
    document.getElementById('preview-img').src = e.target.result;
    document.getElementById('image-preview').style.display = 'block';
  };
  r.readAsDataURL(file);
}

// ── Reliability scoring ───────────────────────────────────
function reliabilityScore(aiScore, confidence, modelsAgree) {
  let base = (1 - aiScore) * 100;
  if (confidence < 0.70) base *= 0.85;
  if (!modelsAgree)      base *= 0.90;
  return Math.round(Math.max(0, Math.min(100, base)));
}

function scoreTier(score) {
  if (score >= 75) return { cls: 'high',   label: 'High Reliability',      color: '#027a48' };
  if (score >= 50) return { cls: 'medium', label: 'Moderate Reliability',  color: '#b54708' };
  if (score >= 25) return { cls: 'low',    label: 'Low Reliability',       color: '#c4320a' };
  return             { cls: 'vlow',  label: 'Very Low Reliability', color: '#d92d20' };
}

function interpretation(score, aiScore, modelsAgree, detailsObj) {
  const pct = Math.round(aiScore * 100);
  let text = '';
  if (score >= 75) {
    text = `Content shows strong indicators of human authorship. AI probability is low at ${pct}%.`;
  } else if (score >= 50) {
    text = `Content shows mixed signals. AI probability is ${pct}% — could be human-written with AI assistance, or lightly edited AI output.`;
  } else if (score >= 25) {
    text = `Content shows significant AI indicators at ${pct}% probability. Likely AI-generated or heavily AI-assisted.`;
  } else {
    text = `Content is highly likely to be AI-generated (${pct}% AI probability). Strong signals detected across multiple analysis methods.`;
  }
  if (!modelsAgree) {
    text += ' Note: classical ML models and transformer-based model disagree — the transformer model (GLYPH) is weighted higher as it covers modern AI systems.';
  }
  if (detailsObj?.glyph && detailsObj?.lr) {
    const glyphSaysAI = detailsObj.glyph.ai_prob >= 0.5;
    const lrSaysAI    = detailsObj.lr.ai_prob >= 0.5;
    if (glyphSaysAI !== lrSaysAI) {
      text += ` GLYPH (semantic analysis) says ${glyphSaysAI ? 'AI' : 'Human'} while LR (word patterns) says ${lrSaysAI ? 'AI' : 'Human'} — this divergence often indicates AI text that has been edited or paraphrased.`;
    }
  }
  return text;
}

function scoreRing(score, tier) {
  const r    = 32;
  const circ = 2 * Math.PI * r;
  const offset = circ - (score / 100) * circ;
  return `
    <div class="score-ring">
      <svg viewBox="0 0 80 80">
        <circle class="score-ring-bg"   cx="40" cy="40" r="${r}"/>
        <circle class="score-ring-fill ${tier.cls}" cx="40" cy="40" r="${r}"
          stroke-dasharray="${circ}" stroke-dashoffset="${offset}"/>
      </svg>
      <div class="score-number">${score}</div>
    </div>`;
}

function warnIcon() {
  return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`;
}

// ── PDF download button ───────────────────────────────────
function pdfButton(reportData) {
  const encoded = encodeURIComponent(JSON.stringify(reportData));
  return `
    <div class="pdf-section">
      <button class="btn-pdf" onclick='downloadPDF(${JSON.stringify(reportData)})'>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
          <polyline points="14 2 14 8 20 8"/>
          <line x1="12" y1="18" x2="12" y2="12"/>
          <polyline points="9 15 12 18 15 15"/>
        </svg>
        Download PDF Report
      </button>
    </div>`;
}

async function downloadPDF(data) {
  const btn = event.target.closest('.btn-pdf');
  const orig = btn.innerHTML;
  btn.innerHTML = `<svg class="spin-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg> Generating...`;
  btn.disabled = true;

  try {
    const resp = await fetch('/download-pdf', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });
    if (!resp.ok) throw new Error('PDF generation failed');
    const blob = await resp.blob();
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = 'truthlens_report.pdf';
    a.click();
    URL.revokeObjectURL(url);
  } catch (e) {
    alert('PDF generation failed: ' + e.message);
  } finally {
    btn.innerHTML = orig;
    btn.disabled  = false;
  }
}

// ── Render full reliability result ────────────────────────
function renderReliability(aiScore, confidence, modelsAgree, signals, warning, detailsObj, inputText, type) {
  const score  = reliabilityScore(aiScore, confidence, modelsAgree);
  const tier   = scoreTier(score);
  const interp = interpretation(score, aiScore, modelsAgree, detailsObj);
  const pct    = v => Math.round((v || 0) * 100);

  // Score header
  let html = `
    <div class="score-header ${tier.cls}">
      ${scoreRing(score, tier)}
      <div class="score-info">
        <div class="score-tier ${tier.cls}">${tier.label}</div>
        <div class="score-interpretation">Reliability Score: ${score}/100 &nbsp;·&nbsp; AI Probability: ${pct(aiScore)}%</div>
      </div>
    </div>`;

  // Probability bar
  html += `
    <div class="prob-section">
      <div class="prob-head"><span>More Human</span><span>More AI</span></div>
      <div class="prob-track"><div class="prob-fill" style="width:${pct(aiScore)}%"></div></div>
    </div>`;

  // Signal breakdown
  if (signals && Object.keys(signals).length) {
    const items = Object.entries(signals)
      .filter(([, v]) => v !== null && v !== undefined)
      .map(([key, val]) => {
        const sc   = val.ai_prob ?? val.ai_score ?? 0.5;
        const isAI = sc >= 0.5;
        const col  = isAI ? '#d92d20' : '#027a48';
        return `<div class="signal-card">
          <div class="sc-name">${key.replace(/_/g,' ')}</div>
          <div class="sc-score" style="color:${col}">${pct(sc)}%</div>
          <div class="sc-label" style="color:${col}">${isAI ? 'AI signal' : 'Human signal'}</div>
        </div>`;
      }).join('');
    if (items) html += `<div class="section-label">Contributing Signals</div><div class="signal-grid">${items}</div>`;
  }

  // Interpretation
  html += `<div class="section-label">Interpretation</div><div class="interp-box">${interp}</div>`;

  // Warning
  if (warning) html += `<div class="warning-box">${warnIcon()}${warning}</div>`;

  // ── PDF report data ───────────────────────────────────
  const reportData = {
    type:            type || 'Content',
    input_preview:   inputText ? inputText.substring(0, 200) + '...' : '',
    reliability_score: score,
    tier:            tier.label,
    ai_probability:  pct(aiScore) + '%',
    confidence:      pct(confidence) + '%',
    models_agree:    modelsAgree ? 'Yes' : 'No',
    interpretation:  interp,
    signals:         signals ? Object.fromEntries(
      Object.entries(signals)
        .filter(([,v]) => v)
        .map(([k, v]) => [k, pct(v.ai_prob ?? v.ai_score ?? 0.5) + '%'])
    ) : {},
    warning: warning || 'None'
  };

  html += pdfButton(reportData);
  return html;
}

// ── TEXT ──────────────────────────────────────────────────
async function analyzeText() {
  const text = document.getElementById('text-input').value.trim();
  const err  = document.getElementById('text-error');
  const load = document.getElementById('text-loader');
  const res  = document.getElementById('text-result');

  err.className = 'error-box'; res.className = 'result card';

  if (text.length < 50) {
    err.textContent = 'Please enter at least 50 characters for reliable scoring.';
    err.className = 'error-box active'; return;
  }

  load.className = 'loader active';
  try {
    const r = await fetch('/predict-text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });
    const d = await r.json();
    if (d.error) throw new Error(d.error);

    const details    = d.details || {};
    const aiScore    = details.final?.ai_score ?? (d.label === 'AI-generated' ? d.confidence : 1 - d.confidence);
    const confidence = d.confidence || 0.5;

    const signals = {};
    if (details.lr)    signals['LR (TF-IDF)']     = details.lr;
    if (details.sgd)   signals['SGD (TF-IDF)']    = details.sgd;
    if (details.glyph) signals['GLYPH (DeBERTa)'] = details.glyph;

    const labels      = Object.values(signals).map(s => s.ai_prob >= 0.5 ? 'ai' : 'human');
    const modelsAgree = new Set(labels).size <= 1;

    res.innerHTML = renderReliability(aiScore, confidence, modelsAgree, signals, d.warning, details, text, 'Text');
    res.className = 'result card active';
  } catch (e) {
    err.textContent = e.message || 'Analysis failed.';
    err.className = 'error-box active';
  } finally { load.className = 'loader'; }
}

// ── IMAGE ─────────────────────────────────────────────────
async function analyzeImage() {
  const input = document.getElementById('image-input');
  const err   = document.getElementById('image-error');
  const load  = document.getElementById('image-loader');
  const res   = document.getElementById('image-result');

  err.className = 'error-box'; res.className = 'result card';

  if (!input.files?.[0]) {
    err.textContent = 'Please upload an image first.';
    err.className = 'error-box active'; return;
  }

  load.className = 'loader active';
  const fd = new FormData(); fd.append('image', input.files[0]);

  try {
    const r = await fetch('/predict-image', { method: 'POST', body: fd });
    const d = await r.json();
    if (d.error) throw new Error(d.error);

    res.innerHTML = renderReliability(
      d.ai_score, d.confidence, d.models_agree,
      d.breakdown, d.warning, null,
      `Image: ${input.files[0].name}`, 'Image'
    );
    res.className = 'result card active';
  } catch (e) {
    err.textContent = e.message || 'Analysis failed.';
    err.className = 'error-box active';
  } finally { load.className = 'loader'; }
}

// ── URL ───────────────────────────────────────────────────
async function analyzeURL() {
  const url  = document.getElementById('url-input').value.trim();
  const err  = document.getElementById('url-error');
  const load = document.getElementById('url-loader');
  const res  = document.getElementById('url-result');

  err.className = 'error-box'; res.className = 'result card';

  if (!url) {
    err.textContent = 'Please enter a URL.';
    err.className = 'error-box active'; return;
  }

  load.className = 'loader active';
  try {
    const r = await fetch('/predict-url', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url })
    });
    const d = await r.json();
    if (d.error) throw new Error(d.error);

    let html = '';

    if (d.title || d.text_preview) {
      html += `<div class="article-meta">`;
      if (d.title)        html += `<div class="article-title">${d.title}</div>`;
      if (d.text_preview) html += `<div class="article-preview">${d.text_preview}</div>`;
      html += `</div>`;
    }

    html += `<div class="split-row">`;
    if (d.text_result) {
      const ts     = d.text_result.confidence ?? 0.5;
      const tAI    = d.text_result.label === 'AI-generated' ? ts : 1 - ts;
      const tScore = reliabilityScore(tAI, ts, true);
      const tTier  = scoreTier(tScore);
      html += `<div class="split-box">
        <div class="sb-label">Text Reliability</div>
        <div class="sb-score" style="color:${tTier.color}">${tScore}<span style="font-size:13px;font-weight:400">/100</span></div>
        <div class="sb-tier" style="color:${tTier.color}">${tTier.label}</div>
      </div>`;
    }
    if (d.image_result) {
      const is_    = d.image_result.confidence ?? 0.5;
      const iAI    = d.image_result.label === 'AI-generated' ? is_ : 1 - is_;
      const iScore = reliabilityScore(iAI, is_, true);
      const iTier  = scoreTier(iScore);
      html += `<div class="split-box">
        <div class="sb-label">Image Reliability (${d.images_checked} checked)</div>
        <div class="sb-score" style="color:${iTier.color}">${iScore}<span style="font-size:13px;font-weight:400">/100</span></div>
        <div class="sb-tier" style="color:${iTier.color}">${iTier.label}</div>
      </div>`;
    }
    html += `</div>`;

    html += `<div class="section-label">Overall Reliability Score</div>`;
    html += renderReliability(
      d.combined_score, d.combined_score, true,
      null, null, null, url, 'URL'
    );

    res.innerHTML = html;
    res.className = 'result card active';
  } catch (e) {
    err.textContent = e.message || 'Analysis failed.';
    err.className = 'error-box active';
  } finally { load.className = 'loader'; }
}

document.getElementById('url-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') analyzeURL();
});