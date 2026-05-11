import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification



# ============================================================
# Load TF-IDF models (trained by us on 728K samples)
# ============================================================
vectorizer = joblib.load("text_vectorizer.pkl")
lr_model   = joblib.load("text_model_lr.pkl")
sgd_model  = joblib.load("text_model_sgd.pkl")

# ============================================================
# Load GLYPH — DeBERTa-v3 pretrained detector
# 98.85% accuracy, covers GPT-2 through GPT-4 (14 AI families)
# Used as benchmark comparison model
# ============================================================
GLYPH_MODEL = "ogmatrixllm/glyph-v1.1"
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading GLYPH model...")
try:
    # IMPORTANT: use_fast=False required — fast tokenizer has
    # a confirmed regression in transformers>=4.47 for DeBERTa-v3
    glyph_tokenizer = AutoTokenizer.from_pretrained(GLYPH_MODEL, use_fast=False)
    glyph_model     = AutoModelForSequenceClassification.from_pretrained(GLYPH_MODEL)
    glyph_model.to(device)
    glyph_model.eval()
    GLYPH_AVAILABLE = True
    print("✓ GLYPH loaded")
    print(glyph_model.config.id2label)
    print("LABEL_0 = HUMAN , LABEL_1= AI")
except Exception as e:
    print(f"✗ GLYPH not available: {e}")
    GLYPH_AVAILABLE = False


# ============================================================
# Individual predictors
# ============================================================

def predict_lr(vec) -> dict:
    proba   = lr_model.predict_proba(vec)[0]
    ai_prob = float(proba[1])           # index 1 = AI
    return {
        "label":      "AI-generated" if ai_prob >= 0.5 else "Human-written",
        "confidence": round(ai_prob, 4),
        "ai_prob":    round(ai_prob, 4)
    }


def predict_sgd(vec) -> dict:
    if hasattr(sgd_model, "predict_proba"):
        proba   = sgd_model.predict_proba(vec)[0]
        ai_prob = float(proba[1])
    else:
        df      = float(sgd_model.decision_function(vec)[0])
        ai_prob = 1 / (1 + np.exp(-df))
    return {
        "label":      "AI-generated" if ai_prob >= 0.5 else "Human-written",
        "confidence": round(ai_prob, 4),
        "ai_prob":    round(ai_prob, 4)
    }


def predict_glyph(text: str) -> dict | None:
    """
    GLYPH — DeBERTa-v3-base fine-tuned detector.
    98.85% accuracy across 14 AI model families (GPT-2 to GPT-4).
    LABEL_0 = human, LABEL_1 = AI-generated.
    """
    if not GLYPH_AVAILABLE:
        return None
    try:
        inputs = glyph_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512          # trained at 512, longer texts truncated
        ).to(device)

        with torch.no_grad():
            logits = glyph_model(**inputs).logits
            probs  = torch.softmax(logits, dim=-1)

        p_human, p_ai = probs[0].tolist()
        ai_prob       = round(float(p_ai), 4)
        label         = "AI-generated" if p_ai > 0.5 else "Human-written"

        return {
            "label":      label,
            "confidence": round(max(p_human, p_ai), 4),
            "ai_prob":    ai_prob
        }
    except Exception as e:
        print(f"GLYPH error: {e}")
        return None


# ============================================================
# Main predict function
# ============================================================

def predict_text(text: str) -> dict:
    vec = vectorizer.transform([text])

    # Run all models
    lr_result    = predict_lr(vec)
    sgd_result   = predict_sgd(vec)
    glyph_result = predict_glyph(text)

    # Weighted average
    # GLYPH gets higher weight — 98.85% accuracy vs ~91% for TF-IDF models
    scores  = [lr_result["ai_prob"], sgd_result["ai_prob"]]
    weights = [1.0, 1.0]

    if glyph_result:
        scores.append(glyph_result["ai_prob"])
        weights.append(2.0)     # GLYPH gets 2x weight — significantly better model

    avg_ai_score = round(
        sum(s * w for s, w in zip(scores, weights)) / sum(weights), 4
    )

    final_label = "AI-generated" if avg_ai_score >= 0.5 else "Human-written"
    confidence  = avg_ai_score if avg_ai_score >= 0.5 else 1 - avg_ai_score

    # Warning logic
    all_labels = [lr_result["label"], sgd_result["label"]]
    if glyph_result:
        all_labels.append(glyph_result["label"])

    warning = None
    if 0.4 <= avg_ai_score <= 0.6:
        warning = "Borderline prediction — low confidence. Consider this result as indicative only."
    elif len(set(all_labels)) > 1:
        warning = "Models disagree on this text. GLYPH result is weighted higher as it covers more AI model families."

    # Short text warning (GLYPH docs: <50 words has reduced F1)
    if len(text.split()) < 50:
        warning = (warning or "") + " Short text detected — accuracy may be reduced. Provide more text for reliable detection."
        warning = warning.strip()

    return {
        "lr":    lr_result,
        "sgd":   sgd_result,
        "glyph": glyph_result,    # None if unavailable
        "final": {
            "label":      final_label,
            "confidence": round(float(confidence), 4),
            "ai_score":   avg_ai_score
        },
        "warning": warning
    }

