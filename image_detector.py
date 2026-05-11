# ============================================================
# AI Image Detection Module
# Models: dima806 (primary) + umm-maybe (secondary) + NYUAD (fallback)
# Physics: FFT frequency analysis + Noise analysis
# ============================================================
#
# SETUP:
#   pip install torch torchvision transformers pillow numpy requests
#   pip install beautifulsoup4 opencv-python-headless scikit-learn
#
# USAGE:
#   from image_detector import predict_image, evaluate_dataset
# ============================================================

import os
import numpy as np
import torch
import requests
import cv2

from PIL import Image
from io import BytesIO
from transformers import (
    AutoModelForImageClassification,
    ViTImageProcessor,
    pipeline
)

# ============================================================
# MODEL LOADING
# ============================================================


print("Loading image detection models...")

# ── Model 1: dima806 — primary, strong on general AI images ──
try:
    dima_pipe      = pipeline("image-classification", model="dima806/ai_vs_real_image_detection", device=0 if torch.cuda.is_available() else -1)
    DIMA_AVAILABLE = True
    print("✓ dima806 loaded")
except Exception as e:
    print(f"✗ dima806 not available: {e}")
    DIMA_AVAILABLE = False

# ── Model 2: umm-maybe — strong on Midjourney/SDXL ───────────
try:
    umm_pipe      = pipeline("image-classification", model="umm-maybe/AI-image-detector", device=0 if torch.cuda.is_available() else -1)
    UMM_AVAILABLE = True
    print("✓ umm-maybe loaded")
except Exception as e:
    print(f"✗ umm-maybe not available: {e}")
    UMM_AVAILABLE = False

# ── Model 3: NYUAD — fallback, trained on DALL-E + SD ────────
try:
    nyuad_processor = ViTImageProcessor.from_pretrained("./nyuad_model")
    nyuad_model     = AutoModelForImageClassification.from_pretrained("./nyuad_model", trust_remote_code=True)
    nyuad_model.eval()
    NYUAD_AVAILABLE = True
    print("✓ NYUAD loaded")
except Exception as e:
    print(f"✗ NYUAD not available: {e}")
    NYUAD_AVAILABLE = False

print("Models ready.\n")


# ============================================================
# INDIVIDUAL MODEL PREDICTORS
# ============================================================

def predict_dima(image: Image.Image) -> dict | None:
    """
    dima806 — primary model.
    Best for: general AI images, news photos, portraits.
    """
    if not DIMA_AVAILABLE:
        return None
    try:
        results  = dima_pipe(image.convert("RGB"))
        ai_score = next(
            (r["score"] for r in results if r["label"].upper() in ["FAKE", "AI", "ARTIFICIAL"]),
            None
        )
        if ai_score is None:
            real_score = next((r["score"] for r in results if r["label"].upper() in ["REAL", "HUMAN"]), 0.5)
            ai_score   = 1 - real_score
        return {
            "model":    "dima806",
            "label":    "AI-generated" if ai_score >= 0.5 else "Real",
            "ai_score": round(float(ai_score), 4)
        }
    except Exception as e:
        print(f"dima806 error: {e}")
        return None


def predict_umm(image: Image.Image) -> dict | None:
    """
    umm-maybe — secondary model.
    Best for: Midjourney, SDXL, newer diffusion models.
    """
    if not UMM_AVAILABLE:
        return None
    try:
        results  = umm_pipe(image.convert("RGB"))
        ai_score = next(
            (r["score"] for r in results if r["label"].upper() in ["FAKE", "AI", "ARTIFICIAL", "GENERATED"]),
            None
        )
        if ai_score is None:
            real_score = next((r["score"] for r in results if r["label"].upper() in ["REAL", "HUMAN"]), 0.5)
            ai_score   = 1 - real_score
        return {
            "model":    "umm-maybe",
            "label":    "AI-generated" if ai_score >= 0.5 else "Real",
            "ai_score": round(float(ai_score), 4)
        }
    except Exception as e:
        print(f"umm-maybe error: {e}")
        return None


def predict_nyuad(image: Image.Image) -> dict | None:
    """
    NYUAD ViT — fallback model.
    Best for: DALL-E, Stable Diffusion 1.x/2.x.
    """
    if not NYUAD_AVAILABLE:
        return None
    try:
        image  = image.convert("RGB")
        inputs = nyuad_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = nyuad_model(**inputs)
        probs      = torch.softmax(outputs.logits, dim=-1).squeeze().tolist()
        scores     = {nyuad_model.config.id2label[i]: round(p, 4) for i, p in enumerate(probs)}
        prediction = max(scores, key=scores.get)
        ai_score   = round(1 - scores.get("real", 0), 4)
        return {
            "model":    "NYUAD",
            "label":    "AI-generated" if prediction != "real" else "Real",
            "ai_score": ai_score,
            "scores":   scores
        }
    except Exception as e:
        print(f"NYUAD error: {e}")
        return None


# ============================================================
# PHYSICS-BASED ANALYSIS
# ============================================================

def fft_analysis(image: Image.Image) -> dict | None:
    """
    FFT Frequency Analysis.

    Real photographs have a natural frequency falloff due to lens optics
    and sensor physics — high frequencies decay smoothly.

    AI images break this pattern:
    - Diffusion models produce unnatural high-frequency peaks
    - GAN images have characteristic checkerboard artifacts in frequency domain
    - Both tend to be unnaturally smooth in mid-frequencies

    This is generator-agnostic — works on any AI model because it
    exploits the physics of real cameras, not model-specific artifacts.
    """
    try:
        gray      = np.array(image.convert("L"), dtype=np.float32)
        fft       = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.log(np.abs(fft_shift) + 1)

        h, w = magnitude.shape

        # Central peak ratio — real photos have stronger center dominance
        center_val  = magnitude[h//2, w//2]
        mean_mag    = magnitude.mean()
        center_ratio = float(center_val / (mean_mag + 1e-8))

        # High frequency corners — AI images leak more energy into corners
        corners = np.concatenate([
            magnitude[:h//8,  :w//8 ].flatten(),
            magnitude[:h//8,  -w//8:].flatten(),
            magnitude[-h//8:, :w//8 ].flatten(),
            magnitude[-h//8:, -w//8:].flatten()
        ])
        hf_ratio = float(corners.mean() / (mean_mag + 1e-8))

        # Mid-frequency uniformity — AI images are too smooth here
        mid_ring  = magnitude[h//4:3*h//4, w//4:3*w//4]
        mid_std   = float(mid_ring.std() / (magnitude.std() + 1e-8))

        # Radial frequency falloff — real images follow power law decay
        # AI images deviate from this natural falloff
        cy, cx    = h // 2, w // 2
        y_idx, x_idx = np.ogrid[:h, :w]
        radius    = np.sqrt((y_idx - cy)**2 + (x_idx - cx)**2).astype(int)
        max_r     = min(cy, cx)
        radial_profile = np.array([magnitude[radius == r].mean() for r in range(1, max_r)])
        # Real images: profile decays monotonically
        # AI images: profile has bumps and inconsistencies
        diffs     = np.diff(radial_profile)
        non_monotonic = float((diffs > 0).mean())  # fraction of increasing steps

        # Combine signals into AI score
        # Higher center_ratio → more real
        # Higher hf_ratio     → more AI
        # Lower mid_std       → more AI (too smooth)
        # Higher non_monotonic → more AI (unnatural falloff)
        center_score    = min(max(1 - (center_ratio - 3) / 10, 0), 1)
        hf_score        = min(max(hf_ratio / 0.8, 0), 1)
        smoothness_score = min(max(1 - mid_std, 0), 1)
        falloff_score   = min(max(non_monotonic * 2, 0), 1)

        ai_score = round(
            0.25 * center_score +
            0.30 * hf_score +
            0.25 * smoothness_score +
            0.20 * falloff_score,
            4
        )

        return {
            "model":          "FFT Analysis",
            "label":          "AI-generated" if ai_score >= 0.5 else "Real",
            "ai_score":       ai_score,
            "center_ratio":   round(center_ratio, 3),
            "hf_ratio":       round(hf_ratio, 3),
            "mid_std":        round(mid_std, 3),
            "non_monotonic":  round(non_monotonic, 3)
        }
    except Exception as e:
        print(f"FFT error: {e}")
        return None


def noise_analysis(image: Image.Image) -> dict | None:
    """
    Sensor Noise Analysis — NEW, replaces EXIF.

    Real camera sensors produce characteristic random noise patterns
    (photon shot noise + read noise). This noise follows specific
    statistical distributions and is spatially random.

    AI generated images are mathematically smooth — they lack this
    natural noise signature entirely, or have unnatural periodic noise
    from the generation process.

    This is more reliable than EXIF because:
    - EXIF is stripped by social media platforms
    - Noise is physically embedded in the pixel values
    - Cannot be removed without degrading the image
    """
    try:
        img_array = np.array(image.convert("RGB"), dtype=np.float32)

        # Extract noise by subtracting a smoothed version
        smoothed  = cv2.GaussianBlur(img_array, (5, 5), 0)
        noise     = img_array - smoothed

        # Real camera noise properties
        noise_std  = float(noise.std())
        noise_mean = float(np.abs(noise).mean())

        # Noise should be spatially random — check autocorrelation
        noise_gray = noise.mean(axis=2)
        autocorr   = np.corrcoef(noise_gray[:-1].flatten(), noise_gray[1:].flatten())[0, 1]
        autocorr   = float(autocorr) if not np.isnan(autocorr) else 0.0

        # Real images: noise_std typically 3-15, autocorr near 0
        # AI images: noise_std typically <2 (too smooth) or >20 (unnatural)
        # AI images: autocorr often higher (periodic noise patterns)

        # Too smooth → likely AI
        smoothness_ai = min(max(1 - (noise_std / 8), 0), 1)

        # High autocorrelation → likely AI (periodic patterns)
        autocorr_ai = min(max(abs(autocorr) * 2, 0), 1)

        # Noise uniformity across channels — real cameras have channel-specific noise
        channel_stds  = [noise[:,:,c].std() for c in range(3)]
        channel_var   = float(np.std(channel_stds) / (np.mean(channel_stds) + 1e-8))
        uniformity_ai = min(max(1 - channel_var * 3, 0), 1)  # too uniform → AI

        ai_score = round(
            0.40 * smoothness_ai +
            0.35 * autocorr_ai +
            0.25 * uniformity_ai,
            4
        )

        return {
            "model":       "Noise Analysis",
            "label":       "AI-generated" if ai_score >= 0.5 else "Real",
            "ai_score":    ai_score,
            "noise_std":   round(noise_std, 3),
            "autocorr":    round(autocorr, 3),
            "channel_var": round(channel_var, 3)
        }
    except Exception as e:
        print(f"Noise analysis error: {e}")
        return None


# ============================================================
# ENSEMBLE COMBINER
# ============================================================

def predict_image_combined(image: Image.Image) -> dict:
    """
    Principled ensemble detection strategy:

    1. Run all available deep learning models
    2. Run physics-based analysis (FFT + Noise)
    3. Combine with confidence-weighted voting:
       - Deep learning models: 70% total weight
       - Physics analysis: 30% total weight
    4. If all models agree → high confidence
       If models disagree → flag as uncertain

    Confidence disclaimer added for uncertain predictions —
    honest uncertainty is better than wrong certainty.
    """
    results = {}

    # ── Deep Learning Models ─────────────────────────────────
    dima_result  = predict_dima(image)
    umm_result   = predict_umm(image)
    nyuad_result = predict_nyuad(image)

    # ── Physics Analysis ──────────────────────────────────────
    fft_result   = fft_analysis(image)
    noise_result = noise_analysis(image)

    # ── Collect available scores ──────────────────────────────
    dl_scores     = []
    physics_scores = []

    if dima_result:
        dl_scores.append(dima_result["ai_score"])
        results["dima806"] = dima_result

    if umm_result:
        dl_scores.append(umm_result["ai_score"])
        results["umm_maybe"] = umm_result

    if nyuad_result and not (dima_result or umm_result):
        # Only use NYUAD if neither primary model available
        dl_scores.append(nyuad_result["ai_score"])
        results["nyuad"] = nyuad_result

    if fft_result:
        physics_scores.append(fft_result["ai_score"])
        results["fft"] = fft_result

    if noise_result:
        physics_scores.append(noise_result["ai_score"])
        results["noise"] = noise_result

    # ── Handle no models available ────────────────────────────
    if not dl_scores and not physics_scores:
        return {
            "label":      "Unknown",
            "confidence": 0.0,
            "ai_score":   0.5,
            "warning":    "No models available",
            "breakdown":  results
        }

    # ── Weighted combination ──────────────────────────────────
    scores  = []
    weights = []

    if dl_scores:
        dl_avg = sum(dl_scores) / len(dl_scores)
        scores.append(dl_avg)
        weights.append(0.70)

    if physics_scores:
        phys_avg = sum(physics_scores) / len(physics_scores)
        scores.append(phys_avg)
        weights.append(0.30)

    total_weight = sum(weights)
    final_score  = round(sum(s * w / total_weight for s, w in zip(scores, weights)), 4)

    # ── Agreement check ───────────────────────────────────────
    all_scores    = dl_scores + physics_scores
    all_labels    = [1 if s >= 0.5 else 0 for s in all_scores]
    agreement     = sum(all_labels) / len(all_labels) if all_labels else 0.5
    models_agree  = agreement >= 0.75 or agreement <= 0.25

    # ── Confidence calculation ────────────────────────────────
    raw_confidence = final_score if final_score >= 0.5 else 1 - final_score
    # Penalize confidence when models disagree
    adjusted_confidence = raw_confidence * (0.7 + 0.3 * (1 if models_agree else 0))

    # ── Warning for uncertain predictions ────────────────────
    warning = None
    if not models_agree:
        warning = "Models disagree — result may be unreliable. Newer AI generators (Midjourney v6, DALL-E 3, Flux) are harder to detect."
    elif adjusted_confidence < 0.65:
        warning = "Low confidence prediction. Treat this result with caution."

    return {
        "label":      "AI-generated" if final_score >= 0.5 else "Real",
        "confidence": round(float(adjusted_confidence), 4),
        "ai_score":   final_score,
        "models_used": list(results.keys()),
        "models_agree": models_agree,
        "warning":    warning,
        "breakdown":  results
    }


# ============================================================
# EVALUATION — run on a folder of labeled images
# ============================================================

def evaluate_dataset(real_folder: str, ai_folder: str, max_images: int = 50) -> dict:
    """
    Evaluate the ensemble on a local dataset.

    Args:
        real_folder: path to folder of real images
        ai_folder:   path to folder of AI generated images
        max_images:  max images per class to evaluate

    Returns:
        dict with accuracy, precision, recall, F1, per-model breakdown
    """
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    import json

    print(f"\nEvaluating on dataset...")
    print(f"Real folder : {real_folder}")
    print(f"AI folder   : {ai_folder}")

    def load_images(folder, label, max_n):
        items = []
        exts  = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        for fname in os.listdir(folder)[:max_n]:
            if os.path.splitext(fname)[1].lower() in exts:
                try:
                    img = Image.open(os.path.join(folder, fname)).convert("RGB")
                    items.append((img, label, fname))
                except Exception:
                    continue
        return items

    real_images = load_images(real_folder, 0, max_images)
    ai_images   = load_images(ai_folder,   1, max_images)
    all_images  = real_images + ai_images

    print(f"Real images : {len(real_images)}")
    print(f"AI images   : {len(ai_images)}")
    print(f"Total       : {len(all_images)}\n")

    y_true, y_pred, y_scores = [], [], []
    per_model_preds = {
        "dima806": [], "umm_maybe": [], "nyuad": [],
        "fft": [], "noise": []
    }
    errors = []

    for i, (img, label, fname) in enumerate(all_images):
        result = predict_image_combined(img)
        pred   = 1 if result["label"] == "AI-generated" else 0

        y_true.append(label)
        y_pred.append(pred)
        y_scores.append(result["ai_score"])

        # Per model predictions
        for model_key in per_model_preds:
            if model_key in result["breakdown"] and result["breakdown"][model_key]:
                score = result["breakdown"][model_key]["ai_score"]
                per_model_preds[model_key].append((label, 1 if score >= 0.5 else 0, score))

        if pred != label:
            errors.append({
                "file":      fname,
                "actual":    "AI" if label == 1 else "Real",
                "predicted": result["label"],
                "score":     result["ai_score"],
                "warning":   result.get("warning")
            })

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(all_images)}...")

    # ── Overall metrics ───────────────────────────────────────
    report = classification_report(y_true, y_pred, target_names=["Real", "AI"], output_dict=True)
    cm     = confusion_matrix(y_true, y_pred)

    try:
        auc = roc_auc_score(y_true, y_scores)
    except Exception:
        auc = None

    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=["Real", "AI"]))
    print(f"Confusion Matrix:\n{cm}")
    if auc:
        print(f"ROC-AUC: {auc:.4f}")

    # ── Per model breakdown ───────────────────────────────────
    print("\nPer-model breakdown:")
    for model_name, preds in per_model_preds.items():
        if preds:
            mt, mp, _ = zip(*preds)
            acc = sum(t == p for t, p in zip(mt, mp)) / len(mt)
            print(f"  {model_name:<15} accuracy: {acc*100:.1f}% ({len(preds)} images)")

    # ── Error analysis ────────────────────────────────────────
    print(f"\nErrors ({len(errors)} total):")
    for e in errors[:10]:
        print(f"  [{e['actual']} → {e['predicted']}] {e['file']} (score={e['score']})")
        if e["warning"]:
            print(f"    ⚠ {e['warning']}")

    return {
        "accuracy":    report["accuracy"],
        "f1":          report["weighted avg"]["f1-score"],
        "precision":   report["weighted avg"]["precision"],
        "recall":      report["weighted avg"]["recall"],
        "auc":         auc,
        "confusion_matrix": cm.tolist(),
        "errors":      errors,
        "per_model":   {k: len(v) for k, v in per_model_preds.items() if v}
    }


# ============================================================
# UTILITY — load image from URL
# ============================================================

def load_image_from_url(url: str) -> Image.Image:
    headers = {"User-Agent": "Mozilla/5.0"}
    resp    = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":
    print("Image detector ready.")
    print("\nAvailable models:")
    print(f"  dima806  : {'✓' if DIMA_AVAILABLE  else '✗'}")
    print(f"  umm-maybe: {'✓' if UMM_AVAILABLE   else '✗'}")
    print(f"  NYUAD    : {'✓' if NYUAD_AVAILABLE else '✗'}")
    print(f"  FFT      : ✓ (always available)")
    print(f"  Noise    : ✓ (always available)")

    print("\nTo evaluate on your own images:")
    print("  from image_detector import evaluate_dataset")
    print("  evaluate_dataset('path/to/real/', 'path/to/ai/', max_images=50)")