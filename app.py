from flask import Flask, request, jsonify, render_template, send_file
from PIL import Image
import io
import requests

from text_detector import predict_text
from image_detector import predict_image_combined
from url_handler import scrape_url
app = Flask(__name__)

# =======================
# HOME
# =======================
@app.route("/")
def home():
    return render_template("index.html")
# =======================
# TEXT
# =======================
@app.route("/predict-text", methods=["POST"])
def predict_text_api():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text or len(text) < 50:
        return jsonify({"error": "Please provide at least 50 characters."}), 400

    try:
        result = predict_text(text)
        return jsonify({
            "label":      result["final"]["label"],
            "confidence": result["final"]["confidence"],
            "warning":    result.get("warning"),
            "details":    result
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
# =======================
# IMAGE
# =======================
@app.route("/predict-image", methods=["POST"])
def predict_image_api():
    if "image" not in request.files:
        return jsonify({"error": "No image provided."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        image  = Image.open(file.stream).convert("RGB")
        result = predict_image_combined(image)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =======================
# URL
# =======================
@app.route("/predict-url", methods=["POST"])
def predict_url_api():
    data = request.get_json()
    url  = data.get("url", "").strip()

    if not url:
        return jsonify({"error": "No URL provided."}), 400

    if not url.startswith("http"):
        url = "https://" + url

    # ── Scrape ────────────────────────────────────────────────
    try:
        scraped = scrape_url(url)
    except Exception as e:
        return jsonify({"error": f"Failed to scrape URL: {str(e)}"}), 400

    if not scraped.get("text"):
        return jsonify({"error": "Could not extract text from this URL."}), 400

    # ── Text analysis ─────────────────────────────────────────
    try:
        text_result = predict_text(scraped["text"])
        text_final  = text_result["final"]
        text_ai_score = (
            text_final["confidence"]
            if text_final["label"] == "AI-generated"
            else 1 - text_final["confidence"]
        )
    except Exception as e:
        text_final    = {"label": "Error", "confidence": 0.5}
        text_ai_score = 0.5

    # ── Image analysis (first 5 images) ──────────────────────
    image_results  = []
    images_checked = 0

    for img_url in scraped.get("images", [])[:5]:
        try:
            resp = requests.get(img_url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
            img  = Image.open(io.BytesIO(resp.content)).convert("RGB")
            r    = predict_image_combined(img)
            image_results.append(r)
            images_checked += 1
        except Exception:
            continue

    if image_results:
        avg_ai = sum(r["ai_score"] for r in image_results) / len(image_results)
        image_final = {
            "label":      "AI-generated" if avg_ai >= 0.5 else "Real",
            "confidence": round(float(avg_ai), 3)
        }
        img_ai_score = avg_ai
    else:
        image_final  = None
        img_ai_score = None

    # ── Combined score (60% text, 40% image) ─────────────────
    if img_ai_score is not None:
        combined_score = round(0.60 * text_ai_score + 0.40 * img_ai_score, 3)
    else:
        combined_score = round(text_ai_score, 3)

    return jsonify({
        "title":          scraped.get("title", ""),
        "text_preview":   scraped["text"][:300] + "..." if len(scraped["text"]) > 300 else scraped["text"],
        "image_url":      scraped.get("images", [None])[0],
        "image_urls":     scraped.get("images", []),
        "images_checked": images_checked,
        "text_result":    text_final,
        "image_result":   image_final,
        "combined_score": combined_score,
        "combined_label": "AI-generated" if combined_score >= 0.5 else "Human-written"
    })


# =======================
# PDF DOWNLOAD (optional)
# =======================
@app.route("/download-pdf", methods=["POST"])
def download_pdf():
    try:
        from pdf_generator import generate_pdf
        data     = request.get_json()
        filename = generate_pdf(data)
        return send_file(filename, as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =======================
if __name__ == "__main__":
    app.run(debug=False)

# .venv\Scripts\activate
# python app.py