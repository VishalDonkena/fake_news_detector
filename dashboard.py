#!/usr/bin/env python3
"""
Flask dashboard for the Fake News Detector.

Changes:
- Index page ("/") shows visualizations and metrics only (no inline prediction form).
- New subpage ("/test") hosts a dedicated front-end testing page with a prediction form
  and a results area.
- Prediction endpoint ("/predict") remains for JSON/form POSTs and is used by the /test page.

Run:
    python3 Code/fake_news_detector/dashboard.py
Open:
    http://127.0.0.1:5000/      -> dashboard (visualizations)
    http://127.0.0.1:5000/test  -> prediction UI

Note:
- This file expects the project layout under the same directory:
    - models/fake_news_model.h5
    - models/tokenizer.pkl
    - data/processed/news.csv
    - outputs/ (populated by evaluation script)
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Optional, List, Tuple

from flask import (
    Flask,
    jsonify,
    make_response,
    render_template_string,
    request,
    send_from_directory,
    url_for,
)

# ML / data libraries
import joblib
import matplotlib

# Use non-interactive backend for server environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow.keras.preprocessing.sequence import pad_sequences  # noqa: E402

# Paths relative to this file
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

MODEL_PATH = os.path.join(MODELS_DIR, "fake_news_model.h5")
TOKENIZER_PATH = os.path.join(MODELS_DIR, "tokenizer.pkl")
TEST_CSV_PATH = os.path.join(DATA_DIR, "news.csv")

# Flask app
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0  # disable caching for development

# Cached globals
_model = None
_tokenizer = None
_preprocessor = None


def load_preprocessor():
    """
    Try to import the project's Preprocessor (src/preprocessor.py). If not present,
    provide a lightweight fallback.
    """
    global _preprocessor
    if _preprocessor is not None:
        return _preprocessor

    src_dir = os.path.join(BASE_DIR, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    try:
        # project Preprocessor should be available as module 'preprocessor'
        from preprocessor import Preprocessor  # type: ignore

        _preprocessor = Preprocessor()
        return _preprocessor
    except Exception:
        # minimal fallback
        import re

        class MinimalPreprocessor:
            def clean_text(self, text: str) -> str:
                t = str(text).lower()
                t = re.sub(r"http\S+", " ", t)
                t = re.sub(r"[^a-zA-Z\s]", " ", t)
                t = " ".join(t.split())
                return t

        _preprocessor = MinimalPreprocessor()
        return _preprocessor


def load_model_and_tokenizer() -> Tuple[Optional[tf.keras.Model], Optional[object]]:
    """
    Lazily load Keras model and tokenizer/vectorizer (joblib).
    Returns (model, tokenizer) where either may be None if not present.
    """
    global _model, _tokenizer
    if _model is None:
        if os.path.exists(MODEL_PATH):
            try:
                _model = tf.keras.models.load_model(MODEL_PATH)
            except Exception:
                _model = None
    if _tokenizer is None:
        if os.path.exists(TOKENIZER_PATH):
            try:
                _tokenizer = joblib.load(TOKENIZER_PATH)
            except Exception:
                _tokenizer = None
    return _model, _tokenizer


def prepare_text_for_model(texts: List[str], tokenizer: object, maxlen: int = 500):
    """
    Prepare text(s) for model input. Supports:
      - Keras Tokenizer with `texts_to_sequences`
      - sklearn-style vectorizer with `transform`
    Returns numpy array ready for model.predict.
    """
    pre = load_preprocessor()
    cleaned = [pre.clean_text(t) for t in texts]

    if tokenizer is None:
        raise RuntimeError("Tokenizer not available")

    # Keras-style tokenizer
    if hasattr(tokenizer, "texts_to_sequences"):
        seqs = tokenizer.texts_to_sequences(cleaned)
        return pad_sequences(seqs, maxlen=maxlen)

    # sklearn-style vectorizer
    if hasattr(tokenizer, "transform"):
        X = tokenizer.transform(cleaned)
        try:
            return X.toarray()
        except Exception:
            return np.asarray(X)

    raise RuntimeError("Unsupported tokenizer type")


def _extract_probability(preds: np.ndarray) -> float:
    """
    Robust extraction of a single probability from `preds` array returned by model.predict.
    Supports scalar, (1,), (1,1), (1,2), or more complex shapes.
    """
    preds = np.asarray(preds)
    try:
        if preds.ndim == 0:
            return float(preds.item())
        if preds.ndim == 1:
            # take first element
            return float(preds.ravel()[0])
        if preds.ndim == 2:
            # (N, 2) -> class probs, use column 1
            if preds.shape[1] == 2:
                return float(preds[0, 1])
            # (N, 1) -> single prob
            if preds.shape[1] == 1:
                return float(preds[0, 0])
            # otherwise squeeze and take first
            s = np.squeeze(preds)
            return float(s.ravel()[0])
        # fallback: squeeze and take first
        s = np.squeeze(preds)
        if np.ndim(s) == 0:
            return float(s.item())
        return float(s.ravel()[0])
    except Exception:
        # extremely defensive fallback
        try:
            return float(np.squeeze(preds))
        except Exception:
            return 0.0


def predict_single(text: str) -> dict:
    """
    Predict a single input text. Returns dict: prediction, probability, confidence.
    """
    model, tokenizer = load_model_and_tokenizer()
    if model is None or tokenizer is None:
        raise RuntimeError("Model or tokenizer not available on server")

    X = prepare_text_for_model([text], tokenizer)
    preds = model.predict(X, verbose=0)
    prob = _extract_probability(preds)
    label = "Fake News" if prob >= 0.5 else "Real News"
    confidence = prob if prob >= 0.5 else 1.0 - prob
    return {
        "prediction": label,
        "probability": float(prob),
        "confidence": float(confidence),
    }


def ensure_outputs_dir():
    if not os.path.exists(OUTPUTS_DIR):
        os.makedirs(OUTPUTS_DIR, exist_ok=True)


def generate_class_distribution_plot() -> Optional[str]:
    """
    Generate class distribution bar plot and save to outputs/class_distribution.png.
    Returns path if created or already exists, otherwise None.
    """
    ensure_outputs_dir()
    out = os.path.join(OUTPUTS_DIR, "class_distribution.png")
    if os.path.exists(out):
        return out
    if not os.path.exists(TEST_CSV_PATH):
        return None
    try:
        df = pd.read_csv(TEST_CSV_PATH)
        if "label" not in df.columns:
            return None
        counts = df["label"].value_counts().sort_index()
        plt.figure(figsize=(5, 4))
        sns.barplot(x=[str(i) for i in counts.index], y=counts.values, palette="muted")
        plt.xlabel("Label")
        plt.ylabel("Count")
        plt.title("Class Distribution")
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        return out
    except Exception:
        traceback.print_exc()
        return None


def generate_top_words_plot(n_top: int = 25) -> Optional[str]:
    """
    Generate a top-word frequency plot (top n_top words across dataset).
    Saves outputs/top_words.png and returns the path or None.
    """
    ensure_outputs_dir()
    out = os.path.join(OUTPUTS_DIR, "top_words.png")
    if os.path.exists(out):
        return out
    if not os.path.exists(TEST_CSV_PATH):
        return None
    try:
        df = pd.read_csv(TEST_CSV_PATH)
        # choose 'text' or 'title'
        text_col = (
            "text"
            if "text" in df.columns
            else ("title" if "title" in df.columns else None)
        )
        if text_col is None:
            return None
        pre = load_preprocessor()
        texts = df[text_col].fillna("").astype(str).tolist()
        cleaned = [pre.clean_text(t) for t in texts]
        from collections import Counter

        cnt = Counter()
        for t in cleaned:
            cnt.update(t.split())
        # remove very short tokens
        for k in list(cnt):
            if len(k) < 3:
                del cnt[k]
        most = cnt.most_common(n_top)
        if not most:
            return None
        words, freqs = zip(*most)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(freqs), y=list(words), palette="viridis")
        plt.xlabel("Frequency")
        plt.title(f"Top {len(words)} Words")
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        return out
    except Exception:
        traceback.print_exc()
        return None


# INDEX page: visualizations only (no inline testing form)
INDEX_HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Fake News Detector - Dashboard</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 24px; background:#f7f7f7; color:#222; }
      .grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap:20px; }
      .card { background: white; padding:12px; border-radius:6px; box-shadow:0 1px 6px rgba(0,0,0,0.06); }
      img { max-width:100%; height:auto; display:block; margin-top:8px; border-radius:4px;}
      h1 { margin-bottom: 8px; }
      .small { font-size:0.9rem; color:#555; }
      a.button { display:inline-block; padding:8px 12px; background:#2b7cff; color:white; text-decoration:none; border-radius:4px; margin-bottom:12px;}
    </style>
  </head>
  <body>
    <h1>Fake News Detector — Dashboard</h1>
    <p class="small">Evaluation visualizations generated by the model evaluation script.</p>
    <p><a class="button" href="{{ url_for('test_page') }}">Go to Test Page (predict single article)</a>
       <a class="button" href="{{ url_for('health') }}" style="background:#4caf50">Health</a></p>

    <div class="grid">
      {% for name, url in images %}
        <div class="card">
          <h3>{{ name }}</h3>
          <a href="{{ url }}" target="_blank"><img src="{{ url }}" alt="{{ name }}"></a>
        </div>
      {% endfor %}
    </div>
  </body>
</html>
"""

# TEST page: separate subpage to test single-article predictions
TEST_HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Fake News Detector - Test</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 24px; background:#f7f7f7; color:#222; }
      .container { max-width:900px; margin:0 auto; }
      .card { background: white; padding:16px; border-radius:6px; box-shadow:0 1px 6px rgba(0,0,0,0.06); margin-bottom:16px; }
      label { font-weight:600; display:block; margin-bottom:6px; }
      textarea { width:100%; min-height:180px; padding:8px; font-size:14px; }
      button { padding:10px 14px; background:#2b7cff; color:white; border:none; border-radius:4px; cursor:pointer; }
      pre { background:#f2f4f7; padding:10px; border-radius:6px; overflow:auto; }
      .small { font-size:0.9rem; color:#555; }
      a { color:#2b7cff; }
    </style>
  </head>
  <body>
    <div class="container">
      <h1>Fake News Detector — Test Page</h1>
      <p class="small">Paste an article (or a short sentence) below and click Predict. This page uses the server /predict endpoint.</p>

      <div class="card">
        <form id="predict-form">
          <label for="text">Article text</label>
          <textarea id="text" name="text" placeholder="Paste article text here..."></textarea>
          <div style="margin-top:12px;">
            <button type="submit">Predict</button>
            <a href="{{ url_for('index') }}" style="margin-left:12px;">Back to Dashboard</a>
          </div>
        </form>
      </div>

      <div class="card">
        <h3>Result</h3>
        <div id="result"><em>Submit the form to see prediction result here.</em></div>
      </div>

    </div>

    <script>
      const form = document.getElementById('predict-form');
      form.onsubmit = async (e) => {
        e.preventDefault();
        const text = document.getElementById('text').value || '';
        if (!text.trim()) {
          document.getElementById('result').innerHTML = '<pre style="color:darkred">Please enter some text</pre>';
          return;
        }
        runPredict(text);
      };

      async function runPredict(text) {
        const resDiv = document.getElementById('result');
        resDiv.innerHTML = '<em>Predicting...</em>';
        try {
          const resp = await fetch('{{ url_for("predict") }}', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
          });
          const data = await resp.json();
          if (resp.ok) {
            // Render a human-friendly result: large label + smaller probability/confidence
            try {
              const predLabel = (data && data.prediction) ? data.prediction : 'Unknown';
              const prob = (data && typeof data.probability === 'number') ? (data.probability * 100) : null;
              const conf = (data && typeof data.confidence === 'number') ? (data.confidence * 100) : null;
              const color = predLabel.toLowerCase().includes('fake') ? '#b00020' : '#1a7f1a';
              const probText = (prob !== null) ? (Math.round(prob * 100) / 100).toFixed(2) + '%' : 'N/A';
              const confText = (conf !== null) ? (Math.round(conf * 100) / 100).toFixed(2) + '%' : 'N/A';
              resDiv.innerHTML =
                '<div style="font-family: Arial, Helvetica, sans-serif;">' +
                  '<div style="font-size:34px; font-weight:700; color:' + color + '; margin-bottom:6px;">' + predLabel + '</div>' +
                  '<div style="font-size:14px; color:#444;">' +
                    'Probability: <strong style=\"font-weight:600;\">' + probText + '</strong>' +
                    ' &nbsp;&nbsp; Confidence: <strong style=\"font-weight:600;\">' + confText + '</strong>' +
                  '</div>' +
                  '<div style="margin-top:10px; font-size:12px; color:#666;">Raw output (debug):</div>' +
                  '<pre style="margin-top:6px; background:#fafafa; padding:8px; border-radius:4px; overflow:auto; font-size:12px;">' +
                    JSON.stringify(data, null, 2) +
                  '</pre>' +
                '</div>';
            } catch (renderErr) {
              // Fallback to raw JSON if formatting fails
              resDiv.innerHTML = '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
            }
          } else {
            resDiv.innerHTML = '<pre style="color:darkred">' + JSON.stringify(data, null, 2) + '</pre>';
          }
        } catch (err) {
          resDiv.innerHTML = '<pre style="color:darkred">' + err.toString() + '</pre>';
        }
      }
    </script>
  </body>
</html>
"""


@app.route("/")
def index():
    """
    Dashboard UI has moved. Please run main.py for the web GUI.
    """
    return jsonify(
        {
            "message": "Dashboard UI moved to main.py. Run `python3 main.py` and open http://127.0.0.1:5000/.",
            "next_steps": [
                "cd to the project root",
                "run: python3 main.py",
                "open: http://127.0.0.1:5000/",
            ],
        }
    )


@app.route("/test")
def test_page():
    """
    Test page moved to the unified web GUI in main.py.
    """
    return jsonify(
        {
            "message": "The test page has moved. Please use the web UI served by main.py at /",
            "next_steps": ["run: python3 main.py", "open: http://127.0.0.1:5000/"],
        }
    )


@app.route("/outputs/<path:filename>")
def serve_output(filename):
    """
    Serve image files from outputs directory.
    """
    filepath = os.path.join(OUTPUTS_DIR, filename)
    if not os.path.exists(filepath):
        return make_response(jsonify({"error": "file not found"}), 404)
    return send_from_directory(OUTPUTS_DIR, filename)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept JSON or form POST and return prediction result.
    JSON body: {"text": "<article text>"}
    """
    try:
        payload = request.get_json(silent=True)
        if payload and "text" in payload:
            text = payload["text"]
        else:
            # fallback to form-data or URL-encoded
            text = request.form.get("text") or request.values.get("text") or ""
        if not text or not text.strip():
            return make_response(jsonify({"error": "empty text"}), 400)

        res = predict_single(text)
        return jsonify(res)
    except Exception as exc:
        tb = traceback.format_exc()
        return make_response(jsonify({"error": str(exc), "trace": tb}), 500)


@app.route("/health")
def health():
    model, tokenizer = load_model_and_tokenizer()
    return jsonify(
        {
            "model_exists": os.path.exists(MODEL_PATH),
            "tokenizer_exists": os.path.exists(TOKENIZER_PATH),
            "model_loaded": model is not None,
            "tokenizer_loaded": tokenizer is not None,
        }
    )


if __name__ == "__main__":
    # Warm up assets
    try:
        ensure_outputs_dir()
        generate_class_distribution_plot()
        generate_top_words_plot()
    except Exception:
        pass

    # Warm-load model/tokenizer if present
    try:
        load_model_and_tokenizer()
    except Exception:
        pass

    # Run dev server
    app.run(host="127.0.0.1", port=5000, debug=True)
