#!/usr/bin/env python3
"""
Fake News Detector - Unified Web GUI and Optional CLI

This refactors the application to primarily serve a Flask web-based UI for
single-article predictions, while keeping an optional CLI mode.

Usage:
  - Web GUI (default):
      python3 main.py
      python3 main.py --host 0.0.0.0 --port 5000 --debug

      Open your browser to:
        http://127.0.0.1:5000/

  - CLI mode:
      python3 main.py --cli

Environment variables:
  - FND_MODE=cli        # force CLI mode
  - FND_HOST=0.0.0.0    # host for Flask
  - FND_PORT=5000       # port for Flask
  - FND_DEBUG=1         # enable Flask debug
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Tuple

from flask import Flask, jsonify, make_response, render_template_string, request

# Ensure project imports
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Model artifacts
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "fake_news_model.h5")
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "models", "tokenizer.pkl")

# Import the detector (relies on src/preprocessor.py as well)
from src.detector import FakeNewsDetector  # noqa: E402


# ----------------------
# App/Detector lifecycle
# ----------------------
app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
_detector: Optional[FakeNewsDetector] = None
_detector_error: Optional[str] = None


def get_detector() -> Tuple[Optional[FakeNewsDetector], Optional[str]]:
    """
    Lazy-initialize a global FakeNewsDetector. Returns (detector, error_message).
    """
    global _detector, _detector_error
    if _detector is not None:
        return _detector, None

    try:
        _detector = FakeNewsDetector(
            model_path=MODEL_PATH, tokenizer_path=TOKENIZER_PATH
        )
        _detector_error = None
    except Exception as e:
        _detector = None
        _detector_error = str(e)
    return _detector, _detector_error


# -----------
# Web UI HTML
# -----------
INDEX_HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Fake News Detector</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style>
      :root {
        --bg: #f7f7f7;
        --fg: #222;
        --card: #fff;
        --muted: #666;
        --primary: #2b7cff;
        --success: #1a7f1a;
        --danger: #b00020;
      }
      * { box-sizing: border-box; }
      body { margin: 0; padding: 24px; background: var(--bg); color: var(--fg); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, "Helvetica Neue", Arial, "Apple Color Emoji","Segoe UI Emoji", "Segoe UI Symbol", sans-serif; }
      .container { max-width: 980px; margin: 0 auto; }
      .header { display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:10px; }
      h1 { margin: 0 0 8px 0; font-weight: 800; font-size: 28px; }
      .desc { margin: 0 0 14px 0; color: #444; }
      .card { background: var(--card); border-radius: 8px; box-shadow: 0 1px 6px rgba(0,0,0,0.06); padding: 16px; }
      label { display:block; font-weight: 600; margin-bottom: 6px; }
      textarea { width: 100%; min-height: 200px; font-size: 14px; padding: 10px; border-radius: 6px; border: 1px solid #ddd; outline: none; }
      textarea:focus { border-color: var(--primary); box-shadow: 0 0 0 3px rgba(43,124,255,0.15); }
      .actions { margin-top: 12px; display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
      button, .btn { display: inline-block; padding: 10px 14px; border-radius: 6px; text-decoration: none; border: 0; cursor: pointer; }
      button.primary { background: var(--primary); color: white; }
      .btn.secondary { background: #eee; color: #111; }
      .muted { color: var(--muted); }
      pre { background: #f2f4f7; border-radius: 8px; padding: 10px; overflow: auto; font-size: 12px; }
      .result-label { font-size: 34px; font-weight: 800; margin-bottom: 6px; }
      .row { display:grid; grid-template-columns: 1fr; gap: 16px; }
      @media (min-width: 860px) {
        .row.two { grid-template-columns: 2fr 1.2fr; }
      }
      .pill { display:inline-block; padding: 3px 8px; border-radius: 999px; background:#eef2ff; color:#1d4ed8; font-weight:600; font-size:12px; }
      .health-ok { color: var(--success); font-weight: 700; }
      .health-bad { color: var(--danger); font-weight: 700; }
    </style>
  </head>
  <body>
    <div class="container">
      <div class="header">
        <div>
          <h1>Fake News Detector</h1>
          <p class="desc">Paste an article or a snippet below and click Predict.</p>
        </div>
        <div>
          <a class="btn secondary" href="{{ url_for('health') }}" target="_blank" rel="noopener">Health</a>
        </div>
      </div>

      <div class="row two">
        <div class="card">
          <form id="predict-form">
            <label for="text">Article text</label>
            <textarea id="text" name="text" placeholder="Paste article text here..."></textarea>
            <div class="actions">
              <button class="primary" type="submit">Predict</button>
              <span class="muted">Tip: Enter complete articles for better accuracy</span>
            </div>
          </form>
        </div>

        <div class="card">
          <h3 style="margin-top:0;">Result</h3>
          <div id="result"><em class="muted">Submit the form to see prediction result here.</em></div>
        </div>
      </div>

      <div style="margin-top: 16px; color: #444;">
        <span class="pill">Model-backed</span>
        <span class="muted"> • </span>

      </div>
    </div>

    <script>
      const form = document.getElementById('predict-form');
      form.onsubmit = async (e) => {
        e.preventDefault();
        const text = document.getElementById('text').value || '';
        if (!text.trim()) {
          renderError('Please enter some text');
          return;
        }
        await runPredict(text);
      };

      function renderError(message, data) {
        const resDiv = document.getElementById('result');
        let extra = data ? ('<pre>' + JSON.stringify(data, null, 2) + '</pre>') : '';
        resDiv.innerHTML = '<div><div class="result-label" style="color:#b00020">Error</div><div class="muted">' + message + '</div>' + extra + '</div>';
      }

      function renderResult(data) {
        const resDiv = document.getElementById('result');
        try {
          const predLabel = (data && data.prediction) ? data.prediction : 'Unknown';
          const prob = (data && typeof data.probability === 'number') ? (data.probability * 100) : null;
          const conf = (data && typeof data.confidence === 'number') ? (data.confidence * 100) : null;
          const color = predLabel.toLowerCase().includes('fake') ? '#b00020' : '#1a7f1a';
          const probText = (prob !== null) ? (Math.round(prob * 100) / 100).toFixed(2) + '%' : 'N/A';
          const confText = (conf !== null) ? (Math.round(conf * 100) / 100).toFixed(2) + '%' : 'N/A';
          resDiv.innerHTML =
            '<div style="font-family: inherit;">' +
              '<div class="result-label" style="color:' + color + ';">' + predLabel + '</div>' +
              '<div style="font-size:14px; color:#444;">' +
                'Probability: <strong>' + probText + '</strong>' +
                ' &nbsp;&nbsp; Confidence: <strong>' + confText + '</strong>' +
              '</div>' +
              '<div style="margin-top:10px; font-size:12px; color:#666;">Raw output (debug):</div>' +
              '<pre style="margin-top:6px;">' + JSON.stringify(data, null, 2) + '</pre>' +
            '</div>';
        } catch (err) {
          resDiv.innerHTML = '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
        }
      }

      async function runPredict(text) {
        const resDiv = document.getElementById('result');
        resDiv.innerHTML = '<em class="muted">Predicting...</em>';
        try {
          const resp = await fetch('{{ url_for("predict") }}', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
          });
          const data = await resp.json();
          if (!resp.ok) {
            renderError(data && data.error ? data.error : 'Prediction failed', data);
            return;
          }
          renderResult(data);
        } catch (err) {
          renderError(err.toString());
        }
      }
    </script>
  </body>
</html>
"""


# -------------
# Flask routes
# -------------
@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML)


@app.route("/predict", methods=["POST"])
def predict():
    det, err = get_detector()
    if det is None:
        return make_response(
            jsonify({"error": err or "Model/tokenizer not available"}), 503
        )

    # Accept JSON { "text": "..." } or form fields
    payload = request.get_json(silent=True)
    if payload and "text" in payload:
        text = payload["text"]
    else:
        text = request.form.get("text") or request.values.get("text") or ""

    text = (text or "").strip()
    if not text:
        return make_response(jsonify({"error": "empty text"}), 400)

    try:
        prediction, confidence = det.predict_with_confidence(text)
        # The model outputs probability for "Fake" in detector; reconstruct it:
        probability_fake = (
            confidence if prediction == "Fake News" else (1.0 - confidence)
        )
        return jsonify(
            {
                "prediction": prediction,
                "probability": float(probability_fake),
                "confidence": float(confidence),
            }
        )
    except Exception as e:
        return make_response(jsonify({"error": str(e)}), 500)


@app.route("/health", methods=["GET"])
def health():
    det, err = get_detector()
    model_exists = os.path.exists(MODEL_PATH)
    tokenizer_exists = os.path.exists(TOKENIZER_PATH)
    return jsonify(
        {
            "model_path": MODEL_PATH,
            "tokenizer_path": TOKENIZER_PATH,
            "model_exists": model_exists,
            "tokenizer_exists": tokenizer_exists,
            "model_loaded": det is not None,
            "tokenizer_loaded": (getattr(det, "tokenizer", None) is not None)
            if det
            else False,
            "error": err,
        }
    )


# ------------
# CLI fallback
# ------------
def cli_main() -> int:
    print("=" * 60)
    print("🔍 FAKE NEWS DETECTOR — CLI")
    print("=" * 60)
    print("Type/paste an article and press Enter to get a prediction.")
    print("Type 'exit' or 'quit' to leave.")
    print()

    det, err = get_detector()
    if det is None:
        print(f"Error loading model/tokenizer: {err or 'unknown error'}")
        print("Make sure the model files exist and are accessible:")
        print(f"  - {MODEL_PATH}")
        print(f"  - {TOKENIZER_PATH}")
        return 1

    try:
        while True:
            article = input("> ").strip()
            if article.lower() in {"exit", "quit", "q"}:
                break
            if not article:
                continue
            prediction, confidence = det.predict_with_confidence(article)
            probability_fake = (
                confidence if prediction == "Fake News" else (1.0 - confidence)
            )
            print(
                f"📰 Prediction: {prediction}  |  Probability(Fake): {probability_fake:.2%}  |  Confidence: {confidence:.2%}"
            )
    except (EOFError, KeyboardInterrupt):
        pass

    print("Goodbye!")
    return 0


# ----
# Main
# ----
def _parse_args():
    parser = argparse.ArgumentParser(description="Fake News Detector - Web GUI and CLI")
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode")
    parser.add_argument(
        "--host",
        default=os.getenv("FND_HOST", "127.0.0.1"),
        help="Host for web server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("FND_PORT", "5000")),
        help="Port for web server (default: 5000)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=os.getenv("FND_DEBUG", "0") in ("1", "true", "True"),
        help="Enable Flask debug mode",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    mode_env = os.getenv("FND_MODE", "").lower()
    use_cli = args.cli or (mode_env == "cli")

    if use_cli:
        raise SystemExit(cli_main())
    else:
        # Warm load detector if possible (non-fatal on failure)
        get_detector()
        app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
