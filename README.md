# Fake News Detector

A deep learning project to detect fake news articles. It now ships with a built-in web GUI served directly from `main.py`, plus an optional CLI mode.

- Web GUI: run `python main.py` and open the browser
- CLI: run `python main.py --cli`
- Model: LSTM-based classifier (Keras/TensorFlow)
- Preprocessing: NLTK-based cleaning and tokenization
- Evaluation: helper script generates metrics and plots

## Quick Start

1) Clone and enter the project
- macOS/Linux:
  - python3 -m venv .venv
  - source .venv/bin/activate
- Windows:
  - py -3 -m venv .venv
  - .venv\Scripts\activate

2) Install dependencies
- pip install flask tensorflow joblib numpy pandas matplotlib seaborn scikit-learn nltk

3) Make sure model artifacts exist
- models/fake_news_model.h5
- models/tokenizer.pkl

4) Start the web GUI
- python fake_news_detector/main.py
- Open http://127.0.0.1:5000/

Optional CLI mode:
- python fake_news_detector/main.py --cli

## Web GUI (main.py)

`fake_news_detector/main.py` now hosts the web interface and the API:
- GET / : Web page with a textarea and Predict button
- POST /predict : Accepts JSON or form data with "text" and returns:
  - prediction ("Fake News" or "Real News")
  - probability (P(fake) as a float)
  - confidence (confidence of predicted class)
- GET /health : Returns model/tokenizer presence and load status

Example curl:
- curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"text\":\"Your article text...\"}"

## CLI

Run:
- python fake_news_detector/main.py --cli

Type/paste an article to get a prediction. Type "exit" or "quit" to leave.

## Data

Current layout:
- data/raw/
  - Fake.csv
  - True.csv
- data/processed/
  - news.csv
- data/test/
  - fake.csv
  - real.csv

Notes:
- The processed combined dataset remains in data/processed/news.csv
- Test data is now split by class into two files: data/test/fake.csv and data/test/real.csv
- The previous combined data/test/test.csv is no longer used

If you need a single test CSV for tooling (columns: text,label), you can combine the splits:
- python - <<'PY'
import csv, os
base = "fake_news_detector/data/test"
files = ["fake.csv", "real.csv"]
out = os.path.join(base, "test_merged.csv")
rows = []
for fn in files:
    with open(os.path.join(base, fn), newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        rows.extend(r)
with open(out, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)
print("Wrote:", out)
PY

## Model and Preprocessing

- LSTM-based binary classifier using Keras/TensorFlow
- Tokenization and sequence padding via Keras Tokenizer
- NLTK-based preprocessing:
  - Lowercasing, punctuation removal
  - Stopword removal
  - Lemmatization
- See:
  - src/detector.py (inference)
  - src/preprocessor.py (cleaning)

## Evaluation and Visualizations

The helper script `evaluate_and_visualize.py` generates metrics and plots (ROC, PR, confusion matrix, probability histogram) and saves outputs to fake_news_detector/outputs.

Usage example:
- python fake_news_detector/evaluate_and_visualize.py --test-csv fake_news_detector/data/processed/news.csv --model fake_news_detector/models/fake_news_model.h5 --tokenizer fake_news_detector/models/tokenizer.pkl

If you want to evaluate against the split test data, merge them into a single CSV first (see Data section).

## Windows Setup

You do not need any .venv from macOS. On Windows, create your own virtual environment and install dependencies:

1) Install Python 3.x (use the same major/minor version when possible)

2) Create and activate a venv
- py -3 -m venv .venv
- .venv\Scripts\activate

3) Install dependencies
- pip install flask tensorflow joblib numpy pandas matplotlib seaborn scikit-learn nltk
- If you encounter TensorFlow install issues on Windows, try:
  - pip install --upgrade pip setuptools wheel
  - pip install tensorflow
  - As a fallback for CPU-only, older setups sometimes use: pip install tensorflow-cpu

4) Run the app
- python fake_news_detector\main.py
- Open http://127.0.0.1:5000/

NLTK data
- The first run downloads required corpora (punkt, stopwords, wordnet). If download prompts appear, allow network access.

GPU (optional)
- Windows + GPU requires proper NVIDIA CUDA/cuDNN setup and a compatible TensorFlow build. If you’re not sure, stick to CPU.

## macOS Notes

- Create and activate a venv:
  - python3 -m venv .venv
  - source .venv/bin/activate
- Install dependencies:
  - pip install flask tensorflow joblib numpy pandas matplotlib seaborn scikit-learn nltk
- Apple Silicon users may consider:
  - pip install tensorflow-macos tensorflow-metal
  - If you do, keep both the training and inference environments consistent

## Virtual Environments and Requirements

- Don’t commit .venv folders; they are OS-specific and disposable
- Recreate venvs locally on each machine
- If you maintain a requirements file, pin versions for reproducibility:
  - pip freeze > requirements.txt
  - pip install -r requirements.txt

## Project Structure

fake_news_detector/
├── data/
│   ├── raw/
│   │   ├── Fake.csv
│   │   └── True.csv
│   ├── processed/
│   │   └── news.csv
│   └── test/
│       ├── fake.csv
│       └── real.csv
├── models/
│   ├── fake_news_model.h5
│   └── tokenizer.pkl
├── outputs/                 # evaluation plots and reports
├── src/
│   ├── detector.py
│   ├── preprocessor.py
│   ├── data_loader.py
│   ├── feature_extractor.py
│   └── model_trainer.py
├── main.py                  # web UI + API + optional CLI entrypoint
├── dashboard.py             # legacy dashboard; no UI needed now
├── evaluate_and_visualize.py
└── README.md

## Troubleshooting

- Model/tokenizer not found:
  - Ensure models/fake_news_model.h5 and models/tokenizer.pkl exist
- Import errors:
  - Activate your venv and reinstall dependencies
- TensorFlow install issues:
  - Upgrade pip: pip install --upgrade pip setuptools wheel
  - Try CPU-only install or platform-specific builds

## License

MIT License