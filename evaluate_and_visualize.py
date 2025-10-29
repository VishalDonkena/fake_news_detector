#!/usr/bin/env python3
"""
evaluate_and_visualize.py

Evaluation and visualization utilities for the Fake News Detector.

Features:
- Load test dataset (CSV with `text` and `label` columns).
- Load Keras model and tokenizer/vectorizer from `models/`.
- Prepare inputs robustly (supports sequence tokenizers and TF-IDF-style vectorizers).
- Compute metrics: accuracy, precision, recall, F1, ROC AUC, PR AUC, confusion matrix.
- Generate and save plots:
  - Confusion matrix heatmap
  - ROC curve
  - Precision-Recall curve
  - Prediction probability histogram
  - Example predictions (a small sample with text + predicted score)
- CLI interface for easy use.

Usage (from project root):
    python3 Code/fake_news_detector/evaluate_and_visualize.py \
        --test-csv Code/fake_news_detector/data/processed/news.csv \
        --model Code/fake_news_detector/models/fake_news_model.h5 \
        --tokenizer Code/fake_news_detector/models/tokenizer.pkl \
        --output-dir Code/fake_news_detector/outputs

Notes:
- The script will try to import the project's `src.preprocessor.Preprocessor`.
  If that import fails it falls back to a minimal cleaning function.
- The model is expected to output a single probability per sample (probability of label 1 = Fake).
  If the model outputs shape (N,2) we take column index 1 as positive class.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Tuple, Optional, Any, Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    auc,
)
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fake-news-eval")


# Try to import the project's Preprocessor. If not available, define a minimal fallback.
def import_preprocessor():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    # Ensure src folder is importable
    src_path = os.path.join(project_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    try:
        from preprocessor import Preprocessor  # type: ignore

        logger.info("Loaded Preprocessor from project src.preprocessor")
        return Preprocessor
    except Exception:
        logger.warning(
            "Could not import project Preprocessor. Falling back to minimal cleaner."
        )

        class MinimalPreprocessor:
            def clean_text(self, text: str) -> str:
                # Very small cleanup similar to the project's preprocessor but without NLTK
                import re

                text = text.lower()
                text = re.sub(r"http\S+", " ", text)  # remove urls
                text = re.sub(r"[^a-zA-Z\s]", " ", text)
                text = re.sub(r"\s+", " ", text).strip()
                return text

        return MinimalPreprocessor


Preprocessor = import_preprocessor()


def load_data(
    csv_path: str, text_col: str = "text", label_col: str = "label"
) -> pd.DataFrame:
    """Load dataset CSV into a DataFrame, keeping only required columns."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Test CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"CSV must contain columns: {text_col}, {label_col}")
    # Drop NA and keep only necessary columns
    df = df[[text_col, label_col]].dropna().reset_index(drop=True)
    return df


def prepare_inputs(
    texts: Sequence[str],
    tokenizer: Any,
    vectorizer: Optional[Any],
    max_length: int = 500,
    preprocessor: Optional[Any] = None,
) -> np.ndarray:
    """
    Convert raw texts into model-ready numeric inputs.

    - If `tokenizer` exposes `texts_to_sequences`, we assume a sequence-based Keras tokenizer.
    - Otherwise, if `vectorizer` (e.g. TF-IDF) is provided, use its `transform`.
    - Falls back to tokenizing with tokenizer.transform if that's present.

    Returns a NumPy array ready to pass to model.predict().
    """
    if preprocessor is None:
        preprocessor = Preprocessor()

    cleaned = [preprocessor.clean_text(str(t)) for t in texts]

    # Sequence-based tokenizer (Keras)
    if hasattr(tokenizer, "texts_to_sequences"):
        logger.info("Preparing inputs with sequence tokenizer (texts_to_sequences).")
        sequences = tokenizer.texts_to_sequences(cleaned)
        padded = pad_sequences(sequences, maxlen=max_length)
        return np.asarray(padded)

    # Scikit-learn vectorizer (TF-IDF, Count)
    if vectorizer is not None and hasattr(vectorizer, "transform"):
        logger.info("Preparing inputs with provided vectorizer.transform().")
        X = vectorizer.transform(cleaned)
        # Ensure dense array if model expects dense input
        try:
            X_arr = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        except Exception:
            X_arr = np.asarray(X)
        return X_arr

    # If tokenizer has transform (like sklearn pipeline)
    if hasattr(tokenizer, "transform"):
        logger.info("Preparing inputs with tokenizer.transform().")
        X = tokenizer.transform(cleaned)
        try:
            X_arr = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        except Exception:
            X_arr = np.asarray(X)
        return X_arr

    raise RuntimeError("Could not prepare inputs: tokenizer/vectorizer not usable.")


def load_model_and_assets(
    model_path: str,
    tokenizer_path: Optional[str] = None,
    vectorizer_path: Optional[str] = None,
) -> Tuple[Any, Optional[Any], Optional[Any]]:
    """Load Keras model and tokenizer/vectorizer if available."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    logger.info(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)

    tokenizer = None
    vectorizer = None

    if tokenizer_path and os.path.exists(tokenizer_path):
        try:
            tokenizer = joblib.load(tokenizer_path)
            logger.info(f"Loaded tokenizer from {tokenizer_path}")
        except Exception as exc:
            logger.warning(f"Failed to load tokenizer: {exc}. Setting tokenizer=None")

    if vectorizer_path and os.path.exists(vectorizer_path):
        try:
            vectorizer = joblib.load(vectorizer_path)
            logger.info(f"Loaded vectorizer from {vectorizer_path}")
        except Exception as exc:
            logger.warning(f"Failed to load vectorizer: {exc}. Setting vectorizer=None")

    return model, tokenizer, vectorizer


def predict_probs(model: Any, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
    """
    Predict probabilities for the positive class (label=1).

    Handles models that output a single probability (shape (N,)) or two-class probabilities (N,2).
    """
    preds = model.predict(X, batch_size=batch_size, verbose=0)
    preds = np.asarray(preds)
    # Common cases:
    if preds.ndim == 1:
        probs = preds
    elif preds.ndim == 2 and preds.shape[1] == 1:
        probs = preds[:, 0]
    elif preds.ndim == 2 and preds.shape[1] == 2:
        probs = preds[:, 1]
    else:
        # Attempt to squeeze last dimension
        probs = np.squeeze(preds)
    probs = probs.astype(float)
    return probs


def evaluate_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict:
    """Compute evaluation metrics and return a dictionary of results."""
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    # ROC AUC requires at least one positive and one negative sample
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        metrics["roc_auc"] = float("nan")

    return metrics


def plot_and_save_confusion(
    y_true: np.ndarray, y_pred: np.ndarray, out_path: str
) -> None:
    """Plot confusion matrix heatmap and save it."""
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Real (0)", "Fake (1)"]
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"Saved confusion matrix to {out_path}")


def plot_roc_pr(y_true: np.ndarray, y_prob: np.ndarray, out_dir: str) -> None:
    """Plot ROC curve and Precision-Recall curve and save both."""
    # ROC
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.2)
        plt.tight_layout()
        roc_path = os.path.join(out_dir, "roc_curve.png")
        plt.savefig(roc_path, dpi=150)
        plt.close()
        logger.info(f"Saved ROC curve to {roc_path}")
    except Exception as exc:
        logger.warning(f"Could not plot ROC: {exc}")

    # Precision-Recall
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower left")
        plt.grid(alpha=0.2)
        plt.tight_layout()
        pr_path = os.path.join(out_dir, "pr_curve.png")
        plt.savefig(pr_path, dpi=150)
        plt.close()
        logger.info(f"Saved Precision-Recall curve to {pr_path}")
    except Exception as exc:
        logger.warning(f"Could not plot Precision-Recall curve: {exc}")


def plot_probability_histogram(y_prob: np.ndarray, out_path: str) -> None:
    """Plot histogram / KDE of predicted probabilities."""
    plt.figure(figsize=(6, 4))
    sns.histplot(y_prob, bins=50, kde=True, stat="density", color="C0")
    plt.xlabel("Predicted probability (Fake)")
    plt.title("Prediction Probability Distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"Saved probability histogram to {out_path}")


def save_classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, out_path: str
) -> None:
    """Write a sklearn classification report to a text file."""
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(report)
    logger.info(f"Saved classification report to {out_path}")


def save_sample_predictions(
    texts: Sequence[str],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    out_path: str,
    n: int = 10,
) -> None:
    """Save a small sample table of example predictions for manual inspection."""
    # Construct DataFrame
    df = pd.DataFrame({"text": texts, "true": y_true, "prob_fake": y_prob})
    # Sort by confidence (both high and low)
    df_sorted = pd.concat(
        [
            df.sort_values("prob_fake", ascending=False).head(n // 2),
            df.sort_values("prob_fake", ascending=True).head(n // 2),
        ]
    )
    df_sorted = df_sorted.reset_index(drop=True)
    # Truncate text for readability
    df_sorted["text_trunc"] = df_sorted["text"].str[:400].str.replace("\n", " ")
    df_sorted = df_sorted[["text_trunc", "true", "prob_fake"]]
    df_sorted.to_csv(out_path, index=False)
    logger.info(f"Saved sample predictions to {out_path}")


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate and visualize Fake News Detector"
    )
    parser.add_argument(
        "--test-csv", type=str, required=True, help="Path to test CSV (text,label)"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to Keras model (.h5)"
    )
    parser.add_argument(
        "--tokenizer", type=str, default=None, help="Path to tokenizer.pkl (joblib)"
    )
    parser.add_argument(
        "--vectorizer",
        type=str,
        default=None,
        help="Path to vectorizer.pkl (joblib - optional)",
    )
    parser.add_argument(
        "--max-length", type=int, default=500, help="Max sequence length for padding"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for classification",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "outputs"),
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=12,
        help="Number of example predictions to save",
    )
    args = parser.parse_args(argv)

    ensure_dir(args.output_dir)

    # Load data
    logger.info("Loading test data...")
    df = load_data(args.test_csv, text_col="text", label_col="label")
    texts = df["text"].astype(str).tolist()
    y_true = df["label"].astype(int).values

    # Load model and assets
    model, tokenizer, vectorizer = load_model_and_assets(
        args.model, args.tokenizer, args.vectorizer
    )

    # Prepare inputs
    X = prepare_inputs(
        texts,
        tokenizer=tokenizer,
        vectorizer=vectorizer,
        max_length=args.max_length,
        preprocessor=Preprocessor(),
    )

    # Predict probabilities
    logger.info("Running model predictions...")
    y_prob = predict_probs(model, X)

    # Evaluate
    metrics = evaluate_metrics(y_true, y_prob, threshold=args.threshold)
    y_pred = (y_prob >= args.threshold).astype(int)

    logger.info("Evaluation metrics:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save textual classification report
    report_path = os.path.join(args.output_dir, "classification_report.txt")
    save_classification_report(y_true, y_pred, report_path)

    # Plots
    plot_and_save_confusion(
        y_true, y_pred, os.path.join(args.output_dir, "confusion_matrix.png")
    )
    plot_roc_pr(y_true, y_prob, args.output_dir)
    plot_probability_histogram(
        y_prob, os.path.join(args.output_dir, "probability_histogram.png")
    )

    # Save examples
    sample_out = os.path.join(args.output_dir, "sample_predictions.csv")
    save_sample_predictions(texts, y_true, y_prob, sample_out, n=args.sample_size)

    # Save summary metrics CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(args.output_dir, "metrics_summary.csv"), index=False)
    logger.info(
        f"Saved metrics summary to {os.path.join(args.output_dir, 'metrics_summary.csv')}"
    )

    logger.info("All outputs saved in %s", args.output_dir)

    # --- Short-sentence tests (quick synthetic checks) ---
    # Add a small set of short, varied sentences to test model behavior on concise inputs.
    try:
        short_sentences = [
            "Win big prize now!",
            "Click here to get rich quick.",
            "Breaking: celebrity dies suddenly.",
            "Government approves new law today.",
            "Study finds coffee improves health.",
            "Conspiracy uncovered by insiders.",
            "Local man wins lottery.",
            "Learn secret to losing weight fast.",
            "Elections postponed indefinitely.",
            "New technology will change everything.",
        ]

        logger.info(
            "Preparing short-sentence test inputs (%d samples)...", len(short_sentences)
        )
        X_short = prepare_inputs(
            short_sentences,
            tokenizer=tokenizer,
            vectorizer=vectorizer,
            max_length=maxlen,
            preprocessor=Preprocessor(),
        )

        logger.info("Running model predictions for short sentences...")
        probs_short = predict_probs(model, X_short)

        short_df = pd.DataFrame(
            {
                "text": short_sentences,
                "prob_fake": probs_short,
                "predicted_label": (probs_short > args.threshold).astype(int),
            }
        )
        short_out = os.path.join(args.output_dir, "short_sentence_tests.csv")
        short_df.to_csv(short_out, index=False)
        logger.info("Saved short-sentence test results to %s", short_out)
    except Exception as exc:
        logger.warning("Could not run short-sentence tests: %s", exc)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
