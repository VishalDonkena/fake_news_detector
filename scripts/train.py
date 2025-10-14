#!/usr/bin/env python3
import argparse
import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Allow imports from src/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from preprocessor import Preprocessor
from feature_extractor import FeatureExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fake news detector on a CSV dataset")
    parser.add_argument("--csv", required=True, help="Path to CSV with columns 'text' and 'label'")
    parser.add_argument("--models_dir", default=os.path.join(PROJECT_ROOT, "models"), help="Directory to save model artifacts")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(args.models_dir, exist_ok=True)

    print(f"Loading dataset: {args.csv}")
    df = pd.read_csv(args.csv)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns")

    preprocessor = Preprocessor()
    print("Preprocessing text...")
    df["cleaned_text"] = df["text"].astype(str).apply(preprocessor.clean_text)

    X = df["cleaned_text"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Extracting TF-IDF features...")
    fe = FeatureExtractor(method="tfidf")
    X_train_features = fe.generate_features(X_train, fit=True)
    X_test_features = fe.generate_features(X_test, fit=False)

    # Build a simple dense model suitable for TF-IDF features
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam

    model = Sequential([
        Dense(512, activation='relu', input_shape=(X_train_features.shape[1],)),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    print("Training model...")
    history = model.fit(
        X_train_features, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.2,
        verbose=1
    )

    print("Evaluating model...")
    y_pred_proba = model.predict(X_test_features)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Real News', 'Fake News']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    vectorizer_path = os.path.join(args.models_dir, "tfidf_vectorizer.pkl")
    model_path = os.path.join(args.models_dir, "fake_news_model.h5")

    print(f"Saving vectorizer -> {vectorizer_path}")
    joblib.dump(fe.vectorizer, vectorizer_path)
    print(f"Saving model -> {model_path}")
    model.save(model_path)
    print("Done.")


if __name__ == "__main__":
    main()


