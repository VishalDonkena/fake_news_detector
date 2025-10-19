#!/usr/bin/env python3
import argparse
import os
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Allow imports from src/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from model_trainer import ModelTrainer
from preprocessor import Preprocessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train fake news detector on a CSV dataset"
    )
    parser.add_argument(
        "--csv", required=True, help="Path to CSV with columns 'text' and 'label'"
    )
    parser.add_argument(
        "--models_dir",
        default=os.path.join(PROJECT_ROOT, "models"),
        help="Directory to save model artifacts",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument(
        "--max_length", type=int, default=256, help="Max sequence length"
    )
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

    print("Tokenizing and padding sequences...")
    tokenizer = Tokenizer(num_words=args.vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_sequences = tokenizer.texts_to_sequences(X_train)
    X_test_sequences = tokenizer.texts_to_sequences(X_test)

    X_train_padded = pad_sequences(
        X_train_sequences, maxlen=args.max_length, padding="post", truncating="post"
    )
    X_test_padded = pad_sequences(
        X_test_sequences, maxlen=args.max_length, padding="post", truncating="post"
    )

    print("Initializing model trainer...")
    model_trainer = ModelTrainer(vocab_size=args.vocab_size, max_length=args.max_length)

    print("Training Bi-LSTM model...")
    model_trainer.train(
        X_train_padded,
        y_train,
        X_val=X_test_padded,
        y_val=y_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    print("Evaluating model...")
    loss, accuracy = model_trainer.evaluate(X_test_padded, y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    tokenizer_path = os.path.join(args.models_dir, "bilstm_tokenizer.pkl")
    model_path = os.path.join(args.models_dir, "fake_news_bilstm_model.h5")

    print(f"Saving tokenizer -> {tokenizer_path}")
    joblib.dump(tokenizer, tokenizer_path)

    print(f"Saving model -> {model_path}")
    model_trainer.save_model(model_path)

    print("Done.")


if __name__ == "__main__":
    main()
