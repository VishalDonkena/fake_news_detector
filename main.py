#!/usr/bin/env python3
"""
Fake News Detection - Main User Interface

This script provides a command-line interface for detecting fake news articles.
Users can input news articles and get predictions on whether they are fake or real.
"""

import os
import sys

# Add the project root to the Python path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.detector import FakeNewsDetector


def main() -> None:
    """Main user interface for the fake news detector."""
    print("=" * 60)
    print("🔍 FAKE NEWS DETECTOR")
    print("=" * 60)
    print("Welcome to the Fake News Detection System!")
    print("This tool analyzes news articles to determine if they are fake or real.")
    print()

    try:
        MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "fake_news_model.h5")
        TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "models", "tokenizer.pkl")
        detector = FakeNewsDetector(
            model_path=MODEL_PATH, tokenizer_path=TOKENIZER_PATH
        )
    except Exception as e:
        print(f"❌ {e}")
        print("Please train the model first using the training notebook.")
        print("Run: jupyter notebook notebooks/02_model_training.ipynb")
        return

    while True:
        print("\nEnter a news article (or type 'exit' to quit):")
        try:
            article = input("> ")
            if article.lower() == "exit":
                break
            if not article.strip():
                continue

            prediction, confidence = detector.predict_with_confidence(article)

            print(f"\n📰 Prediction: {prediction} (Confidence: {confidence:.2%})")

        except (KeyboardInterrupt, EOFError):
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    print("\nThank you for using the Fake News Detector!")


def print_help():
    """Print help information for the user."""
    print("\n" + "=" * 50)
    print("📖 HELP - FAKE NEWS DETECTOR")
    print("=" * 50)
    print("This tool analyzes news articles to detect fake news.")
    print()
    print("Commands:")
    print("  • Enter any news article text to analyze it")
    print("  • Type 'quit', 'exit', or 'q' to stop the program")
    print("  • Type 'help', 'h', or '?' to show this help")
    print()
    print("Tips:")
    print("  • Enter complete articles for better accuracy")
    print("  • The tool works best with news articles in English")
    print("  • Results include confidence scores")
    print("=" * 50)


if __name__ == "__main__":
    main()
