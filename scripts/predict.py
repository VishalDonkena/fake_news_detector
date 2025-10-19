import sys
import os

# Add the project root to the Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.detector import FakeNewsDetector


def main():
    # Construct the absolute paths to the model and vectorizer
    MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "fake_news_model.h5")
    VECTORIZER_PATH = os.path.join(PROJECT_ROOT, "models", "tfidf_vectorizer.pkl")

    detector = FakeNewsDetector(model_path=MODEL_PATH, vectorizer_path=VECTORIZER_PATH)

    # Example news text
    news_text = "The government has announced a new set of economic policies aimed at boosting growth and reducing unemployment. The plan includes tax cuts for small businesses and investments in infrastructure projects. Economists have expressed cautious optimism about the potential impact of these measures."

    prediction, confidence = detector.predict_with_confidence(news_text)

    print(f"News text: '{news_text}'")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.2f}")


if __name__ == "__main__":
    main()
