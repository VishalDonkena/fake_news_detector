import tensorflow as tf
import joblib
import numpy as np
import os
from .preprocessor import Preprocessor
from tensorflow.keras.preprocessing.sequence import pad_sequences


class FakeNewsDetector:
    def __init__(self, model_path, tokenizer_path, max_length=500):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.max_length = max_length

        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")

        if os.path.exists(tokenizer_path):
            self.tokenizer = joblib.load(tokenizer_path)
            print(f"Tokenizer loaded from {tokenizer_path}")
        else:
            raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")

        self.preprocessor = Preprocessor()

    def predict_with_confidence(self, text):
        cleaned_text = self.preprocessor.clean_text(text)

        sequence = self.tokenizer.texts_to_sequences([cleaned_text])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_length)

        prediction_proba = self.model.predict(padded_sequence, verbose=0)[0][0]

        if prediction_proba > 0.5:
            prediction = "Fake News"
            confidence = prediction_proba
        else:
            prediction = "Real News"
            confidence = 1 - prediction_proba

        return prediction, confidence

    def predict(self, text):
        prediction, _ = self.predict_with_confidence(text)
        return prediction
