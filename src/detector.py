import tensorflow as tf
import joblib
import numpy as np
import os
from .preprocessor import Preprocessor


class FakeNewsDetector:
    """Main class for fake news detection that orchestrates the entire pipeline."""
    
    def __init__(self, model_path, vectorizer_path):
        """
        Initialize the fake news detector.
        
        Args:
            model_path (str): Path to the saved Keras model
            vectorizer_path (str): Path to the saved TF-IDF vectorizer
        """
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        
        # Load the pre-trained model
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Load the fitted vectorizer
        if os.path.exists(vectorizer_path):
            self.vectorizer = joblib.load(vectorizer_path)
            print(f"Vectorizer loaded from {vectorizer_path}")
        else:
            raise FileNotFoundError(f"Vectorizer file not found at {vectorizer_path}")
        
        # Initialize preprocessor
        self.preprocessor = Preprocessor()
    
    def predict(self, text):
        """
        Predict whether a news article is fake or real.
        
        Args:
            text (str): Raw news article text
            
        Returns:
            str: "Fake News" or "Real News"
        """
        try:
            # Step 1: Clean the text using preprocessor
            cleaned_text = self.preprocessor.clean_text(text)
            
            # Step 2: Transform text using the loaded vectorizer
            text_features = self.vectorizer.transform([cleaned_text]).toarray()
            
            # Step 3: Get prediction from the loaded model
            prediction_proba = self.model.predict(text_features, verbose=0)[0][0]
            
            # Step 4: Convert probability to label
            if prediction_proba > 0.5:
                return "Fake News"
            else:
                return "Real News"
                
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return "Error: Could not process the text"
    
    def predict_with_confidence(self, text):
        """
        Predict with confidence score.
        
        Args:
            text (str): Raw news article text
            
        Returns:
            tuple: (prediction, confidence_score)
        """
        try:
            # Step 1: Clean the text using preprocessor
            cleaned_text = self.preprocessor.clean_text(text)
            
            # Step 2: Transform text using the loaded vectorizer
            text_features = self.vectorizer.transform([cleaned_text]).toarray()
            
            # Step 3: Get prediction from the loaded model
            prediction_proba = self.model.predict(text_features, verbose=0)[0][0]
            
            # Step 4: Convert probability to label and confidence
            if prediction_proba > 0.5:
                prediction = "Fake News"
                confidence = prediction_proba
            else:
                prediction = "Real News"
                confidence = 1 - prediction_proba
            
            return prediction, confidence
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return "Error: Could not process the text", 0.0
    
    def batch_predict(self, texts):
        """
        Predict multiple texts at once.
        
        Args:
            texts (list): List of raw news article texts
            
        Returns:
            list: List of predictions
        """
        predictions = []
        for text in texts:
            prediction = self.predict(text)
            predictions.append(prediction)
        return predictions
