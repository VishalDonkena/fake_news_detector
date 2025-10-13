import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, LSTM, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np


class ModelTrainer:
    """Class for training deep learning models for fake news detection."""
    
    def __init__(self, vocab_size=5000, embedding_dim=128, max_length=100):
        """
        Initialize the model trainer.
        
        Args:
            vocab_size (int): Size of the vocabulary
            embedding_dim (int): Dimension of the embedding layer
            max_length (int): Maximum length of input sequences
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.model = None
        self._build_model()
    
    def _build_model(self):
        """Build the Keras Sequential model architecture."""
        self.model = Sequential([
            Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_length
            ),
            SpatialDropout1D(0.2),
            LSTM(100, dropout=0.2, recurrent_dropout=0.2),
            Dense(1, activation='sigmoid')
        ])
        
        # Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model architecture:")
        self.model.summary()
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=10, batch_size=32, validation_split=0.2):
        """
        Train the model on the provided data.
        
        Args:
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training labels
            X_val (numpy.ndarray, optional): Validation features
            y_val (numpy.ndarray, optional): Validation labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            validation_split (float): Fraction of training data to use for validation
            
        Returns:
            tensorflow.keras.callbacks.History: Training history
        """
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            validation_split = None
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            validation_data=validation_data,
            verbose=1
        )
        
        return history
    
    def save_model(self, filepath):
        """
        Save the trained model to a file.
        
        Args:
            filepath (str): Path where to save the model
        """
        if self.model is not None:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
        else:
            raise ValueError("No model to save. Train the model first.")
    
    def load_model(self, filepath):
        """
        Load a pre-trained model from a file.
        
        Args:
            filepath (str): Path to the saved model
        """
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X (numpy.ndarray): Input features
            
        Returns:
            numpy.ndarray: Predictions
        """
        if self.model is not None:
            return self.model.predict(X)
        else:
            raise ValueError("No model available. Train or load a model first.")
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test (numpy.ndarray): Test features
            y_test (numpy.ndarray): Test labels
            
        Returns:
            tuple: (loss, accuracy)
        """
        if self.model is not None:
            return self.model.evaluate(X_test, y_test, verbose=0)
        else:
            raise ValueError("No model available. Train or load a model first.")
