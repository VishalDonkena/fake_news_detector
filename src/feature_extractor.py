from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class FeatureExtractor:
    """Class for converting text documents into numerical features."""
    
    def __init__(self, method='tfidf'):
        """
        Initialize the feature extractor.
        
        Args:
            method (str): Feature extraction method. Currently only 'tfidf' is supported.
        """
        self.method = method
        self.vectorizer = None
        
        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
        else:
            raise ValueError(f"Unsupported method: {method}")
    
    def generate_features(self, texts, fit=True):
        """
        Generate features from text documents.
        
        Args:
            texts (list): List of text documents
            fit (bool): Whether to fit the vectorizer on the provided texts
            
        Returns:
            numpy.ndarray: Feature matrix
        """
        if self.method == 'tfidf':
            if fit:
                features = self.vectorizer.fit_transform(texts)
            else:
                features = self.vectorizer.transform(texts)
            return features.toarray()
        else:
            raise ValueError(f"Unsupported method: {method}")
    
    def get_feature_names(self):
        """
        Get the feature names from the vectorizer.
        
        Returns:
            list: List of feature names
        """
        if self.vectorizer is not None:
            return self.vectorizer.get_feature_names_out().tolist()
        return []
