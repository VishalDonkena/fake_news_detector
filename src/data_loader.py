import pandas as pd
import os
from typing import Optional, Tuple


class DataLoader:
    """Class for loading and managing datasets for fake news detection."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data loader.
        
        Args:
            data_dir (str): Directory containing the data files
        """
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
    
    def load_dataset(self, filename: str, raw: bool = True) -> Optional[pd.DataFrame]:
        """
        Load a dataset from CSV file.
        
        Args:
            filename (str): Name of the CSV file
            raw (bool): Whether to load from raw or processed directory
            
        Returns:
            pd.DataFrame: Loaded dataset or None if file not found
        """
        if raw:
            filepath = os.path.join(self.raw_dir, filename)
        else:
            filepath = os.path.join(self.processed_dir, filename)
        
        try:
            df = pd.read_csv(filepath)
            print(f"Dataset loaded successfully from {filepath}")
            print(f"Shape: {df.shape}")
            return df
        except FileNotFoundError:
            print(f"Dataset not found at {filepath}")
            return None
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            return None
    
    def save_dataset(self, df: pd.DataFrame, filename: str, raw: bool = False) -> bool:
        """
        Save a dataset to CSV file.
        
        Args:
            df (pd.DataFrame): Dataset to save
            filename (str): Name of the CSV file
            raw (bool): Whether to save to raw or processed directory
            
        Returns:
            bool: True if successful, False otherwise
        """
        if raw:
            filepath = os.path.join(self.raw_dir, filename)
        else:
            filepath = os.path.join(self.processed_dir, filename)
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            df.to_csv(filepath, index=False)
            print(f"Dataset saved successfully to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving dataset: {str(e)}")
            return False
    
    def get_dataset_info(self, df: pd.DataFrame) -> dict:
        """
        Get information about the dataset.
        
        Args:
            df (pd.DataFrame): Dataset to analyze
            
        Returns:
            dict: Dictionary containing dataset information
        """
        info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict()
        }
        
        # If label column exists, get class distribution
        if 'label' in df.columns:
            info['class_distribution'] = df['label'].value_counts().to_dict()
        
        return info
    
    def validate_dataset(self, df: pd.DataFrame) -> Tuple[bool, list]:
        """
        Validate that the dataset has the required structure.
        
        Args:
            df (pd.DataFrame): Dataset to validate
            
        Returns:
            tuple: (is_valid, list_of_errors)
        """
        errors = []
        
        # Check if required columns exist
        required_columns = ['text', 'label']
        for col in required_columns:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")
        
        # Check for empty dataset
        if df.empty:
            errors.append("Dataset is empty")
        
        # Check for missing values in text column
        if 'text' in df.columns and df['text'].isnull().any():
            errors.append("Text column contains missing values")
        
        # Check for missing values in label column
        if 'label' in df.columns and df['label'].isnull().any():
            errors.append("Label column contains missing values")
        
        # Check label values (should be 0 and 1)
        if 'label' in df.columns:
            unique_labels = df['label'].unique()
            if not all(label in [0, 1] for label in unique_labels):
                errors.append("Label column should contain only 0 and 1 values")
        
        is_valid = len(errors) == 0
        return is_valid, errors
