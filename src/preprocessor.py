import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class Preprocessor:
    """Class for cleaning and preprocessing text data."""
    
    def __init__(self):
        """Initialize the preprocessor with required NLTK components."""
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text):
        """
        Clean and preprocess text data.
        
        Args:
            text (str): Raw text to be cleaned
            
        Returns:
            str: Cleaned text as a single string
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize the text
        tokens = word_tokenize(text)
        
        # Remove English stopwords
        tokens = [token for token in tokens if token not in self.stop_words]
        
        # Lemmatize the tokens
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Join tokens back into a single string
        cleaned_text = ' '.join(tokens)
        
        return cleaned_text
