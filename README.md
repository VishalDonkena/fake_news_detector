# Fake News Detector 🔍

A machine learning project for detecting fake news articles using deep learning techniques.

## Project Structure

```
fake_news_detector/
├── data/
│   ├── raw/           # Raw datasets
│   └── processed/     # Processed/cleaned data
├── notebooks/         # Jupyter notebooks
│   └── 02_model_training.ipynb
├── src/               # Source code
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── feature_extractor.py
│   ├── model_trainer.py
│   └── detector.py
├── models/            # Saved models and vectorizers
├── requirements.txt   # Python dependencies
└── main.py           # Main user interface
```

## Features

- **Text Preprocessing**: Clean and normalize text data using NLTK
- **Feature Extraction**: Convert text to numerical features using TF-IDF
- **Deep Learning Model**: Neural network with dense layers for classification
- **User Interface**: Command-line interface for real-time predictions
- **Training Pipeline**: Complete Jupyter notebook for model training

## Installation

1. Clone the repository:
```bash
git clone https://github.com/VishalDonkena/fake_news_detector.git
cd fake_news_detector
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

1. **Prepare your dataset**: Place your CSV file in `data/raw/` with columns `text` and `label` (0 for real news, 1 for fake news)

2. **Run the training notebook**:
```bash
jupyter notebook notebooks/02_model_training.ipynb
```

3. **Update the dataset path** in the notebook to match your filename

4. **Execute all cells** to train the model and save it to `models/`

### Using the Detector

Run the main interface:
```bash
python main.py
```

Then enter news articles to analyze them for fake news detection.

## Components

### Preprocessor (`src/preprocessor.py`)
- Converts text to lowercase
- Removes punctuation and numbers
- Tokenizes text
- Removes English stopwords
- Lemmatizes tokens

### Feature Extractor (`src/feature_extractor.py`)
- Uses TF-IDF vectorization
- Configurable parameters for feature extraction
- Handles both training and inference

### Model Trainer (`src/model_trainer.py`)
- Deep learning model with dense layers
- Adam optimizer with binary crossentropy loss
- Model saving and loading capabilities

### Detector (`src/detector.py`)
- Main prediction pipeline
- Orchestrates preprocessing, feature extraction, and prediction
- Provides confidence scores

### Data Loader (`src/data_loader.py`)
- Dataset loading and validation
- Support for raw and processed data
- Dataset information and validation utilities

## Model Architecture

The project supports two model architectures:

### 1. TF-IDF + Dense Layers (Default)
- Input: TF-IDF features (5000 dimensions)
- Hidden layers: 512 → 256 → 128 neurons with ReLU activation
- Dropout layers for regularization
- Output: Single neuron with sigmoid activation for binary classification

### 2. Bi-LSTM Architecture (Advanced)
- Input: Word embeddings (5000 vocabulary, 128 dimensions)
- Embedding layer: Converts words to dense vectors
- Bidirectional LSTM: Processes text in both directions for better context understanding
- Dropout layers for regularization
- Output: Single neuron with sigmoid activation for binary classification

The Bi-LSTM model is available in the `ModelTrainer` class and provides superior performance for text classification tasks.

## Requirements

- Python 3.9+
- TensorFlow 2.x
- scikit-learn
- pandas
- numpy
- nltk
- matplotlib
- seaborn
- jupyter

## Dataset Format

Your dataset should be a CSV file with the following structure:

```csv
text,label
"This is a real news article about important events.",0
"This is clearly fake news with misleading information.",1
```

- `text`: The news article content
- `label`: 0 for real news, 1 for fake news

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Built using TensorFlow/Keras for deep learning
- Uses NLTK for natural language processing
- scikit-learn for feature extraction and evaluation metrics
