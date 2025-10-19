# Fake News Detector 🔍

A deep learning project to detect fake news articles using an LSTM (Long Short-Term Memory) neural network.

## Features

-   **Text Preprocessing**: Cleans and normalizes text data using NLTK (lowercase, remove punctuation, stopwords, lemmatization).
-   **LSTM Model**: Uses a Long Short-Term Memory (LSTM) network to understand the sequence and context of words in the text.
-   **Command-Line Interface**: Allows for real-time prediction of news articles.
-   **Script-Based Training**: A simple Python script to train the model from scratch.

## Model Architecture

This project uses an LSTM-based neural network for classification. The architecture is as follows:

1.  **Embedding Layer**: Converts words into dense vectors of a fixed size (128 dimensions). These embeddings capture the semantic meaning of the words.
2.  **LSTM Layer**: Processes the sequence of word embeddings to capture contextual information from the text.
3.  **Dropout Layers**: Used for regularization to prevent the model from overfitting.
4.  **Dense Layers**: Fully connected layers for classification, with a final `sigmoid` activation function to output a probability between 0 and 1.

This architecture was chosen to effectively learn from the sequential nature of text data and to address issues of "cheating" where the model might learn simple heuristics like article length.

## Dataset

The dataset used for training is a combination of two CSV files, `Fake.csv` and `True.csv`, which are preprocessed into a single `news.csv` file by the `scripts/prepare_data.py` script. The final dataset contains a `text` column with the article content and a `label` column (0 for real news, 1 for fake news).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/fake_news_detector.git
    cd fake_news_detector
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

There are two main steps: preparing the data and training the model, and then running the detector.

### 1. Prepare the Data

First, you need to combine the raw `Fake.csv` and `True.csv` files into a single dataset for training.

```bash
python scripts/prepare_data.py
```
This will create the `data/processed/news.csv` file.

### 2. Train the Model

To train the LSTM model from scratch, run the training script:

```bash
python scripts/train_from_notebook.py
```

This script will:
-   Load the processed data.
-   Preprocess the text.
-   Train the LSTM model.
-   Save the trained model to `models/fake_news_model.h5`.
-   Save the tokenizer to `models/tokenizer.pkl`.

### 3. Run the Detector

Once the model is trained, you can run the fake news detector:

```bash
python main.py
```

You will be prompted to enter a news article, and the model will predict whether it is "Fake News" or "Real News" along with a confidence score.

## Project Structure

```
fake_news_detector/
├── data/
│   ├── raw/           # Raw datasets (Fake.csv, True.csv)
│   └── processed/     # Processed data (news.csv)
├── src/               # Source code
│   ├── detector.py    # Class for prediction
│   └── preprocessor.py# Text preprocessing logic
├── models/            # Saved model and tokenizer
├── scripts/
│   ├── prepare_data.py # Script to prepare the dataset
│   └── train_from_notebook.py # Script to train the model
├── requirements.txt   # Python dependencies
└── main.py            # Main user interface
```

## License

This project is open source and available under the MIT License.