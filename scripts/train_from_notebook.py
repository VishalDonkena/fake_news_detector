import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
import os
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Add src directory to path
sys.path.append("/Users/vishaldonkena/Code/fake_news_detector/src")

from preprocessor import Preprocessor

# Load the dataset
dataset_path = "/Users/vishaldonkena/Code/fake_news_detector/data/processed/news.csv"
df = pd.read_csv(dataset_path)

# Preprocessing
preprocessor = Preprocessor()
if "text" in df.columns:
    df["cleaned_text"] = df["text"].apply(preprocessor.clean_text)
else:
    print("Error: 'text' column not found in the dataset.")
    sys.exit(1)

# Tokenization and Padding
max_words = 10000
max_length = 500  # Increased max_length

tokenizer = Tokenizer(num_words=max_words, oov_token="<unk>")
tokenizer.fit_on_texts(df["cleaned_text"].values)

X = tokenizer.texts_to_sequences(df["cleaned_text"].values)
X = pad_sequences(X, maxlen=max_length)

y = df["label"].values

# Save the tokenizer
tokenizer_path = "/Users/vishaldonkena/Code/fake_news_detector/models/tokenizer.pkl"
os.makedirs("/Users/vishaldonkena/Code/fake_news_detector/models", exist_ok=True)
joblib.dump(tokenizer, tokenizer_path)
print(f"Tokenizer saved to {tokenizer_path}")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model Training
vocab_size = len(tokenizer.word_index) + 1

model = Sequential(
    [
        Embedding(input_dim=vocab_size, output_dim=128, input_length=max_length),
        LSTM(128, return_sequences=False),
        Dropout(0.5),  # Increased dropout
        Dense(64, activation="relu"),
        Dropout(0.5),  # Increased dropout
        Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_data=(X_test, y_test),
    verbose=1,
)

# Save the trained model
model_path = "/Users/vishaldonkena/Code/fake_news_detector/models/fake_news_model.h5"
model.save(model_path)
print(f"Model saved to {model_path}")
