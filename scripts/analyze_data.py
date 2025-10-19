import pandas as pd
import numpy as np
import sys

# Add src directory to path
sys.path.append("/Users/vishaldonkena/Code/fake_news_detector/src")
from preprocessor import Preprocessor

# Load the dataset
dataset_path = "/Users/vishaldonkena/Code/fake_news_detector/data/processed/news.csv"
df = pd.read_csv(dataset_path)

# Add a column with the length of the text
if "text" in df.columns:
    df["text_length"] = df["text"].str.len()
else:
    print("Error: 'text' column not found in the dataset.")
    sys.exit(1)

# Group by label and calculate the average length
average_length = df.groupby("label")["text_length"].mean()

print("Average article length by category:")
print(average_length)

# Map labels to human-readable names
label_map = {1: "Fake News", 0: "Real News"}
print("\nCategory mapping:")
print(label_map)
