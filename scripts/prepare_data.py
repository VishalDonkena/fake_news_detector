import pandas as pd

# Load the datasets
df_fake = pd.read_csv("/Users/vishaldonkena/Code/fake_news_detector/data/raw/Fake.csv")
df_true = pd.read_csv("/Users/vishaldonkena/Code/fake_news_detector/data/raw/True.csv")

# Add labels
df_fake["label"] = 1
df_true["label"] = 0

# Combine the dataframes
df = pd.concat([df_fake, df_true], ignore_index=True)

# Save the processed data
df.to_csv(
    "/Users/vishaldonkena/Code/fake_news_detector/data/processed/news.csv", index=False
)

print("Data prepared and saved to data/processed/news.csv")
