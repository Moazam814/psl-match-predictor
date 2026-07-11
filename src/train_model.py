import pandas as pd

df = pd.read_csv("data/processed/features.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# check how many matches per season, so we pick a sensible cutoff for the test set
print(df["season"].value_counts().sort_index())
train_df = df[~df["season"].isin(["2025", "2026"])].copy()
test_df = df[df["season"].isin(["2025", "2026"])].copy()

print(f"Train: {train_df.shape[0]} matches ({train_df['season'].min()} to {train_df['season'].max()})")
print(f"Test: {test_df.shape[0]} matches (seasons: {sorted(test_df['season'].unique())})")
print(f"Train date range: {train_df['date'].min()} to {train_df['date'].max()}")
print(f"Test date range: {test_df['date'].min()} to {test_df['date'].max()}")