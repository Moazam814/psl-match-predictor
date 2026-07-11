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
# columns we'll actually feed into the model
categorical_cols = ["team1", "team2", "venue", "toss_winner", "toss_decision", "team1_home_away", "team2_home_away"]
numeric_cols = ["team1_last5_win_pct", "team2_last5_win_pct", "team1_h2h_win_pct", "team2_h2h_win_pct"]

target_col = "team1_won"

# combine train+test temporarily just for consistent one-hot columns, then split back apart
combined = pd.concat([train_df, test_df], keys=["train", "test"])
combined_encoded = pd.get_dummies(combined, columns=categorical_cols)

X_train = combined_encoded.loc["train"][numeric_cols + [c for c in combined_encoded.columns if c not in df.columns]]
X_test = combined_encoded.loc["test"][numeric_cols + [c for c in combined_encoded.columns if c not in df.columns]]
y_train = train_df[target_col]
y_test = test_df[target_col]

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(X_train.columns.tolist()[:15])