import pandas as pd

df = pd.read_csv("data/processed/matches.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

print(df.shape)
print(df.head())
# reshape: one row per team per match, so we can compute rolling stats per team
team1_rows = df[["date", "season", "team1", "winner"]].copy()
team1_rows = team1_rows.rename(columns={"team1": "team"})
team1_rows["won"] = (team1_rows["team"] == team1_rows["winner"]).astype(int)

team2_rows = df[["date", "season", "team2", "winner"]].copy()
team2_rows = team2_rows.rename(columns={"team2": "team"})
team2_rows["won"] = (team2_rows["team"] == team2_rows["winner"]).astype(int)

team_matches = pd.concat([team1_rows, team2_rows], ignore_index=True)
team_matches = team_matches.sort_values(["team", "date"]).reset_index(drop=True)

print(team_matches.head(10))
# rolling win rate over the PREVIOUS 5 matches (shift(1) excludes the current match itself)
team_matches["last5_win_pct"] = (
    team_matches.groupby("team")["won"]
    .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
)

print(team_matches.head(10))
# merge team1's form back onto the original match rows
df = df.merge(
    team_matches[["date", "team", "last5_win_pct"]],
    left_on=["date", "team1"], right_on=["date", "team"],
    how="left"
).rename(columns={"last5_win_pct": "team1_last5_win_pct"}).drop(columns="team")

# merge team2's form back onto the original match rows
df = df.merge(
    team_matches[["date", "team", "last5_win_pct"]],
    left_on=["date", "team2"], right_on=["date", "team"],
    how="left"
).rename(columns={"last5_win_pct": "team2_last5_win_pct"}).drop(columns="team")

print(df[["date", "team1", "team2", "team1_last5_win_pct", "team2_last5_win_pct"]].head(10))
def compute_head_to_head(df):
    h2h_team1 = []
    h2h_team2 = []

    for idx, row in df.iterrows():
        current_date = row["date"]
        t1, t2 = row["team1"], row["team2"]

        # all previous matches between these two teams, in either order
        past_matches = df[
            (df["date"] < current_date) &
            (((df["team1"] == t1) & (df["team2"] == t2)) |
             ((df["team1"] == t2) & (df["team2"] == t1)))
        ]

        if len(past_matches) == 0:
            h2h_team1.append(None)
            h2h_team2.append(None)
        else:
            t1_wins = (past_matches["winner"] == t1).sum()
            h2h_team1.append(t1_wins / len(past_matches))
            h2h_team2.append(1 - (t1_wins / len(past_matches)))

    df["team1_h2h_win_pct"] = h2h_team1
    df["team2_h2h_win_pct"] = h2h_team2
    return df

df = compute_head_to_head(df)
print(df[["date", "team1", "team2", "team1_h2h_win_pct", "team2_h2h_win_pct"]].head(15))
team_home_city = {
    "Lahore Qalandars": "Lahore",
    "Karachi Kings": "Karachi",
    "Islamabad United": "Rawalpindi",
    "Peshawar Zalmi": "Rawalpindi",
    "Multan Sultans": "Multan",
    "Quetta Gladiators": "Quetta",
}

def get_home_away(row, team_col):
    team = row[team_col]
    home_city = team_home_city.get(team)
    if home_city is None:
        return "unknown"
    return "home" if home_city in str(row["venue"]) else "away"

df["team1_home_away"] = df.apply(lambda r: get_home_away(r, "team1"), axis=1)
df["team2_home_away"] = df.apply(lambda r: get_home_away(r, "team2"), axis=1)

print(df[["venue", "team1", "team1_home_away", "team2", "team2_home_away"]].drop_duplicates().head(15))
# for first-ever matches/matchups, there's no history yet - fill with a neutral 0.5 (50/50, no info)
feature_cols = [
    "team1_last5_win_pct", "team2_last5_win_pct",
    "team1_h2h_win_pct", "team2_h2h_win_pct",
]
for col in feature_cols:
    df[col] = df[col].fillna(0.5)
# create binary target: did team1 win?
df["team1_won"] = (df["winner"] == df["team1"]).astype(int)
# sanity check: is there any accidental bias in how team1/team2 were assigned?
print(df["team1_won"].value_counts(normalize=True))
# save the final feature table - this is what modeling will load
df.to_csv("data/processed/features.csv", index=False)
print(f"Saved final feature table: {df.shape[0]} rows, {df.shape[1]} columns")
print(df.columns.tolist())