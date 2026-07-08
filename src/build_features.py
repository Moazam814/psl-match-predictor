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