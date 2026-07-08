import json
import os
import pandas as pd

RAW_DATA_DIR = "data/raw"
OUTPUT_PATH = "data/processed/matches.csv"
def parse_match(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)

    info = data["info"]

    # some old/odd files might be missing outcome (e.g. abandoned matches) - skip those safely
    if "winner" not in info.get("outcome", {}):
        return None

    team1, team2 = info["teams"]
    toss_winner = info["toss"]["winner"]
    toss_decision = info["toss"]["decision"]  # "bat" or "field"

    # derive who actually batted first - this isn't given directly, we have to work it out
    if toss_decision == "bat":
        bat_first_team = toss_winner
    else:
        bat_first_team = team2 if toss_winner == team1 else team1

    match_row = {
        "date": info["dates"][0],          # first date is enough even for 2-day matches
        "season": info["season"],
        "venue": info["venue"],
        "team1": team1,
        "team2": team2,
        "toss_winner": toss_winner,
        "toss_decision": toss_decision,
        "bat_first_team": bat_first_team,
        "winner": info["outcome"]["winner"],
    }
    return match_row
if __name__ == "__main__":
    all_matches = []
    skipped = 0

    for filename in os.listdir(RAW_DATA_DIR):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(RAW_DATA_DIR, filename)
        row = parse_match(filepath)

        if row is None:
            skipped += 1
            continue

        all_matches.append(row)

    df = pd.DataFrame(all_matches)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Parsed {len(df)} matches, skipped {skipped} (no result/abandoned).")
    print(df.head())