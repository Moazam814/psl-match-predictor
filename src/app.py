import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="PSL Match Predictor", page_icon="🏏")

model = joblib.load("model.pkl")
feature_columns = joblib.load("feature_columns.pkl")
teams_list = joblib.load("teams_list.pkl")
venues_list = joblib.load("venues_list.pkl")

# load historical match data so we can compute real form/h2h for the selected teams
matches = pd.read_csv("../data/processed/features.csv", parse_dates=["date"])
matches = matches.sort_values("date")

st.title("🏏 PSL Match Outcome Predictor")
st.caption("Predicts the winner between two PSL teams using historical form and head-to-head record.")
col1, col2 = st.columns(2)
with col1:
    team1 = st.selectbox("Team A", teams_list, key="team1")
with col2:
    team2_options = [t for t in teams_list if t != team1]
    team2 = st.selectbox("Team B", team2_options, key="team2")

venue = st.selectbox("Venue", venues_list)
toss_winner = st.radio("Toss Winner", [team1, team2])
toss_decision = st.radio("Toss Decision", ["bat", "field"])

home_cities = {
    "Lahore Qalandars": "Lahore",
    "Karachi Kings": "Karachi",
    "Islamabad United": "Rawalpindi",
    "Peshawar Zalmi": "Rawalpindi",
    "Multan Sultans": "Multan",
    "Quetta Gladiators": "Quetta",
}

def get_home_away(team, venue):
    home_city = home_cities.get(team)
    if home_city is None:
        return "unknown"
    return "home" if home_city in venue else "away"

team1_home_away = get_home_away(team1, venue)
team2_home_away = get_home_away(team2, venue)
def get_current_form(team, matches_df, n=5):
    """Look at this team's last n matches (any position) and return their win rate."""
    team_history = matches_df[
        (matches_df["team1"] == team) | (matches_df["team2"] == team)
    ].sort_values("date")

    if team_history.empty:
        return 0.5  # no history at all - neutral, same choice we made during training

    recent = team_history.tail(n)
    wins = (recent["winner"] == team).sum()
    return wins / len(recent)

def get_head_to_head(team1, team2, matches_df):
    """All historical matches between this exact pair, regardless of order."""
    pair_history = matches_df[
        ((matches_df["team1"] == team1) & (matches_df["team2"] == team2)) |
        ((matches_df["team1"] == team2) & (matches_df["team2"] == team1))
    ]

    if pair_history.empty:
        return 0.5, 0.5  # no meetings yet - neutral, same as training

    team1_wins = (pair_history["winner"] == team1).sum()
    team1_pct = team1_wins / len(pair_history)
    return team1_pct, 1 - team1_pct

team1_last5 = get_current_form(team1, matches)
team2_last5 = get_current_form(team2, matches)
team1_h2h, team2_h2h = get_head_to_head(team1, team2, matches)
if st.button("Predict Winner"):
    input_dict = {
        "team1_last5_win_pct": team1_last5,
        "team2_last5_win_pct": team2_last5,
        "team1_h2h_win_pct": team1_h2h,
        "team2_h2h_win_pct": team2_h2h,
        f"toss_decision_{toss_decision}": 1,
        f"team1_home_away_{team1_home_away}": 1,
        f"team2_home_away_{team2_home_away}": 1,
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    winner = team1 if prediction == 1 else team2
    win_prob = probability[1] if prediction == 1 else probability[0]

    st.subheader(f"Predicted Winner: {winner}")
    st.metric("Win Probability", f"{win_prob*100:.1f}%")

    st.caption("Note: predictions are based on historical patterns and inherent randomness in cricket — treat this as a data-driven estimate, not a certainty.")