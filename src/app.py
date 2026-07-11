import os
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="PSL Match Predictor", page_icon="🏏", layout="wide")
st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background-color: #13293D;
        border: 1px solid #22405A;
        border-radius: 10px;
        padding: 12px 16px;
    }
    div[data-testid="stMetricLabel"] {
        color: #A8B8C4;
    }
    h1 {
        border-bottom: 3px solid #F2B134;
        padding-bottom: 12px;
    }
            
</style>
""", unsafe_allow_html=True)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
feature_columns = joblib.load(os.path.join(BASE_DIR, "feature_columns.pkl"))
teams_list = joblib.load(os.path.join(BASE_DIR, "teams_list.pkl"))
venues_list = joblib.load(os.path.join(BASE_DIR, "venues_list.pkl"))

matches = pd.read_csv(os.path.join(BASE_DIR, "..", "data", "processed", "features.csv"), parse_dates=["date"])
matches = matches.sort_values("date")

st.title("🏏 PSL Match Outcome Predictor")
st.caption("Predicts the winner between two PSL teams using historical form and head-to-head record.")

# ---------- SIDEBAR: all inputs live here ----------
with st.sidebar:
    st.header("Match Setup")

    team1 = st.selectbox("Team A", teams_list, key="team1")
    team2_options = [t for t in teams_list if t != team1]
    team2 = st.selectbox("Team B", team2_options, key="team2")

    venue = st.selectbox("Venue", venues_list)
    toss_winner = st.radio("Toss Winner", [team1, team2])
    toss_decision = st.radio("Toss Decision", ["bat", "field"])

    predict_clicked = st.button("Predict Winner", use_container_width=True)

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

def get_current_form(team, matches_df, n=5):
    team_history = matches_df[
        (matches_df["team1"] == team) | (matches_df["team2"] == team)
    ].sort_values("date")
    if team_history.empty:
        return 0.5
    recent = team_history.tail(n)
    wins = (recent["winner"] == team).sum()
    return wins / len(recent)

def get_head_to_head(t1, t2, matches_df):
    pair_history = matches_df[
        ((matches_df["team1"] == t1) & (matches_df["team2"] == t2)) |
        ((matches_df["team1"] == t2) & (matches_df["team2"] == t1))
    ]
    if pair_history.empty:
        return 0.5, 0.5, pair_history
    t1_wins = (pair_history["winner"] == t1).sum()
    t1_pct = t1_wins / len(pair_history)
    return t1_pct, 1 - t1_pct, pair_history

team1_home_away = get_home_away(team1, venue)
team2_home_away = get_home_away(team2, venue)
team1_last5 = get_current_form(team1, matches)
team2_last5 = get_current_form(team2, matches)
team1_h2h, team2_h2h, pair_history = get_head_to_head(team1, team2, matches)

# ---------- MAIN AREA: context + result ----------
col1, col2 = st.columns(2)
with col1:
    st.metric(f"{team1} — Last 5 Form", f"{team1_last5*100:.0f}%")
with col2:
    st.metric(f"{team2} — Last 5 Form", f"{team2_last5*100:.0f}%")

st.divider()

if predict_clicked:
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

    team1_win_prob = probability[1]
    team2_win_prob = probability[0]
    winner = team1 if prediction == 1 else team2

    
    st.markdown(f"""
<div style="background-color:#13293D; border-left: 5px solid #F2B134; padding: 16px 20px; border-radius: 8px; margin-bottom: 16px;">
    <span style="color:#A8B8C4; font-size: 14px;">PREDICTED WINNER</span><br>
    <span style="color:#F2B134; font-size: 28px; font-weight: 700;">🏆 {winner}</span>
</div>
""", unsafe_allow_html=True)
    prob_df = pd.DataFrame({
        "Team": [team1, team2],
        "Win Probability": [team1_win_prob, team2_win_prob],
    }).set_index("Team")
    st.bar_chart(prob_df, horizontal=True, color=["#F2B134"])

    st.caption("Note: predictions are based on historical patterns and inherent randomness in cricket — treat this as a data-driven estimate, not a certainty.")

    st.divider()
    st.subheader(f"Head-to-Head History: {team1} vs {team2}")
    if pair_history.empty:
        st.write("These teams haven't played each other yet in the dataset.")
    else:
        recent_meetings = pair_history[["date", "venue", "winner"]].sort_values("date", ascending=False).head(5)
        recent_meetings["date"] = recent_meetings["date"].dt.strftime("%Y-%m-%d")
        st.dataframe(recent_meetings, hide_index=True, use_container_width=True)
        st.caption(f"{team1} has won {team1_h2h*100:.0f}% of {len(pair_history)} historical meetings.")
else:
    st.info("Set up the match in the sidebar and click **Predict Winner** to see the result.")