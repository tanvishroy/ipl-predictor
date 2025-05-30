import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import base64

# --- Streamlit Page Config ---
st.set_page_config(page_title="IPL Predictor", page_icon="🏏", layout="centered")

# --- Encode Background Image ---
def encode_bg(file_path):
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# --- Inject Custom CSS ---
def local_css():
    st.markdown(f"""
        <style>
        body {{
            background-image: url("data:image/png;base64,{encode_bg('assets/background.png')}");
            background-size: cover;
            background-attachment: fixed;
        }}
        .stButton > button {{
            background-color: #f0ad4e;
            color: white;
            font-weight: bold;
            transition: 0.3s;
            border-radius: 8px;
            padding: 0.5em 1em;
        }}
        .stButton > button:hover {{
            background-color: #ec971f;
            color: white;
            transform: scale(1.05);
            cursor: pointer;
        }}
        </style>
    """, unsafe_allow_html=True)

local_css()

# --- Load Models ---
model_powerplay = joblib.load("models/powerplay_model.pkl")
model_middle = joblib.load("models/middle_model.pkl")
model_death = joblib.load("models/death_model.pkl")
model_total = joblib.load("models/total_model.pkl")
model_winner = joblib.load("models/winner_model.pkl")
winner_encoder = joblib.load("models/winner_label_encoder.pkl")
team_encoder = joblib.load("models/team_encoder.pkl")
venue_encoder = joblib.load("models/venue_encoder.pkl")
toss_encoder = joblib.load("models/toss_encoder.pkl")

# Normalize old team names
TEAM_RENAME_MAP = {
    "Delhi Daredevils": "Delhi Capitals",
    "Deccan Chargers": "Sunrisers Hyderabad",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "Pune Warriors": "Pune Warriors India",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bangaluru",
    "Gujarat Lions": "Gujarat Titans"
}

# --- Input Vector Builder ---
def get_input_vector(team1, team2, venue, toss_winner, toss_decision, team1_avg, team2_avg):
    return pd.DataFrame([{
        "Team1": team_encoder.transform([team1])[0],
        "Team2": team_encoder.transform([team2])[0],
        "Venue": venue_encoder.transform([venue])[0],
        "Toss_Winner": team_encoder.transform([toss_winner])[0],
        "Toss_Decision": toss_encoder.transform([toss_decision])[0],
        "Team1_Recent_Avg": team1_avg,
        "Team2_Recent_Avg": team2_avg
    }])

# --- Prediction Functions ---
def predict_scores(input_df):
    pp = model_powerplay.predict(input_df)[0]
    mid = model_middle.predict(input_df)[0]
    death = model_death.predict(input_df)[0]
    total = model_total.predict(input_df)[0]
    return pp, mid, death, total

def predict_winner(input_df):
    pred = model_winner.predict(input_df)[0]
    return winner_encoder.inverse_transform([pred])[0]

# --- App UI ---
st.markdown("## 🏏 IPL Match Predictor", unsafe_allow_html=True)
st.markdown("Enter match details to get predictions based on real IPL data.")

team_list = [
    "Chennai Super Kings", "Mumbai Indians", "Royal Challengers Bangalore",
    "Delhi Capitals", "Kolkata Knight Riders", "Sunrisers Hyderabad",
    "Rajasthan Royals", "Punjab Kings", "Lucknow Super Giants", "Gujarat Titans"
]
venue_list = [
    "Wankhede Stadium, Mumbai",
    "M. A. Chidambaram Stadium, Chennai",
    "Arun Jaitley Stadium, Delhi",
    "M. Chinnaswamy Stadium, Bengaluru",
    "Eden Gardens, Kolkata",
    "Narendra Modi Stadium, Ahmedabad",
    "Sawai Mansingh Stadium, Jaipur",
    "Rajiv Gandhi International Stadium, Hyderabad",
    "Punjab Cricket Association IS Bindra Stadium, Mohali",
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow",
    "Dr DY Patil Sports Academy, Mumbai",
    "Maharashtra Cricket Association Stadium, Pune",
    "Barsapara Cricket Stadium, Guwahati",
    "Himachal Pradesh Cricket Association Stadium, Dharamsala"
]

t1 = st.selectbox("Team 1", team_list)
t2 = st.selectbox("Team 2", [t for t in team_list if t != t1])
venue = st.selectbox("Venue", venue_list)
toss_option = st.selectbox("Toss Winner (optional)", ["Predict", t1, t2])
toss_decision = st.selectbox("Toss Decision", ["bat", "field"])

# Dummy recent averages (you could calculate dynamically later)
team1_avg = 160
team2_avg = 155

if st.button("Predict Match"):
    toss_winner = t1 if toss_option == "Predict" else toss_option

    # Apply name normalization to selected teams
    t1 = TEAM_RENAME_MAP.get(t1, t1)
    t2 = TEAM_RENAME_MAP.get(t2, t2)
    toss_winner = TEAM_RENAME_MAP.get(toss_winner, toss_winner)

    input_df_1st = get_input_vector(t1, t2, venue, toss_winner, toss_decision, team1_avg, team2_avg)
    pp1, mid1, death1, total1 = predict_scores(input_df_1st)

    input_df_2nd = get_input_vector(t2, t1, venue, toss_winner, toss_decision, team2_avg, team1_avg)
    pp2, mid2, death2, total2 = predict_scores(input_df_2nd)

    winner = predict_winner(input_df_1st)

    st.subheader("📊 Prediction Results")
    st.write(f"**Toss Winner:** {toss_winner}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### {t1}")
        st.write(f"Powerplay: {pp1:.1f} runs")
        st.write(f"Middle Overs: {mid1:.1f} runs")
        st.write(f"Death Overs: {death1:.1f} runs")
        st.success(f"Total: {total1:.1f} runs")
    with col2:
        st.markdown(f"### {t2}")
        st.write(f"Powerplay: {pp2:.1f} runs")
        st.write(f"Middle Overs: {mid2:.1f} runs")
        st.write(f"Death Overs: {death2:.1f} runs")
        st.success(f"Total: {total2:.1f} runs")

    st.markdown("---")
    st.subheader("🏆 Predicted Match Winner")
    st.success(f"**{winner}**")