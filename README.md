import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import base64

# --- Streamlit Page Config ---

st.set\_page\_config(page\_title="IPL Predictor", page\_icon="🏏", layout="centered")

# --- Encode and Set Background Image ---

def set\_background(image\_path):
with open(image\_path, "rb") as image\_file:
encoded = base64.b64encode(image\_file.read()).decode()
st.markdown(f""" <style>
body {{
background-image: url("data\:image/png;base64,{encoded}");
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
.stButton > button\:hover {{
background-color: #ec971f;
color: white;
transform: scale(1.05);
cursor: pointer;
}} </style>
""", unsafe\_allow\_html=True)

set\_background("assets/background.png")

# --- Load Models ---

model\_powerplay = joblib.load("models/powerplay\_model.pkl")
model\_middle = joblib.load("models/middle\_model.pkl")
model\_death = joblib.load("models/death\_model.pkl")
model\_total = joblib.load("models/total\_model.pkl")
model\_winner = joblib.load("models/winner\_model.pkl")
winner\_encoder = joblib.load("models/winner\_label\_encoder.pkl")
team\_encoder = joblib.load("models/team\_encoder.pkl")
venue\_encoder = joblib.load("models/venue\_encoder.pkl")
toss\_encoder = joblib.load("models/toss\_encoder.pkl")

# Normalize old team names

TEAM\_RENAME\_MAP = {
"Delhi Daredevils": "Delhi Capitals",
"Deccan Chargers": "Sunrisers Hyderabad",
"Rising Pune Supergiants": "Rising Pune Supergiant",
"Pune Warriors": "Pune Warriors India",
"Kings XI Punjab": "Punjab Kings",
"Royal Challengers Bangalore": "Royal Challengers Bangaluru",
"Gujarat Lions": "Gujarat Titans"
}

# --- Input Vector Builder ---

def get\_input\_vector(team1, team2, venue, toss\_winner, toss\_decision, team1\_avg, team2\_avg):
return pd.DataFrame(\[{
"Team1": team\_encoder.transform(\[team1])\[0],
"Team2": team\_encoder.transform(\[team2])\[0],
"Venue": venue\_encoder.transform(\[venue])\[0],
"Toss\_Winner": team\_encoder.transform(\[toss\_winner])\[0],
"Toss\_Decision": toss\_encoder.transform(\[toss\_decision])\[0],
"Team1\_Recent\_Avg": team1\_avg,
"Team2\_Recent\_Avg": team2\_avg
}])

# --- Prediction Functions ---

def predict\_scores(input\_df):
pp = model\_powerplay.predict(input\_df)\[0]
mid = model\_middle.predict(input\_df)\[0]
death = model\_death.predict(input\_df)\[0]
total = model\_total.predict(input\_df)\[0]
return pp, mid, death, total

def predict\_winner(input\_df):
pred = model\_winner.predict(input\_df)\[0]
return winner\_encoder.inverse\_transform(\[pred])\[0]

# --- App UI ---

st.markdown("## 🏏 IPL Match Predictor", unsafe\_allow\_html=True)
st.markdown("Enter match details to get predictions based on real IPL data.")

team\_list = \[
"Chennai Super Kings", "Mumbai Indians", "Royal Challengers Bangalore",
"Delhi Capitals", "Kolkata Knight Riders", "Sunrisers Hyderabad",
"Rajasthan Royals", "Punjab Kings", "Lucknow Super Giants", "Gujarat Titans"
]
venue\_list = \[
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

t1 = st.selectbox("Team 1", team\_list)
t2 = st.selectbox("Team 2", \[t for t in team\_list if t != t1])
venue = st.selectbox("Venue", venue\_list)
toss\_option = st.selectbox("Toss Winner (optional)", \["Predict", t1, t2])
toss\_decision = st.selectbox("Toss Decision", \["bat", "field"])

# Dummy recent averages (you could calculate dynamically later)

team1\_avg = 160
team2\_avg = 155

if st.button("Predict Match"):
toss\_winner = t1 if toss\_option == "Predict" else toss\_option

```
# Apply name normalization to selected teams
t1 = TEAM_RENAME_MAP.get(t1, t1)
t2 = TEAM_RENAME_MAP.get(t2, t2)
toss_winner = TEAM_RENAME_MAP.get(toss_winner, toss_winner)

input_df_1st = get_input_vector(t1, t2, venue, toss_winner, toss_decision, team1_avg, team2_avg)
pp1, mid1, death1, total1 = predict_scores(input_df_1st)

input_df_2nd = get_input_vector(t2, t1, venue, toss_winner, toss_decision, team2_avg, team1_avg)
pp2, mid2, death2, total2 = predict_scores(input_df_2nd)

winner = predict_winner(input_df_1st)
if winner not in [t1, t2]:
    winner = t1 if total1 > total2 else t2

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
```