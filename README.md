import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import base64
import time

# --- Streamlit Page Config ---

st.set\_page\_config(page\_title="IPL Predictor", page\_icon="🏏", layout="centered")

# --- Encode Background Image ---

def encode\_bg(file\_path):
with open(file\_path, "rb") as image\_file:
return base64.b64encode(image\_file.read()).decode()

# --- Inject Custom CSS ---

def local\_css():
st.markdown(f""" <style>
body {{
background-image: url("data\:image/png;base64,{encode\_bg('assets/background.png')}");
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

local\_css()

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

# --- Normalize Team Names ---

TEAM\_RENAME\_MAP = {
"Delhi Daredevils": "Delhi Capitals",
"Deccan Chargers": "Sunrisers Hyderabad",
"Rising Pune Supergiants": "Rising Pune Supergiant",
"Pune Warriors": "Pune Warriors India",
"Kings XI Punjab": "Punjab Kings"
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

# --- Load IPL Data for Recent Averages ---

df\_data = pd.read\_csv("data/processed/ipl\_model\_data\_with\_recent\_form.csv")

@st.cache\_data
def get\_recent\_avg\_scores(team\_name, df):
recent\_matches = df\[
(df\["Team1"] == team\_name) | (df\["Team2"] == team\_name)
].sort\_values(by="Date", ascending=False).head(5)
return recent\_matches\["First\_Total"].mean()

# --- App UI ---

st.markdown("## 🏏 IPL Match Predictor", unsafe\_allow\_html=True)
st.markdown("Enter match details to get predictions based on real IPL data.")

team\_list = team\_encoder.classes\_.tolist()
venue\_list = venue\_encoder.classes\_.tolist()

t1 = st.selectbox("Team 1", team\_list)
t2 = st.selectbox("Team 2", \[t for t in team\_list if t != t1])
venue = st.selectbox("Venue", venue\_list)
toss\_option = st.selectbox("Toss Winner (optional)", \["Predict", t1, t2])
toss\_decision = st.selectbox("Toss Decision", \["bat", "field"])

if st.button("Predict Match"):
with st.spinner("Predicting match outcome..."):
time.sleep(1.5)

```
    # Normalize names
    t1 = TEAM_RENAME_MAP.get(t1, t1)
    t2 = TEAM_RENAME_MAP.get(t2, t2)
    toss_winner = t1 if toss_option == "Predict" else toss_option
    toss_winner = TEAM_RENAME_MAP.get(toss_winner, toss_winner)

    team1_avg = get_recent_avg_scores(t1, df_data)
    team2_avg = get_recent_avg_scores(t2, df_data)

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
    st.progress(min(int(pp1 / 70 * 100), 100), text=f"Powerplay: {pp1:.1f} runs")
    st.progress(min(int(mid1 / 80 * 100), 100), text=f"Middle Overs: {mid1:.1f} runs")
    st.progress(min(int(death1 / 70 * 100), 100), text=f"Death Overs: {death1:.1f} runs")
    st.success(f"Total: {total1:.1f} runs")
with col2:
    st.markdown(f"### {t2}")
    st.progress(min(int(pp2 / 70 * 100), 100), text=f"Powerplay: {pp2:.1f} runs")
    st.progress(min(int(mid2 / 80 * 100), 100), text=f"Middle Overs: {mid2:.1f} runs")
    st.progress(min(int(death2 / 70 * 100), 100), text=f"Death Overs: {death2:.1f} runs")
    st.success(f"Total: {total2:.1f} runs")

st.markdown("---")
st.subheader("🏆 Predicted Match Winner")
st.success(f"**{winner}**")
```
