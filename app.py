import os
import streamlit as st
import pandas as pd
import joblib

# Load pre-trained models (placeholders for now)
# model_powerplay = joblib.load("models/powerplay_model.pkl")
# model_middle = joblib.load("models/middle_model.pkl")
# model_death = joblib.load("models/death_model.pkl")
# model_total = joblib.load("models/total_model.pkl")
# model_winner = joblib.load("models/winner_model.pkl")

# --- Fake prediction functions ---
# Replace these with model.predict(input_vector) later
def predict_phase_score(phase, team, venue, opponent, recent_avg):
    import random
    if phase == "powerplay":
        return round(recent_avg * 0.3 + random.randint(30, 50), 1)
    elif phase == "middle":
        return round(recent_avg * 0.4 + random.randint(40, 60), 1)
    elif phase == "death":
        return round(recent_avg * 0.3 + random.randint(30, 50), 1)

def predict_winner(team1_score, team2_score, team1, team2):
    if team1_score > team2_score:
        return team1
    elif team2_score > team1_score:
        return team2
    else:
        return "Tie"

# --- UI ---
st.title("🏏 IPL Match Predictor")

st.markdown("""
Enter match details below and get predicted scores for Powerplay, Middle Overs, Death Overs,
Total Scores for both teams, and the predicted Winner!
""")

team_list = ['CSK', 'MI', 'RCB', 'KKR', 'DC', 'RR', 'SRH', 'PBKS', 'LSG', 'GT']  # Sample team list
venue_list = ['Wankhede', 'Chinnaswamy', 'Eden Gardens', 'Narendra Modi Stadium']

team1 = st.selectbox("Team 1", team_list)
team2 = st.selectbox("Team 2", [t for t in team_list if t != team1])
venue = st.selectbox("Venue", venue_list)
toss_winner = st.selectbox("Toss Winner (optional)", ["Predict", team1, team2])

if st.button("Predict Match"):
    # Fake recent form values (replace with actual lookup later)
    team1_recent_avg = 160
    team2_recent_avg = 155

    # Predict scores for both teams
    t1_powerplay = predict_phase_score("powerplay", team1, venue, team2, team1_recent_avg)
    t1_middle = predict_phase_score("middle", team1, venue, team2, team1_recent_avg)
    t1_death = predict_phase_score("death", team1, venue, team2, team1_recent_avg)
    t1_total = t1_powerplay + t1_middle + t1_death

    t2_powerplay = predict_phase_score("powerplay", team2, venue, team1, team2_recent_avg)
    t2_middle = predict_phase_score("middle", team2, venue, team1, team2_recent_avg)
    t2_death = predict_phase_score("death", team2, venue, team1, team2_recent_avg)
    t2_total = t2_powerplay + t2_middle + t2_death

    winner = predict_winner(t1_total, t2_total, team1, team2)
    predicted_toss = toss_winner if toss_winner != "Predict" else team1  # Dummy logic

    # Display Results
    st.subheader("📊 Prediction Results")
    st.write(f"**Toss Winner:** {predicted_toss}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### {team1}")
        st.write(f"Powerplay: {t1_powerplay} runs")
        st.write(f"Middle Overs: {t1_middle} runs")
        st.write(f"Death Overs: {t1_death} runs")
        st.success(f"Total: {t1_total} runs")
    with col2:
        st.markdown(f"### {team2}")
        st.write(f"Powerplay: {t2_powerplay} runs")
        st.write(f"Middle Overs: {t2_middle} runs")
        st.write(f"Death Overs: {t2_death} runs")
        st.success(f"Total: {t2_total} runs")

    st.markdown("---")
    st.subheader("🏆 Predicted Match Winner")
    st.success(f"**{winner}**")
