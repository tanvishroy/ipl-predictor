import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score

# Load dataset
DATA_PATH = "data/processed/ipl_model_data_with_recent_form.csv"
df = pd.read_csv(DATA_PATH)

# Drop rows with missing targets
df = df.dropna(subset=[
    "Powerplay_Scores", "Middle_Overs_Scores", "Death_Overs_Scores",
    "First_Total", "Match_Winner"
])

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

for col in ["Team1", "Team2", "Toss_Winner", "Match_Winner"]:
    df[col] = df[col].replace(TEAM_RENAME_MAP)

# Encode categorical features
for col in ["Team1", "Team2", "Toss_Winner", "Venue", "Toss_Decision"]:
    df[col] = df[col].astype(str)

team_encoder = LabelEncoder()
venue_encoder = LabelEncoder()
toss_encoder = LabelEncoder()

team_encoder.fit(pd.concat([df["Team1"], df["Team2"], df["Toss_Winner"]]))
venue_encoder.fit(df["Venue"])
toss_encoder.fit(df["Toss_Decision"])

df["Team1"] = team_encoder.transform(df["Team1"])
df["Team2"] = team_encoder.transform(df["Team2"])
df["Toss_Winner"] = team_encoder.transform(df["Toss_Winner"])
df["Venue"] = venue_encoder.transform(df["Venue"])
df["Toss_Decision"] = toss_encoder.transform(df["Toss_Decision"])

# Encode match winner
winner_encoder = LabelEncoder()
df["Match_Winner"] = winner_encoder.fit_transform(df["Match_Winner"])

# Common input features
features = ["Team1", "Team2", "Venue", "Toss_Winner", "Toss_Decision", "Team1_Recent_Avg", "Team2_Recent_Avg"]

# Create models folder
os.makedirs("models", exist_ok=True)

# --- Regression Model Trainer ---
def train_regression(target_col, model_name):
    X = df[features]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    joblib.dump(model, f"models/{model_name}.pkl")
    print(f"✅ Trained {model_name} | RMSE: {rmse:.2f}")

# --- Classification Model Trainer ---
def train_winner_classifier():
    X = df[features]
    y = df["Match_Winner"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    joblib.dump(model, "models/winner_model.pkl")
    joblib.dump(winner_encoder, "models/winner_label_encoder.pkl")
    print(f"✅ Trained winner_model | Accuracy: {acc:.2f}")

# --- Train All Models ---
train_regression("Powerplay_Scores", "powerplay_model")
train_regression("Middle_Overs_Scores", "middle_model")
train_regression("Death_Overs_Scores", "death_model")
train_regression("First_Total", "total_model")
train_winner_classifier()

# Save encoders
joblib.dump(team_encoder, "models/team_encoder.pkl")
joblib.dump(venue_encoder, "models/venue_encoder.pkl")
joblib.dump(toss_encoder, "models/toss_encoder.pkl")
print("✅ Saved all label encoders and models.")
