# 🏏 IPL Match Predictor

An AI-powered IPL prediction web app built using **Streamlit** and trained on **ball-by-ball IPL data (2008–2025)**. It predicts:

- Powerplay, Middle, and Death Overs scores
- Total scores for both teams
- Toss winner (optional or predicted)
- Final match winner

---

## 🚀 Live App

👉 [https://ipl-predictor.streamlit.app](https://ipl-predictor.streamlit.app)  
(*Once deployed via Streamlit Cloud*)

---

## 🎯 Features

- 📥 User inputs: Team1, Team2, Stadium, Toss winner (optional)
- 🔮 Predictions using real ML models:
  - Powerplay (0–6 overs)
  - Middle Overs (7–15 overs)
  - Death Overs (16–20 overs)
  - Total Score (1st innings)
  - Match Winner
- 🎨 Stylish UI with:
  - Background image
  - Hover effects on buttons
  - Score progress bars
  - Optional team avatars/logo support
- ⚡ Fast deployment with Streamlit Cloud

---

## 📂 Folder Structure

ipl-predictor/
├── app.py # Streamlit frontend
├── train_model.py # Model training pipeline
├── requirements.txt # Python dependencies
├── README.md
│
├── data/
│ └── processed/
│ └── ipl_model_data_with_recent_form.csv
│
├── models/ # Saved ML models
│ ├── powerplay_model.pkl
│ ├── middle_model.pkl
│ ├── death_model.pkl
│ ├── total_model.pkl
│ ├── winner_model.pkl
│ ├── winner_label_encoder.pkl
│ ├── team_encoder.pkl
│ ├── venue_encoder.pkl
│ └── toss_encoder.pkl
│
└── assets/ # Optional images like logo or background
├── logo.png
└── background.jpg

yaml
Copy
Edit

---

## 🧠 Machine Learning

Models trained with **Random Forest** using:
- Features: `Team1`, `Team2`, `Venue`, `Toss_Winner`, `Toss_Decision`, `Recent_Form`
- Targets:
  - Phase scores: `Powerplay_Scores`, `Middle_Overs_Scores`, `Death_Overs_Scores`
  - `First_Total`
  - `Match_Winner` (classification)

---

## 📦 Installation

```bash
git clone https://github.com/tanvishroy/ipl-predictor.git
cd ipl-predictor
pip install -r requirements.txt
▶️ Run Locally
bash
Copy
Edit
streamlit run app.py
Make sure the models are present in the /models directory and that background/logo images (if used) are correctly referenced.

🌐 Deploy on Streamlit Cloud
Push code to GitHub

Go to https://streamlit.io/cloud

Create a new app:

Repo: tanvishroy/ipl-predictor

Branch: main

File: app.py

Click Deploy ✅

🙋‍♂️ Author
Tanvish Roy
Made with 💙 for the love of cricket and ML.
GitHub · LinkedIn

📄 License
This project is licensed under the MIT License.

python
Copy
Edit

---

Let me know if you'd like:
- A `LICENSE` file added
- Badges (GitHub stars, Streamlit status, etc.)
- A live screenshot or banner for the top of the README

I'll help you polish it all for portfolio-ready quality!