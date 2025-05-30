# 🏏 IPL Match Predictor

An AI-powered web app that predicts:

* Powerplay, Middle Overs, and Death Overs scores
* Total score for both teams
* Toss winner (optional)
* Final match winner

All predictions are based on 2008–2025 IPL ball-by-ball data, recent team performance, venue, and toss dynamics.

---

## 🚀 Live App

🔗 [https://ipl-predictor.streamlit.app](https://ipl-predictor.streamlit.app) *(Once deployed)*

## 📂 Project Structure

```
ipl-predictor/
├── app.py                    # Streamlit frontend
├── train_model.py           # Model training pipeline
├── requirements.txt         # Python dependencies
├── README.md
│
├── data/
│   └── processed/
│       └── ipl_model_data_with_recent_form.csv
│
├── models/                  # Saved models
│   ├── powerplay_model.pkl
│   ├── middle_model.pkl
│   ├── death_model.pkl
│   ├── total_model.pkl
│   ├── winner_model.pkl
│   ├── winner_label_encoder.pkl
│   └── encoders (team, venue, toss)
```

---

## 💡 Features

* **Match input**: Choose two teams, venue, optional toss winner
* **Score prediction**: Estimates each phase of the game
* **Outcome prediction**: Final match winner
* **Based on real IPL data (2008–2025)**

---

## 🧠 Machine Learning

* Models: Random Forest (regression & classification)
* Trained on:

  * Team1, Team2
  * Venue
  * Toss Winner, Toss Decision
  * Recent team performance (avg. score from last 5 matches)

---

## 📦 Requirements

```bash
pip install -r requirements.txt
```

## ▶️ Run Locally

```bash
streamlit run app.py
```

---

## 🧠 Author

Made with ❤️ by \[Your Name]

## 📄 License

MIT License — free to use, fork, and build on.
